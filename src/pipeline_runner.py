"""Per-camera ANPR pipeline runner.

Processes video frames through the full detection → tracking → OCR → DB →
alerts → analytics pipeline for a single camera feed.

Workers deliberately do **not** create an isolated cross-camera fusion
engine: sightings are written to the shared SQLite database and consumed by
the single shared fusion service (``src.fusion.engine``, optionally running
as its own process via ``python -m src.fusion.worker``).  Each worker
triggers a fusion batch at the end of its run so trajectories are always
available even when the launcher is not used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.detection.detector import (
    HF_PLATE_MODEL_FILENAME,
    HF_PLATE_MODEL_REPO_ID,
    PlateDetector,
    resolve_plate_model_from_hub,
)
from src.ocr.reader import PlateOCR
from src.tracking.tracker import VehicleTracker
from src.alerts.engine import AlertEngine
from src.db.schema import insert_row
from src.analytics.engine import TrafficAnalytics
from src.calibration import (
    DEFAULT_CAMERAS_PATH,
    load_cameras_config,
    log_validation,
    merge_camera_config,
    validate_cameras_config,
)

logger = logging.getLogger(__name__)

_FORMAT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"


def _bbox_iou(bbox_a: list, bbox_b: list) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    if inter == 0:
        return 0.0
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PipelineRunner:
    """Runs the full ANPR pipeline for a single camera.

    Args:
        camera_id: Identifier for this camera (e.g. ``"cam_1"``).
        video_path: Path to the input video file.
        camera_config: Dict with at least ``gps_lat`` and ``gps_lon`` keys.
            Calibration values (``pixel_to_meter_ratio``, ``compass_bearing``)
            are merged from ``data/calibration/cameras.json``.
        device: Compute device string (``"cpu"``, ``"cuda:0"``, etc.) or
            *None* to auto-detect.
    """

    def __init__(
        self,
        camera_id: str,
        video_path: str | Path,
        camera_config: dict[str, Any],
        device: Optional[str] = None,
        min_ocr_confidence: float = 0.50,
        min_vehicle_height_px: int = 60,
    ) -> None:
        self.camera_id = camera_id
        self.video_path = Path(video_path)
        self.device = device
        self.min_ocr_confidence = min_ocr_confidence
        self.min_vehicle_height_px = min_vehicle_height_px
        self._cameras_config_path = DEFAULT_CAMERAS_PATH

        # Merge calibration (GPS, pixel_to_meter_ratio, compass_bearing) onto
        # the values supplied by the caller/CLI.  Validate + warn at startup.
        warn_issues: list[str] = []
        try:
            cameras_cfg = load_cameras_config(self._cameras_config_path)
            issues = validate_cameras_config(cameras_cfg)
            log_validation(issues)
            warn_issues = [m for m in issues if m.startswith("[WARN]")]
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Camera calibration unavailable (%s); using caller config only.", exc)
            cameras_cfg = {}

        self.camera_config: dict[str, Any] = merge_camera_config(camera_id, cameras_cfg, camera_config)
        warn_issues = [m for m in warn_issues if camera_id not in m]
        for w in warn_issues:
            logger.warning("%s (for this camera)", w)

        self.gps_lat: float = float(self.camera_config.get("gps_lat", 0.0))
        self.gps_lon: float = float(self.camera_config.get("gps_lon", 0.0))
        self.pixel_to_meter_ratio: Optional[float] = self.camera_config.get("pixel_to_meter_ratio")
        self.compass_bearing: Optional[float] = self.camera_config.get("compass_bearing")
        self.direction: Optional[str] = (
            f"{self.compass_bearing:.0f}deg" if self.compass_bearing is not None else None
        )

        logger.info(
            "Initialising pipeline runner for camera %s (video=%s, ratio=%s, bearing=%s)",
            self.camera_id,
            self.video_path,
            self.pixel_to_meter_ratio,
            self.compass_bearing,
        )

        # Load the dedicated license-plate detector from Hugging Face.  This is
        # a hard dependency of the plate-localization stage: without weights the
        # stage cannot run, so fail loudly rather than silently producing zero
        # plate reads or falling back to the old heuristic crop.
        logger.info("Fetching license-plate detector from Hugging Face (%s/%s) …",
                    HF_PLATE_MODEL_REPO_ID, HF_PLATE_MODEL_FILENAME)
        plate_model_path = resolve_plate_model_from_hub()
        self.detector = PlateDetector(device=self.device, plate_model_path=plate_model_path)
        self.ocr = PlateOCR()
        self.tracker = VehicleTracker()
        self.alert_engine = AlertEngine()
        self.analytics = TrafficAnalytics(cameras_config_path=str(self._cameras_config_path))

        self._sightings: list[dict[str, Any]] = []
        self._alerts: list[dict[str, Any]] = []
        self._analytics_entries: list[dict[str, Any]] = []

        # Per-run diagnostics for the detection -> OCR path.
        self._diag: dict[str, Any] = {
            "vehicles_detected": 0,
            "vehicle_too_small": 0,
            "vehicles_with_plate_box": 0,
            "no_plate_box_found": 0,
            "plate_found_below_gate": 0,
            "plate_crop_heights": [],
            "plate_gate_cleared": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        speed_factor: float = 1.0,
        max_frames: Optional[int] = None,
        run_fusion: bool = True,
    ) -> dict[str, Any]:
        """Open the video and process frames through the full pipeline.

        Args:
            speed_factor: Playback speed multiplier.  ``1.0`` attempts
                real-time pacing; ``0`` disables sleeping.
            max_frames: Stop after this many frames.  *None* means
                process until end-of-video.
            run_fusion: Trigger the shared fusion service at the end of the
                run so trajectories persist immediately.

        Returns:
            Summary dict with keys ``frames_processed``, ``vehicles_detected``,
            ``plates_read``, ``alerts_triggered``.
        """
        if not self.video_path.exists():
            logger.error("Video file not found: %s", self.video_path)
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        init_db_safe()
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sleep_interval = (1.0 / fps) * speed_factor if speed_factor > 0 else 0.0

        logger.info(
            "Video: %s | %.1f fps | %d frames | sleep=%.3f s",
            self.video_path.name,
            fps,
            total_frames_video,
            sleep_interval,
        )

        summary = {
            "frames_processed": 0,
            "vehicles_detected": 0,
            "plates_read": 0,
            "alerts_triggered": 0,
        }

        frame_number = 0
        try:
            while True:
                if max_frames is not None and frame_number >= max_frames:
                    logger.info("Reached max_frames limit (%d).", max_frames)
                    break

                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached at frame %d.", frame_number)
                    break

                result = self.process_frame(frame, frame_number)

                summary["frames_processed"] += 1
                summary["vehicles_detected"] += result["vehicles_detected"]
                summary["plates_read"] += result["plates_read"]
                summary["alerts_triggered"] += result["alerts_triggered"]

                frame_number += 1

                if frame_number % 100 == 0:
                    logger.info(
                        "Progress: %d frames | %d vehicles | %d plates | %d alerts",
                        summary["frames_processed"],
                        summary["vehicles_detected"],
                        summary["plates_read"],
                        summary["alerts_triggered"],
                    )

                if sleep_interval > 0:
                    time.sleep(sleep_interval)

        finally:
            cap.release()
            logger.info("Video capture released.")

        self._flush_analytics()

        if run_fusion:
            try:
                from src.fusion.engine import run_fusion_once
                stats = run_fusion_once(cameras_config_path=str(self._cameras_config_path))
                logger.info(
                    "[FUSION] run complete: %d sightings -> %d trajectories",
                    stats["num_sightings"],
                    stats["num_trajectories"],
                )
            except Exception:
                logger.exception("[FUSION] end-of-run fusion failed")

        logger.info(
            "Pipeline run complete. %d frames processed, %d vehicles detected, "
            "%d plates read, %d alerts triggered.",
            summary["frames_processed"],
            summary["vehicles_detected"],
            summary["plates_read"],
            summary["alerts_triggered"],
        )

        summary["plate_diagnostics"] = {
            "vehicles_detected": self._diag["vehicles_detected"],
            "vehicle_too_small": self._diag["vehicle_too_small"],
            "vehicles_with_plate_box": self._diag["vehicles_with_plate_box"],
            "no_plate_box_found": self._diag["no_plate_box_found"],
            "plate_found_below_gate": self._diag["plate_found_below_gate"],
            "plate_crop_heights_px": self._diag["plate_crop_heights"],
            "plate_confidence_gate_cleared": self._diag["plate_gate_cleared"],
        }
        self._print_diagnostic_summary(summary)
        return summary

    def _print_diagnostic_summary(self, summary: dict[str, Any]) -> None:
        """Print the per-run detection -> OCR diagnostic breakdown."""
        d = self._diag
        heights: list[int] = d["plate_crop_heights"]
        attempted = d["vehicles_detected"] - d["vehicle_too_small"]
        plate_conf = self.detector.plate_conf_threshold
        agg = (
            f"{min(heights)} / {sum(heights)/len(heights):.1f} / {max(heights)}"
            if heights else "n/a"
        )

        lines = [
            "",
            "===== PLATE DIAGNOSTIC SUMMARY =====",
            f"Camera                       : {self.camera_id}",
            f"Video                        : {self.video_path.name}",
            f"Frames processed             : {summary['frames_processed']}",
            f"Vehicles detected            : {d['vehicles_detected']}",
            f"  - too small to attempt     : {d['vehicle_too_small']}  (height < {self.min_vehicle_height_px}px)",
            f"  - attempted                : {attempted}",
            f"  - no plate box found       : {d['no_plate_box_found']}  (plate conf < {plate_conf})",
            f"  - plate box localized      : {d['vehicles_with_plate_box']}",
            f"Plate crop height (px)       : {heights}",
            f"  - min / avg / max          : {agg}",
            f"Plate found but below gate   : {d['plate_found_below_gate']}  (ocr conf < {self.min_ocr_confidence})",
            f"Plate gate cleared           : {d['plate_gate_cleared']}  (ocr conf >= {self.min_ocr_confidence})",
            f"Plates read (sightings)      : {summary['plates_read']}",
            f"Alerts triggered             : {summary['alerts_triggered']}",
            "==================================",
        ]
        text = "\n".join(lines)
        print(text)
        logger.info(text)

    def process_frame(self, frame: np.ndarray, frame_number: int) -> dict[str, Any]:
        """Process a single frame through detection, tracking, OCR, and alerts.

        Stage order: detection -> tracking -> OCR (track-aware) -> DB -> alerts
        -> analytics.  Tracking runs before OCR so that each plate reading is
        associated with ``(camera_id, track_id)`` and voted on per vehicle.

        Returns:
            Dict with ``vehicles_detected``, ``plates_read``, ``alerts_triggered``,
            ``detections`` (vehicle list), ``plates`` (plate list with OCR text),
            ``tracked`` (tracked vehicle list).
        """
        ts = datetime.now().timestamp()

        # 1. Detection (vehicles only; plates are localized per-vehicle below)
        vehicles = self.detector.detect_vehicles(frame)

        class_counts = Counter(v.get("class_name", "?") for v in vehicles)
        logger.debug(
            "[DETECTION] %s frame=%d detections=%d classes=%s",
            self.camera_id, frame_number, len(vehicles),
            dict(class_counts),
        )

        # 2. Tracking (before OCR so readings carry a track identity)
        tracked = self.tracker.update(vehicles, frame=frame)
        logger.debug(
            "[TRACK] %s frame=%d active_tracks=%d ids=%s",
            self.camera_id, frame_number, len(tracked),
            sorted(t["track_id"] for t in tracked),
        )
        veh_track_ids = self._match_vehicle_tracks(vehicles, tracked)

        vehicles_detected = len(vehicles)
        plates_read = 0
        alerts_triggered = 0
        self._diag["vehicles_detected"] += vehicles_detected

        ocr_plates: list[dict[str, Any]] = []

        # 3. Per-vehicle plate localization -> OCR.  The dedicated plate model
        # runs on each vehicle crop (not the full frame).  If no plate box
        # clears the plate confidence threshold, OCR is skipped for that
        # vehicle entirely - there is no fallback to the old heuristic crop.
        crop_ocr = self._ocr_vehicle_plates_by_model(
            frame, vehicles, veh_track_ids, frame_number
        )
        ocr_plates = crop_ocr["plate_entries"]
        plates_read = crop_ocr["read_count"]
        for sighting in crop_ocr["sightings"]:
            self._handle_sighting(sighting)

        # 4. Analytics (active track ids + per-track speeds)
        active_ids = [t["track_id"] for t in tracked]
        self.analytics.update_camera_count(
            self.camera_id,
            ts,
            len(tracked),
            frame_shape=frame.shape,
            track_ids=active_ids,
        )
        self.analytics.update_tracked_speeds(self.camera_id, tracked, ts)
        self._analytics_entries.append(
            {
                "camera_id": self.camera_id,
                "timestamp": ts,
                "vehicle_count": len(tracked),
                "frame_number": frame_number,
            }
        )

        return {
            "vehicles_detected": vehicles_detected,
            "plates_read": plates_read,
            "alerts_triggered": alerts_triggered,
            "detections": vehicles,
            "plates": ocr_plates,
            "tracked": tracked,
        }

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _match_vehicle_tracks(
        self, vehicles: list[dict], tracked: list[dict]
    ) -> dict[int, Optional[int]]:
        """Map each vehicle detection index to its tracker track id."""
        mapping: dict[int, Optional[int]] = {}
        for idx, veh in enumerate(vehicles):
            best_tid, best_iou = None, 0.0
            for t in tracked:
                iou = _bbox_iou(veh["bbox"], t["bbox"])
                if iou > best_iou:
                    best_iou, best_tid = iou, t["track_id"]
            mapping[idx] = best_tid if best_iou >= 0.3 else None
        return mapping

    def _track_key(self, track_id: Optional[int], frame_number: int, index: int) -> tuple:
        """Build a per-vehicle OCR voting identity.

        With a valid track id: ``(camera_id, track_id)``.  Without one the key
        is frame-unique so readings from different unmatched vehicles are never
        pooled into the same vote.
        """
        if track_id is not None:
            return (self.camera_id, int(track_id))
        return (self.camera_id, f"unmatched:{frame_number}:{index}")

    def _ocr_vehicle_plates_by_model(
        self,
        frame: np.ndarray,
        vehicles: list[dict],
        veh_track_ids: dict[int, Optional[int]],
        frame_number: int,
    ) -> dict[str, Any]:
        """Localize each vehicle's plate box and OCR only that plate crop.

        For every vehicle that clears ``min_vehicle_height_px``:

        1. The dedicated plate model runs on the vehicle crop to find the
           plate box within the vehicle.
        2. If no plate box clears the plate confidence threshold (0.25), OCR
           is skipped for that vehicle entirely (no heuristic fallback).
        3. Otherwise the plate crop is preprocessed (grayscale -> CLAHE ->
           3-4x upscale -> unsharp) and read by EasyOCR under its
           ``(camera_id, track_id)`` voting identity.

        Diagnostics distinguish *vehicle too small*, *no plate box found*,
        *plate found but below confidence gate*, and *gate cleared*.
        """
        h_max, w_max = frame.shape[:2]
        plate_entries: list[dict[str, Any]] = []
        sightings: list[dict[str, Any]] = []
        read_count = 0

        for idx, veh in enumerate(vehicles):
            x1, y1, x2, y2 = veh["bbox"]
            veh_h = y2 - y1
            if veh_h < self.min_vehicle_height_px:
                logger.info(
                    "[PLATE] %s frame=%d veh=%d vehicle too small to attempt "
                    "(h=%dpx < min_vehicle_height_px=%d)",
                    self.camera_id, frame_number, idx,
                    veh_h, self.min_vehicle_height_px,
                )
                self._diag["vehicle_too_small"] += 1
                continue

            # Run the plate model on the vehicle crop (not the full frame).
            plate_dets = self.detector.detect_plates_in_crop(frame, veh["bbox"])
            if not plate_dets:
                logger.info(
                    "[PLATE] %s frame=%d veh=%d no plate box found (plate conf < %s)",
                    self.camera_id, frame_number, idx,
                    self.detector.plate_conf_threshold,
                )
                self._diag["no_plate_box_found"] += 1
                continue

            best = max(plate_dets, key=lambda p: p["confidence"])
            self._diag["vehicles_with_plate_box"] += 1

            bx1 = max(0, min(int(best["bbox"][0]), w_max))
            by1 = max(0, min(int(best["bbox"][1]), h_max))
            bx2 = max(0, min(int(best["bbox"][2]), w_max))
            by2 = max(0, min(int(best["bbox"][3]), h_max))
            if bx2 <= bx1 or by2 <= by1:
                logger.debug(
                    "[OCR] %s frame=%d veh=%d degenerate plate box; skipping",
                    self.camera_id, frame_number, idx,
                )
                continue

            crop_h = by2 - by1
            crop_w = bx2 - bx1
            self._diag["plate_crop_heights"].append(crop_h)

            crop = frame[by1:by2, bx1:bx2]
            if crop_w < 20 or crop_h < 10:
                logger.debug(
                    "[OCR] %s frame=%d veh=%d plate crop too small (w=%d h=%d); skipping",
                    self.camera_id, frame_number, idx, crop_w, crop_h,
                )
                continue

            track_id = veh_track_ids.get(idx)
            track_key = self._track_key(track_id, frame_number, idx)

            ocr_result = self.ocr.read_plate(crop, track_key=track_key)
            text = ocr_result["text"]
            conf = ocr_result["confidence"]

            entry = {
                "bbox": [bx1, by1, bx2, by2],
                "confidence": best["confidence"],
                "plate_text": text,
                "ocr_confidence": conf,
                "vehicle_idx": idx,
                "track_id": track_id,
                "source": "plate_model",
            }
            plate_entries.append(entry)

            if not text:
                logger.info(
                    "[OCR] %s frame=%d veh=%d plate found but OCR read nothing",
                    self.camera_id, frame_number, idx,
                )
                continue

            if conf < self.min_ocr_confidence:
                logger.info(
                    "[OCR] %s frame=%d veh=%d plate found but below confidence "
                    "gate '%s' (conf=%.2f < %.2f)",
                    self.camera_id, frame_number, idx,
                    text, conf, self.min_ocr_confidence,
                )
                self._diag["plate_found_below_gate"] += 1
                continue

            # Confident reading -> gate cleared -> sighting.
            read_count += 1
            self._diag["plate_gate_cleared"] += 1
            ts = datetime.now().timestamp()
            logger.info(
                "[OCR] %s track=%s plate=%s conf=%.2f (votes=%d)",
                self.camera_id, track_id, text, conf,
                ocr_result.get("voted_count", 0),
            )
            sightings.append(
                {
                    "plate": text,
                    "camera_id": self.camera_id,
                    "gps_lat": self.gps_lat,
                    "gps_lon": self.gps_lon,
                    "timestamp": ts,
                    "confidence": conf,
                    "vehicle_bbox": veh["bbox"],
                    "plate_bbox": [bx1, by1, bx2, by2],
                    "frame_number": frame_number,
                    "track_id": track_id,
                    "direction": self.direction,
                    "class_name": veh["class_name"],
                }
            )

        return {"plate_entries": plate_entries, "sightings": sightings, "read_count": read_count}

    def _handle_sighting(self, sighting: dict[str, Any]) -> None:
        """Persist a sighting to the shared DB and run the alert check."""
        self._insert_sighting(sighting)
        self._sightings.append(sighting)
        logger.info(
            "[SIGHTING] %s track=%s plate=%s stored (conf=%.2f)",
            self.camera_id, sighting.get("track_id"), sighting["plate"], sighting["confidence"],
        )

        ts = sighting["timestamp"]
        alerts = self.alert_engine.check_plate(
            plate=sighting["plate"],
            camera_id=self.camera_id,
            gps_lat=sighting["gps_lat"],
            gps_lon=sighting["gps_lon"],
            timestamp=ts,
        )
        for alert in alerts:
            logger.info(
                "[ALERT] %s detected for plate=%s type=%s matched=%s",
                self.camera_id, alert["plate"], alert["alert_type"],
                alert.get("details", ""),
            )
        self._alerts.extend(alerts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_sighting(sighting: dict[str, Any]) -> None:
        """Write a sighting row to the shared database (retry on locks)."""
        insert_row(
            """
            INSERT INTO sightings
                (plate, plate_confidence, camera_id, gps_lat, gps_lon,
                 timestamp, vehicle_class, vehicle_bbox, plate_bbox, frame_number,
                 track_id, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sighting["plate"],
                sighting["confidence"],
                sighting["camera_id"],
                sighting["gps_lat"],
                sighting["gps_lon"],
                sighting["timestamp"],
                sighting.get("class_name"),
                json.dumps(sighting["vehicle_bbox"]) if sighting.get("vehicle_bbox") else None,
                json.dumps(sighting["plate_bbox"]),
                sighting["frame_number"],
                sighting.get("track_id"),
                sighting.get("direction"),
            ),
        )

    def _flush_analytics(self) -> None:
        """Persist accumulated analytics to the shared database."""
        if not self._analytics_entries:
            return

        summary = self.analytics.get_analytics_summary()
        cam_summary = summary.get(self.camera_id, {})

        avg_speed = cam_summary.get("avg_speed_kmh")
        active = cam_summary.get("active_vehicles", 0)
        congestion = cam_summary.get("congestion", {})
        level = congestion.get("congestion_level", "low")
        is_congested = congestion.get("is_congested", False)

        if self._analytics_entries:
            last = self._analytics_entries[-1]
            self.analytics.store_analytics(
                camera_id=self.camera_id,
                timestamp=last["timestamp"],
                vehicle_count=active,
                avg_speed=avg_speed,
                density_level=level,
                congestion_flag=is_congested,
            )
            speed_txt = f"{avg_speed:.1f}km/h" if avg_speed is not None else "n/a"
            logger.info(
                "[ANALYTICS] %s active_tracks=%d avg_speed=%s density=%s",
                self.camera_id, active, speed_txt, level,
            )


def init_db_safe() -> None:
    """Initialise / upgrade the shared database schema."""
    from src.db.schema import init_db
    init_db()


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ANPR pipeline for a single camera.",
    )
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Camera identifier (e.g. cam_1).",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--gps-lat",
        type=float,
        default=0.0,
        help="GPS latitude of the camera (default: 0.0).",
    )
    parser.add_argument(
        "--gps-lon",
        type=float,
        default=0.0,
        help="GPS longitude of the camera (default: 0.0).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (cpu, cuda:0, etc.). Auto-detected when omitted.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for ``python -m src.pipeline_runner``."""
    logging.basicConfig(level=logging.INFO, format=_FORMAT)

    parser = _build_parser()
    args = parser.parse_args(argv)

    logger.setLevel(getattr(logging, args.log_level))

    camera_config = {
        "gps_lat": args.gps_lat,
        "gps_lon": args.gps_lon,
    }

    runner = PipelineRunner(
        camera_id=args.camera_id,
        video_path=args.video,
        camera_config=camera_config,
        device=args.device,
    )

    summary = runner.run(
        speed_factor=args.speed_factor,
        max_frames=args.max_frames,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()