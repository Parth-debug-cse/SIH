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

from src.detection.detector import PlateDetector
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

        self.detector = PlateDetector(device=self.device)
        self.ocr = PlateOCR()
        self.tracker = VehicleTracker()
        self.alert_engine = AlertEngine()
        self.analytics = TrafficAnalytics(cameras_config_path=str(self._cameras_config_path))

        self._sightings: list[dict[str, Any]] = []
        self._alerts: list[dict[str, Any]] = []
        self._analytics_entries: list[dict[str, Any]] = []

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
        return summary

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

        # 1. Detection
        det_result = self.detector.detect(frame)
        vehicles = det_result["vehicles"]
        plates = det_result["plates"]

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

        ocr_plates: list[dict[str, Any]] = []

        # 3a. OCR on each directly detected plate (requires a plate model)
        for plate_det in plates:
            plate_entry, sighting = self._process_plate_detection(
                frame, plate_det, vehicles, veh_track_ids, ts, frame_number
            )
            if plate_entry is not None:
                ocr_plates.append(plate_entry)
                if sighting is not None:
                    plates_read += 1
                    self._handle_sighting(sighting)

        # 3b. Plate-crop fallback: when no plate model exists, crop the
        # lower-center region of each vehicle bbox (where plates sit) and OCR.
        if not plates and vehicles:
            crop_reads = self._ocr_vehicle_plate_crops(
                frame, vehicles, veh_track_ids, frame_number
            )
            ocr_plates.extend(crop_reads["plate_entries"])
            plates_read += crop_reads["read_count"]
            for sighting in crop_reads["sightings"]:
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

    def _process_plate_detection(
        self,
        frame: np.ndarray,
        plate_det: dict,
        vehicles: list[dict],
        veh_track_ids: dict[int, Optional[int]],
        ts: float,
        frame_number: int,
    ) -> tuple[Optional[dict], Optional[dict]]:
        """OCR a directly-detected plate and build a sighting when accepted."""
        x1, y1, x2, y2 = plate_det["bbox"]
        h, w = frame.shape[:2]
        cx1 = max(0, int(x1))
        cy1 = max(0, int(y1))
        cx2 = min(w, int(x2))
        cy2 = min(h, int(y2))

        if cx2 <= cx1 or cy2 <= cy1:
            logger.debug("[OCR] skipping degenerate plate crop at frame %d", frame_number)
            return None, None

        veh_idx = plate_det.get("vehicle_idx", -1)
        track_id = (
            veh_track_ids.get(veh_idx)
            if 0 <= veh_idx < len(vehicles) else None
        )
        track_key = self._track_key(track_id, frame_number, veh_idx)

        crop = frame[cy1:cy2, cx1:cx2]
        ocr_result = self.ocr.read_plate(crop, track_key=track_key)
        plate_text = ocr_result["text"]
        plate_confidence = ocr_result["confidence"]

        plate_entry = dict(plate_det)
        plate_entry["plate_text"] = plate_text
        plate_entry["ocr_confidence"] = plate_confidence
        plate_entry["track_id"] = track_id

        if not plate_text or plate_confidence < self.min_ocr_confidence:
            if plate_text:
                logger.debug(
                    "[OCR] rejected low-confidence plate '%s' (conf=%.2f < %.2f)",
                    plate_text, plate_confidence, self.min_ocr_confidence,
                )
            return plate_entry, None

        logger.info("[OCR] %s track=%s plate=%s conf=%.2f (votes=%d)",
                    self.camera_id, track_id, plate_text, plate_confidence,
                    ocr_result.get("voted_count", 0))

        sighting = {
            "plate": plate_text,
            "camera_id": self.camera_id,
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "timestamp": ts,
            "confidence": plate_confidence,
            "vehicle_bbox": vehicles[veh_idx]["bbox"] if 0 <= veh_idx < len(vehicles) else None,
            "plate_bbox": plate_det["bbox"],
            "frame_number": frame_number,
            "track_id": track_id,
            "direction": self.direction,
            "class_name": vehicles[veh_idx].get("class_name") if 0 <= veh_idx < len(vehicles) else None,
        }
        return plate_entry, sighting

    def _ocr_vehicle_plate_crops(
        self,
        frame: np.ndarray,
        vehicles: list[dict],
        veh_track_ids: dict[int, Optional[int]],
        frame_number: int,
    ) -> dict[str, Any]:
        """OCR the lower-center crop of each vehicle bbox (plate location).

        No dedicated plate-detection model is available, so we crop the
        region of each vehicle where number plates typically sit (bottom
        ~30% of the bbox) and run OCR on the real pixels.  Each crop is read
        and voted on under its own ``(camera_id, track_id)`` identity.
        """
        h_max, w_max = frame.shape[:2]
        plate_entries: list[dict[str, Any]] = []
        sightings: list[dict[str, Any]] = []
        read_count = 0
        candidates = 0

        for idx, veh in enumerate(vehicles):
            x1, y1, x2, y2 = veh["bbox"]
            veh_h = y2 - y1
            if veh_h < self.min_vehicle_height_px:
                continue
            candidates += 1

            bw = (x2 - x1) * 1.0
            bh = (y2 - y1) * 0.30
            px1 = int(x1 + bw * 0.15)
            py1 = int(y2 - bh)
            px2 = int(x2 - bw * 0.15)
            py2 = int(y2 - 1)

            # Clamp to frame
            cx1 = max(0, min(px1, w_max))
            cy1 = max(0, min(py1, h_max))
            cx2 = max(0, min(px2, w_max))
            cy2 = max(0, min(py2, h_max))
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = frame[cy1:cy2, cx1:cx2]
            # Skip tiny crops
            if crop.shape[0] < 10 or crop.shape[1] < 20:
                continue

            track_id = veh_track_ids.get(idx)
            track_key = self._track_key(track_id, frame_number, idx)

            # Upscale 2x so OCR can see small plate glyphs
            upscale = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            ocr_result = self.ocr.read_plate(upscale, track_key=track_key)
            text = ocr_result["text"]
            conf = ocr_result["confidence"]

            entry = {
                "bbox": [cx1, cy1, cx2, cy2],
                "confidence": 0.0,
                "plate_text": text,
                "ocr_confidence": conf,
                "vehicle_idx": idx,
                "track_id": track_id,
                "source": "vehicle_crop_fallback",
            }
            plate_entries.append(entry)

            # Only accept confident plate readings (anti-hallucination gate)
            if text and conf >= self.min_ocr_confidence:
                read_count += 1
                ts = datetime.now().timestamp()
                logger.info("[OCR] %s track=%s plate=%s conf=%.2f (votes=%d)",
                            self.camera_id, track_id, text, conf,
                            ocr_result.get("voted_count", 0))
                sightings.append(
                    {
                        "plate": text,
                        "camera_id": self.camera_id,
                        "gps_lat": self.gps_lat,
                        "gps_lon": self.gps_lon,
                        "timestamp": ts,
                        "confidence": conf,
                        "vehicle_bbox": veh["bbox"],
                        "plate_bbox": [cx1, cy1, cx2, cy2],
                        "frame_number": frame_number,
                        "track_id": track_id,
                        "direction": self.direction,
                        "class_name": veh["class_name"],
                    }
                )
            elif text:
                logger.debug(
                    "[OCR] %s rejected low-confidence plate reading '%s' (conf=%.2f < %.2f)",
                    self.camera_id, text, conf, self.min_ocr_confidence,
                )

        if read_count:
            logger.info(
                "[SIGHTING] %s frame=%d plate-crop fallback read %d plate(s) from %d vehicle crop(s) (%d candidates)",
                self.camera_id, frame_number, read_count, len(plate_entries), candidates,
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