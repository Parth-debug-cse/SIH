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
from src.ocr.reader import PlateOCR, looks_like_plate, position_correct
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
        plate_pad_ratio_x: float = 0.15,
        plate_pad_ratio_y: float = 0.30,
        debug_crops_dir: Optional[str | Path] = None,
    ) -> None:
        self.camera_id = camera_id
        self.video_path = Path(video_path)
        self.device = device
        self.min_ocr_confidence = min_ocr_confidence
        self.min_vehicle_height_px = min_vehicle_height_px
        self.plate_pad_ratio_x = plate_pad_ratio_x
        self.plate_pad_ratio_y = plate_pad_ratio_y
        self.debug_crops_dir: Optional[Path] = Path(debug_crops_dir) if debug_crops_dir else None
        if self.debug_crops_dir is not None:
            self.debug_crops_dir.mkdir(parents=True, exist_ok=True)
        self._debug_saved: dict[str, int] = {}
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
        # Candidate ledger: one row per evaluated plate box (diagnostic
        # only; never influences acceptance).  See _record_candidate().
        self._ledger: list[dict[str, Any]] = []
        # Direct-recognition trials across variants A–F (diagnostic only).
        self._variant_trials: list[dict[str, Any]] = []
        # Best-variant images retained for the top-5 Drive save.
        self._top_images: list[dict[str, Any]] = []
        # fast-plate-ocr A/B trials (diagnostic only; lazy backend).
        self._fast_trials: list[dict[str, Any]] = []
        self._fast_top_images: list[dict[str, Any]] = []
        self.fast_ocr = None

        # Per-run diagnostics for the detection -> OCR path.
        self._diag: dict[str, Any] = {
            "vehicles_detected": 0,
            "vehicle_too_small": 0,
            "vehicles_with_plate_box": 0,
            "no_plate_box_found": 0,
            "plate_found_below_gate": 0,
            "plate_crop_heights": [],
            "plate_crop_widths": [],
            "plate_fullframe_fallback_used": 0,
            "plate_fullframe_boxes": 0,
            "plate_crop_tiny_skipped": 0,
            # Explicit candidate-ledger counters (diagnostic mirrors; the
            # acceptance thresholds they describe are unchanged):
            "candidate_boxes": 0,      # every boxed candidate evaluated
            "aspect_rejected": 0,      # run-delta of detector aspect filter
            "position_rejected": 0,    # run-delta of detector bumper filter
            "tiny_rejected": 0,        # == plate_crop_tiny_skipped
            "detector_conf_rejected": 0,  # == no_plate_box_found: no box
                                       # cleared the plate-conf threshold
            "ocr_attempted": 0,        # every acceptance-path read_plate call
            "ocr_empty": 0,            # OCR returned no text at all
            "ocr_low_conf": 0,         # == plate_found_below_gate
            "regex_rejected": 0,       # == plate_confidence_cleared_format_rejected
            "accepted": 0,             # == plate_gate_cleared
            "plate_conf_cleared": 0,
            "plate_confidence_cleared_format_rejected": 0,
            "plate_format_rejected_samples": [],
            "plate_gate_cleared": 0,
            "plate_aspect_ratio_rejected": 0,
            "preprocess_calls": 0,
            "preprocessed_crop_saved": None,
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

        # Post-run diagnostics: how many plate candidates the aspect-ratio
        # filter rejected, how many times preprocessing actually ran, and the
        # one preprocessed crop saved for manual inspection this run.
        self._diag["plate_aspect_ratio_rejected"] = self.detector.plate_aspect_ratio_rejects
        self._diag["preprocess_calls"] = self.ocr.preprocess_count
        if self.ocr.last_preprocessed is not None:
            out_dir = Path("data") / "debug_ocr"
            out_dir.mkdir(parents=True, exist_ok=True)
            preprocessed_path = out_dir / f"preprocessed_{self.camera_id}.jpg"
            cv2.imwrite(str(preprocessed_path), self.ocr.last_preprocessed)
            self._diag["preprocessed_crop_saved"] = str(preprocessed_path)
            logger.info(
                "[OCR-PREPROCESS] saved preprocessed plate crop for manual "
                "inspection: %s",
                preprocessed_path,
            )

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

        # Detector-internal filter deltas for this run (the detector object
        # is per-runner, so end-of-run totals are the run deltas).  These
        # boxes never reach the ledger — they are filtered inside the
        # detector — hence reported as aggregate counters, not rows.
        self._diag["aspect_rejected"] = self.detector.plate_aspect_ratio_rejects
        self._diag["position_rejected"] = self.detector.plate_position_rejects
        self._diag["detector_conf_rejected"] = self._diag["no_plate_box_found"]

        ledger_csv = self._save_ledger_csv()
        top5_csv = self._save_top5()
        fast_top5_csv = self._save_fast_top5()
        track_reports = self._print_track_fusion_report()
        stable_csv = self._save_stable_tracks(track_reports)

        summary["plate_diagnostics"] = {
            "vehicles_detected": self._diag["vehicles_detected"],
            "vehicle_too_small": self._diag["vehicle_too_small"],
            "vehicles_with_plate_box": self._diag["vehicles_with_plate_box"],
            "no_plate_box_found": self._diag["no_plate_box_found"],
            "plate_found_below_gate": self._diag["plate_found_below_gate"],
            "plate_crop_heights_px": self._diag["plate_crop_heights"],
            "plate_crop_widths_px": self._diag["plate_crop_widths"],
            "plate_confidence_gate_cleared": self._diag["plate_conf_cleared"],
            "plate_format_check_rejected": self._diag["plate_confidence_cleared_format_rejected"],
            "plate_format_rejected_samples": self._diag["plate_format_rejected_samples"],
            "plate_both_gates_cleared": self._diag["plate_gate_cleared"],
            "plate_aspect_ratio_rejected": self._diag["plate_aspect_ratio_rejected"],
            "plate_fullframe_fallback_used": self._diag["plate_fullframe_fallback_used"],
            "plate_fullframe_boxes": self._diag["plate_fullframe_boxes"],
            "plate_crop_tiny_skipped": self._diag["plate_crop_tiny_skipped"],
            "candidate_boxes": self._diag["candidate_boxes"],
            "aspect_rejected": self._diag["aspect_rejected"],
            "position_rejected": self._diag["position_rejected"],
            "tiny_rejected": self._diag["tiny_rejected"],
            "detector_conf_rejected": self._diag["detector_conf_rejected"],
            "ocr_attempted": self._diag["ocr_attempted"],
            "ocr_empty": self._diag["ocr_empty"],
            "ocr_low_conf": self._diag["ocr_low_conf"],
            "regex_rejected": self._diag["regex_rejected"],
            "accepted": self._diag["accepted"],
            "candidate_ledger_csv": ledger_csv,
            "candidate_ledger_rows": len(self._ledger),
            "variant_trials": len(self._variant_trials),
            "top5_direct_csv": top5_csv,
            "fast_trials": len(self._fast_trials),
            "fast_top5_csv": fast_top5_csv,
            "fast_tracks": len({o["track_key"] for o in self.ocr._observations
                                if o["source"] == "fastplate"}),
            "stable_tracks_csv": stable_csv,
            "preprocess_calls": self._diag["preprocess_calls"],
            "preprocessed_crop_saved": self._diag["preprocessed_crop_saved"],
        }
        self._print_diagnostic_summary(summary)
        self._print_ledger_summary()
        return summary

    def _print_diagnostic_summary(self, summary: dict[str, Any]) -> None:
        """Print the per-run detection -> OCR diagnostic breakdown."""
        d = self._diag
        heights: list[int] = d["plate_crop_heights"]
        widths: list[int] = d["plate_crop_widths"]
        attempted = d["vehicles_detected"] - d["vehicle_too_small"]
        plate_conf = self.detector.plate_conf_threshold
        aspect_min, aspect_max = self.detector.plate_aspect_ratio_bounds
        agg = (
            f"{min(heights)} / {sum(heights)/len(heights):.1f} / {max(heights)}"
            if heights else "n/a"
        )
        wagg = (
            f"{min(widths)} / {sum(widths)/len(widths):.1f} / {max(widths)}"
            if widths else "n/a"
        )
        format_samples = d["plate_format_rejected_samples"]
        sample_txt = ", ".join(f"'{s}'" for s in format_samples[:10]) or "none"

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
            f"  - aspect-ratio rejected    : {d['plate_aspect_ratio_rejected']}  (outside {aspect_min:g}:1 - {aspect_max:g}:1)",
            f"  - plate box localized      : {d['vehicles_with_plate_box']}",
            f"  - full-frame fallback hits : {d['plate_fullframe_fallback_used']}  (full-frame boxes total: {d['plate_fullframe_boxes']})",
            f"  - tiny crops rejected      : {d['plate_crop_tiny_skipped']}  (w<20 or h<10, never sent to OCR)",
            f"Plate crop height (px)       : {heights}",
            f"  - min / avg / max          : {agg}",
            f"Plate crop width (px)        : {widths}",
            f"  - min / avg / max          : {wagg}",
            f"Plate found but below gate   : {d['plate_found_below_gate']}  (ocr conf < {self.min_ocr_confidence})",
            f"Confidence gate cleared      : {d['plate_conf_cleared']}",
            f"  - format check rejected    : {d['plate_confidence_cleared_format_rejected']}  (not an Indian plate)"
            + (f"  e.g. {sample_txt}" if format_samples else ""),
            f"Plate both gates cleared     : {d['plate_gate_cleared']}  (conf >= {self.min_ocr_confidence} AND plate format)",
            f"Plates read (sightings)      : {summary['plates_read']}",
            f"Preprocessing calls          : {d['preprocess_calls']}",
            f"Preprocessed crop saved      : {d['preprocessed_crop_saved'] or 'none'}",
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
        2. Plate candidates outside the 2:1-5:1 aspect-ratio band are
           rejected by the detector (no crop is made for them).
        3. If no plate box clears the plate confidence threshold (0.25), OCR
           is skipped for that vehicle entirely (no heuristic fallback).
        4. Otherwise the plate box is padded (~15-20% width / ~30% height) so
           leading state/series characters are not clipped, then the crop is
           preprocessed (grayscale -> CLAHE -> 3-4x upscale -> unsharp) and
           read by EasyOCR under its ``(camera_id, track_id)`` identity.
        5. A sighting is written only when the read clears **both** gates:
           ``ocr confidence >= min_ocr_confidence`` AND the text looks like an
           Indian plate (``looks_like_plate``).

        Diagnostics distinguish *vehicle too small*, *no plate box found*,
        *below confidence gate*, *confidence cleared but format rejected*, and
        *both gates cleared*.
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
            from_fallback = False
            if not plate_dets:
                # Fallback B: full-frame plate pass, associated back to this
                # vehicle via the detector's IoU matching.  Cached per frame
                # so N vehicles cost one extra inference, not N.
                if frame_number != getattr(self, "_ff_cache_frame", -1):
                    try:
                        self._ff_cache = self.detector.detect_plates(
                            frame, vehicle_detections=vehicles
                        )
                    except Exception:
                        logger.exception("[PLATE] full-frame fallback inference failed")
                        self._ff_cache = []
                    self._ff_cache_frame = frame_number
                    self._diag["plate_fullframe_boxes"] += len(self._ff_cache)
                plate_dets = [p for p in self._ff_cache if p.get("vehicle_idx") == idx]
                from_fallback = bool(plate_dets)
                if from_fallback:
                    self._diag["plate_fullframe_fallback_used"] += 1
                    logger.info(
                        "[PLATE] %s frame=%d veh=%d in-crop empty, full-frame fallback gave %d box(es)",
                        self.camera_id, frame_number, idx, len(plate_dets),
                    )
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
            self._diag["candidate_boxes"] += 1
            det_conf = float(best["confidence"])
            det_source = "fullframe_fallback" if from_fallback else "in_crop"

            track_id = veh_track_ids.get(idx)
            track_key = self._track_key(track_id, frame_number, idx)

            # Bumper/position verdict for the ledger (report-only: fallback
            # boxes already passed the detector's internal filter, in-crop
            # boxes are only evaluated here — nothing is filtered out).
            try:
                bumper_passed: Optional[bool] = bool(
                    self.detector._plate_sits_in_bumper_zone(best["bbox"], veh["bbox"])
                )
            except Exception:
                logger.exception(
                    "[LEDGER] %s frame=%d veh=%d bumper check failed",
                    self.camera_id, frame_number, idx,
                )
                bumper_passed = None

            row: dict[str, Any] = {
                "frame": frame_number,
                "track_id": track_id,
                "det_source": det_source,
                "plate_bbox": [int(v) for v in best["bbox"]],
                "det_conf": det_conf,
                "det_threshold": self.detector.plate_conf_threshold,
                "crop_w": None, "crop_h": None, "aspect": None,
                "bumper_passed": bumper_passed,
                "ocr_raw_text": "", "ocr_raw_conf": 0.0,
                "ocr_pre_text": "", "ocr_pre_conf": 0.0,
                "accept_text": "", "accept_conf": 0.0,
                "corrected": "", "regex_pass": False,
                "best_variant": "", "best_variant_text": "", "best_variant_conf": 0.0,
                "fast_text": "", "fast_conf": None,
                "ocr_threshold": self.min_ocr_confidence,
                "variant_error": "",
                "direct_error": "",
                "final_reason": "",
            }

            bx1 = max(0, min(int(best["bbox"][0]), w_max))
            by1 = max(0, min(int(best["bbox"][1]), h_max))
            bx2 = max(0, min(int(best["bbox"][2]), w_max))
            by2 = max(0, min(int(best["bbox"][3]), h_max))
            if bx2 <= bx1 or by2 <= by1:
                logger.info(
                    "[OCR] %s frame=%d veh=%d degenerate plate box; skipping",
                    self.camera_id, frame_number, idx,
                )
                row["final_reason"] = "degenerate_box"
                self._record_candidate(row)
                continue

            raw_crop_w = bx2 - bx1
            raw_crop_h = by2 - by1
            logger.debug(
                "[PLATE-BBOX] %s frame=%d veh=%d plate box %d x %d px (conf=%.2f)",
                self.camera_id, frame_number, idx,
                raw_crop_w, raw_crop_h, best["confidence"],
            )

            # Padding margin around the detected plate box (~15-20% of width,
            # ~30% of height) so leading state/series characters are not
            # clipped out of the crop before OCR.
            pad_x = int(raw_crop_w * self.plate_pad_ratio_x)
            pad_y = int(raw_crop_h * self.plate_pad_ratio_y)
            pbx1 = max(0, bx1 - pad_x)
            pby1 = max(0, by1 - pad_y)
            pbx2 = min(w_max, bx2 + pad_x)
            pby2 = min(h_max, by2 + pad_y)

            crop_w = pbx2 - pbx1
            crop_h = pby2 - pby1
            self._diag["plate_crop_heights"].append(crop_h)
            self._diag["plate_crop_widths"].append(crop_w)
            row["crop_w"] = crop_w
            row["crop_h"] = crop_h
            row["aspect"] = round(crop_w / crop_h, 2) if crop_h > 0 else None

            crop = frame[pby1:pby2, pbx1:pbx2]
            if crop_w < 20 or crop_h < 10:
                logger.info(
                    "[OCR] %s frame=%d veh=%d plate crop tiny, rejected before OCR (w=%d h=%d)",
                    self.camera_id, frame_number, idx, crop_w, crop_h,
                )
                self._diag["plate_crop_tiny_skipped"] += 1
                self._diag["tiny_rejected"] += 1
                row["final_reason"] = "tiny_rejected"
                self._save_candidate_set("tiny_rejected", frame, crop, None,
                                         [pbx1, pby1, pbx2, pby2], veh["bbox"],
                                         frame_number, idx)
                self._record_candidate(row)
                continue

            # Diagnostic-only variant reads (direct EasyOCR calls, no vote
            # history touched): raw 3x-upscaled vs full CLAHE+upscale+unsharp
            # preprocessing.  Acceptance below uses read_plate() unchanged.
            variants = self._diagnose_variants(crop, frame_number, idx)
            row["ocr_pre_text"] = variants["pre_text"]
            row["ocr_pre_conf"] = variants["pre_conf"]
            row["variant_error"] = variants["error"]

            # Direct-recognition experiment (diagnostic ONLY — acceptance
            # below still uses read_plate() unchanged): whole-crop
            # recognize() over variants A–F, bypassing text detection.
            import traceback as _tb

            direct_trials: list[dict[str, Any]] = []
            try:
                direct_trials = self.ocr.recognize_variants(crop)
            except Exception:
                _tb.print_exc()
                logger.exception("[DIAG] %s frame=%d veh=%d direct-recognize failed",
                                 self.camera_id, frame_number, idx)
                row["direct_error"] = "recognize_variants raised"
            for t in direct_trials:
                self._variant_trials.append({
                    "frame": frame_number, "track_id": track_id,
                    "det_source": det_source,
                    "plate_bbox": str([int(v) for v in best["bbox"]]),
                    "det_conf": det_conf,
                    "variant": t["variant"], "text": t["text"],
                    "conf": t["conf"], "normalized": t["normalized"],
                    "regex_pass": t["regex_pass"], "status": t["status"],
                })
            scored = [t for t in direct_trials
                      if t["status"] == "ok" and t["text"]]
            best_trial = max(scored, key=lambda t: t["conf"]) if scored else None
            if best_trial is not None:
                row["best_variant"] = best_trial["variant"]
                row["best_variant_text"] = best_trial["text"]
                row["best_variant_conf"] = best_trial["conf"]
                if best_trial["image"] is not None:
                    self._top_images.append({
                        "conf": best_trial["conf"],
                        "image": best_trial["image"],
                        "frame": frame_number, "track_id": track_id,
                        "variant": best_trial["variant"],
                        "text": best_trial["text"],
                        "det_conf": det_conf,
                        "normalized": best_trial["normalized"],
                    })
            logger.info(
                "[DIRECT] %s frame=%d veh=%d trials=%s",
                self.camera_id, frame_number, idx,
                [(t["variant"], t["text"], round(t["conf"], 3), t["status"])
                 for t in direct_trials],
            )

            # fast-plate-ocr A/B (diagnostic ONLY): RAW crop, no aggressive
            # preprocessing.  Acceptance below still uses read_plate().
            fast = self._run_fastplate(crop, frame_number, idx)
            row["fast_text"] = fast["text"]
            row["fast_conf"] = fast["conf"]
            self._fast_trials.append({
                "frame": frame_number, "track_id": track_id,
                "det_source": det_source,
                "plate_bbox": str([int(v) for v in best["bbox"]]),
                "det_conf": det_conf, "crop_w": crop_w, "crop_h": crop_h,
                "text": fast["text"], "conf": fast["conf"],
                "normalized": fast["normalized"],
                "regex_pass": fast["regex_pass"], "status": fast["status"],
            })
            if fast["status"] == "ok" and fast["text"]:
                self._fast_top_images.append({
                    "conf": fast["conf"] if fast["conf"] is not None else -1.0,
                    "image": crop,
                    "frame": frame_number, "track_id": track_id,
                    "text": fast["text"], "det_conf": det_conf,
                    "normalized": fast["normalized"],
                })
            logger.info(
                "[FASTPLATE] %s frame=%d veh=%d text='%s' conf=%s regex=%s status=%s",
                self.camera_id, frame_number, idx, fast["text"],
                fast["conf"], fast["regex_pass"], fast["status"],
            )
            # Track-level observation stream (both backends, source-tagged).
            # Voting/deque semantics untouched; acceptance uses read_plate().

            self._diag["ocr_attempted"] += 1
            ocr_result = self.ocr.read_plate(crop, track_key=track_key)
            text = ocr_result["text"]
            conf = ocr_result["confidence"]
            row["accept_text"] = text
            row["accept_conf"] = conf
            try:
                row["corrected"] = position_correct(text) if text else ""
            except Exception:
                logger.exception("[LEDGER] position_correct failed")
                row["corrected"] = text
            row["regex_pass"] = bool(text) and looks_like_plate(text)

            # Track-level observation stream (both backends, source-tagged).
            # Placed AFTER the acceptance read so text/conf exist.  Logging
            # never alters voting/deque semantics.
            ts_obs = datetime.now().timestamp()
            if text:
                self.ocr.log_observation(text, conf, track_key,
                                         source="easyocr",
                                         frame=frame_number, timestamp=ts_obs)
            if fast["text"]:
                self.ocr.log_observation(
                    fast["text"],
                    fast["conf"] if fast["conf"] is not None else 0.0,
                    track_key, source="fastplate",
                    frame=frame_number, timestamp=ts_obs)

            entry = {
                "bbox": [pbx1, pby1, pbx2, pby2],
                "confidence": best["confidence"],
                "plate_text": text,
                "ocr_confidence": conf,
                "vehicle_idx": idx,
                "track_id": track_id,
                "source": "plate_model",
            }
            plate_entries.append(entry)

            if not text:
                # OCR returned nothing: counted explicitly (was previously
                # an uncounted silent branch).  Thresholds unchanged.
                logger.info(
                    "[OCR] %s frame=%d veh=%d plate found but OCR read nothing",
                    self.camera_id, frame_number, idx,
                )
                self._diag["ocr_empty"] += 1
                row["final_reason"] = "ocr_empty"
                self._save_candidate_set("ocr_empty", frame, crop, variants,
                                         [pbx1, pby1, pbx2, pby2], veh["bbox"],
                                         frame_number, idx)
                self._record_candidate(row)
                continue

            # Gate 1 (UNCHANGED): OCR confidence >= min_ocr_confidence.
            # below_gate fires only here — non-empty text under threshold.
            if conf < self.min_ocr_confidence:
                logger.info(
                    "[OCR] %s frame=%d veh=%d plate found but below confidence "
                    "gate '%s' (conf=%.2f < %.2f)",
                    self.camera_id, frame_number, idx,
                    text, conf, self.min_ocr_confidence,
                )
                self._diag["plate_found_below_gate"] += 1
                self._diag["ocr_low_conf"] += 1
                row["final_reason"] = "ocr_low_conf"
                self._save_candidate_set("low_conf", frame, crop, variants,
                                         [pbx1, pby1, pbx2, pby2], veh["bbox"],
                                         frame_number, idx)
                self._record_candidate(row)
                continue

            # First gate cleared: confidence.  Record it before the second
            # (format) gate so "before vs after" is quantifiable.
            self._diag["plate_conf_cleared"] += 1

            # Gate 2 (UNCHANGED), alongside Gate 1: Indian plate format.
            # format_rejected fires only here — confident but non-plate text.
            if not looks_like_plate(text):
                logger.info(
                    "[OCR] %s frame=%d veh=%d plate found but format check "
                    "failed '%s' (conf=%.2f >= %.2f)",
                    self.camera_id, frame_number, idx,
                    text, conf, self.min_ocr_confidence,
                )
                self._diag["plate_confidence_cleared_format_rejected"] += 1
                self._diag["regex_rejected"] += 1
                samples = self._diag["plate_format_rejected_samples"]
                if len(samples) < 30:
                    samples.append(text)
                row["final_reason"] = "regex_rejected"
                self._save_candidate_set("regex_rejected", frame, crop, variants,
                                         [pbx1, pby1, pbx2, pby2], veh["bbox"],
                                         frame_number, idx)
                self._record_candidate(row)
                continue

            # Both gates cleared (UNCHANGED conditions).
            read_count += 1
            self._diag["plate_gate_cleared"] += 1
            self._diag["accepted"] += 1
            row["final_reason"] = "accepted"
            self._save_candidate_set("accepted", frame, crop, variants,
                                     [pbx1, pby1, pbx2, pby2], veh["bbox"],
                                     frame_number, idx)
            self._record_candidate(row)
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
                    "plate_bbox": [pbx1, pby1, pbx2, pby2],
                    "frame_number": frame_number,
                    "track_id": track_id,
                    "direction": self.direction,
                    "class_name": veh["class_name"],
                }
            )

        return {"plate_entries": plate_entries, "sightings": sightings, "read_count": read_count}

    def _maybe_save_debug(self, category: str, crop, frame_number: int, veh_idx: int,
                            max_per_category: int = 5) -> None:
        """Save a bounded number of plate crops per category for inspection.

        No-op unless ``debug_crops_dir`` was configured.  Saves at most
        ``max_per_category`` images per category so a long run cannot fill
        the disk/Drive.
        """
        if self.debug_crops_dir is None:
            return
        try:
            n = self._debug_saved.get(category, 0)
            if n >= max_per_category:
                return
            safe = "".join(c if c.isalnum() or c in "_x" else "_" for c in category)[:40]
            out = self.debug_crops_dir / f"{self.camera_id}_f{frame_number}_v{veh_idx}_{safe}_{n}.jpg"
            cv2.imwrite(str(out), crop)
            self._debug_saved[category] = n + 1
        except Exception:
            logger.exception("[DEBUG] failed to save debug crop")

    def _diagnose_variants(self, crop, frame_number: int, veh_idx: int) -> dict[str, Any]:
        """Run diagnostic-only OCR variants on *crop* (no vote history touched).

        Compares raw 3x-upscaled vs full CLAHE+upscale+unsharp preprocessing
        using direct EasyOCR calls + ``_extract_text``.  Failures are logged
        with ``traceback`` and recorded — never raised, never silent — so one
        bad crop cannot kill a run, but nothing is hidden either.
        """
        import traceback

        out = {"raw_text": "", "raw_conf": 0.0, "pre_text": "", "pre_conf": 0.0,
               "raw_up": None, "pre": None, "error": ""}
        try:
            h, w = crop.shape[:2]
            scale = 3.0 if max(h, w) < 400 else 160.0 / max(h, w)
            raw_up = cv2.resize(crop, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
            out["raw_up"] = raw_up
            raw_res = self.ocr.reader.readtext(raw_up, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            t, c = self.ocr._extract_text(raw_res)
            out["raw_text"], out["raw_conf"] = t, float(c)
        except Exception:
            traceback.print_exc()
            out["error"] += "raw_variant_failed; "
            logger.exception("[DIAG] %s frame=%d veh=%d raw-variant OCR failed",
                             self.camera_id, frame_number, veh_idx)
        try:
            pre = self.ocr.preprocess_plate(crop)
            out["pre"] = pre
            pre_res = self.ocr.reader.readtext(pre, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            t, c = self.ocr._extract_text(pre_res)
            out["pre_text"], out["pre_conf"] = t, float(c)
        except Exception:
            traceback.print_exc()
            out["error"] += "pre_variant_failed; "
            logger.exception("[DIAG] %s frame=%d veh=%d pre-variant OCR failed",
                             self.camera_id, frame_number, veh_idx)
        return out

    def _record_candidate(self, row: dict[str, Any]) -> None:
        """Append one candidate row to the ledger with a compact log line."""
        self._ledger.append(row)
        logger.info(
            "[CANDIDATE] %s frame=%d track=%s src=%s bbox=%s det_conf=%.3f(>=%.2f) "
            "crop=%sx%s aspect=%s bumper=%s | raw='%s'@%.3f pre='%s'@%.3f "
            "accept='%s'@%.3f corrected='%s' regex=%s thr=%.2f -> %s",
            self.camera_id, row["frame"], row["track_id"], row["det_source"],
            row["plate_bbox"], row["det_conf"], row["det_threshold"],
            row["crop_w"], row["crop_h"], row["aspect"], row["bumper_passed"],
            row["ocr_raw_text"], row["ocr_raw_conf"],
            row["ocr_pre_text"], row["ocr_pre_conf"],
            row["accept_text"], row["accept_conf"],
            row["corrected"], row["regex_pass"], row["ocr_threshold"],
            row["final_reason"],
        )

    def _save_candidate_set(self, reason: str, frame, crop, variants,
                            pbbox: list[int], vbbox: list[int],
                            frame_number: int, veh_idx: int) -> None:
        """Save raw/up/pre/annotated images for one candidate (bounded)."""
        if self.debug_crops_dir is None:
            return
        try:
            self._maybe_save_debug(f"{reason}_raw", crop, frame_number, veh_idx)
            if variants is not None:
                if variants.get("raw_up") is not None:
                    self._maybe_save_debug(f"{reason}_up", variants["raw_up"],
                                           frame_number, veh_idx)
                if variants.get("pre") is not None:
                    self._maybe_save_debug(f"{reason}_pre", variants["pre"],
                                           frame_number, veh_idx)
            ann = frame.copy()
            x1, y1, x2, y2 = [int(v) for v in vbbox]
            ann = cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
            px1, py1, px2, py2 = [int(v) for v in pbbox]
            ann = cv2.rectangle(ann, (px1, py1), (px2, py2), (0, 0, 255), 2)
            self._maybe_save_debug(f"{reason}_ann", ann, frame_number, veh_idx)
        except Exception:
            import traceback
            traceback.print_exc()
            logger.exception("[DEBUG] candidate-set save failed")

    def _save_ledger_csv(self) -> Optional[str]:
        """Persist the candidate ledger as CSV next to the debug crops."""
        import csv

        if not self._ledger:
            return None
        out_dir = self.debug_crops_dir or (Path("data") / "debug_ocr")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"candidate_ledger_{self.camera_id}.csv"
        cols = ["frame", "track_id", "det_source", "plate_bbox", "det_conf",
                "det_threshold", "crop_w", "crop_h", "aspect", "bumper_passed",
                "ocr_raw_text", "ocr_raw_conf", "ocr_pre_text", "ocr_pre_conf",
                "accept_text", "accept_conf", "corrected", "regex_pass",
                "best_variant", "best_variant_text", "best_variant_conf",
                "fast_text", "fast_conf",
                "ocr_threshold", "variant_error", "direct_error", "final_reason"]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(self._ledger)
            logger.info("[LEDGER] wrote %d rows to %s", len(self._ledger), path)
            return str(path)
        except Exception:
            import traceback
            traceback.print_exc()
            logger.exception("[LEDGER] CSV save failed")
            return None

    def _run_fastplate(self, crop, frame_number: int, veh_idx: int) -> dict[str, Any]:
        """Run fast-plate-ocr on the RAW crop (diagnostic ONLY).

        Lazily builds the backend (model downloads once on first use).
        Never raises: failures are logged with tracebacks and recorded.
        """
        import traceback

        out = {"text": "", "conf": None, "normalized": "",
               "regex_pass": False, "status": ""}
        try:
            if self.fast_ocr is None:
                from src.ocr.fastplate import FastPlateOCR
                self.fast_ocr = FastPlateOCR()
            res = self.fast_ocr.read_crop(crop)
            out["text"] = res["text"]
            out["conf"] = res["conf"]
            out["status"] = res["status"]
            norm = self.ocr._normalize_confusions(res["text"]) if res["text"] else ""
            out["normalized"] = norm
            out["regex_pass"] = looks_like_plate(norm)
        except Exception:
            traceback.print_exc()
            logger.exception("[FASTPLATE] %s frame=%d veh=%d failed",
                             self.camera_id, frame_number, veh_idx)
            out["status"] = "failed"
        return out

    def _save_top5(self) -> Optional[str]:
        """Save the top-5 direct-recognition trials by OCR confidence.

        Writes 5 variant images + ``top5_direct.csv`` (frame, bbox, det
        conf, variant, text, conf, normalized, regex) to Drive for visual
        inspection.  Bounded by construction: never more than 5 images.
        """
        import csv

        if not self._top_images:
            return None
        out_dir = (self.debug_crops_dir or (Path("data") / "debug_ocr")) / "top5"
        out_dir.mkdir(parents=True, exist_ok=True)
        ranked = sorted(self._top_images, key=lambda e: e["conf"], reverse=True)[:5]
        try:
            for i, e in enumerate(ranked):
                cv2.imwrite(
                    str(out_dir / f"top{i + 1}_{e['variant']}_conf{e['conf']:.3f}"
                                   f"_f{e['frame']}_t{e['track_id']}.jpg"),
                    e["image"],
                )
            csv_path = out_dir / "top5_direct.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["frame", "track_id", "variant", "text", "conf",
                                "normalized", "det_conf"],
                    extrasaction="ignore",
                )
                w.writeheader()
                for e in ranked:
                    w.writerow(e)
            logger.info("[DIRECT] saved top-5 to %s", out_dir)
            return str(csv_path)
        except Exception:
            import traceback
            traceback.print_exc()
            logger.exception("[DIRECT] top-5 save failed")
            return None

    def _save_fast_top5(self) -> Optional[str]:
        """Save the top-5 fast-plate-ocr trials by confidence to Drive.

        Writes 5 raw-crop images + ``fast_top5.csv`` (frame, det conf,
        crop size, text, conf, normalized, regex).  Bounded: max 5 images.
        """
        import csv

        if not self._fast_top_images:
            return None
        out_dir = (self.debug_crops_dir or (Path("data") / "debug_ocr")) / "fast_top5"
        out_dir.mkdir(parents=True, exist_ok=True)
        ranked = sorted(self._fast_top_images, key=lambda e: e["conf"], reverse=True)[:5]
        try:
            for i, e in enumerate(ranked):
                c = e["conf"]
                cstr = f"{c:.4f}" if c is not None else "noconf"
                cv2.imwrite(
                    str(out_dir / f"top{i + 1}_conf{cstr}"
                                   f"_f{e['frame']}_t{e['track_id']}.jpg"),
                    e["image"],
                )
            csv_path = out_dir / "fast_top5.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["frame", "track_id", "text", "conf",
                                "normalized", "det_conf"],
                    extrasaction="ignore",
                )
                w.writeheader()
                for e in ranked:
                    w.writerow(e)
            logger.info("[FASTPLATE] saved top-5 to %s", out_dir)
            return str(csv_path)
        except Exception:
            import traceback
            traceback.print_exc()
            logger.exception("[FASTPLATE] top-5 save failed")
            return None

    def _print_track_fusion_report(self) -> list[dict]:
        """Fuse fastplate observations per track with the EXISTING fusion.

        Prints track_id, observation count, individual predictions +
        confidences, fused result + confidence, regex verdict and
        accept/reject reason.  Report-only: no sightings written, no
        thresholds changed.  Stability is NEVER labeled correctness.
        Returns the per-track reports.
        """
        from collections import Counter

        keys = sorted({o["track_key"] for o in self.ocr._observations
                       if o["source"] == "fastplate"})
        reports = []
        lines = ["", "===== TRACK-LEVEL FASTPLATE FUSION (UNVERIFIED) =====",
                 "Stability across frames is NOT correctness. Nothing below",
                 "was accepted as ground truth or written as a sighting."]
        for key in keys:
            rep = self.ocr.track_fusion_report(key, source="fastplate",
                                               min_confidence=self.min_ocr_confidence)
            reports.append(rep)
            tid = key[1] if len(key) > 1 else key
            preds = ", ".join(f"'{t}'@{c:.3f}(f{f})" for t, c, f in rep["predictions"])
            lines.append(
                f"track={tid} n={rep['n_observations']} [{preds}] "
                f"-> fused='{rep['fused']}' conf={rep['fused_confidence']} "
                f"({rep['method']}) regex={rep['regex_pass']} verdict={rep['verdict']}"
            )
        if not reports:
            lines.append("no fastplate observations logged")
        easy_n = sum(1 for o in self.ocr._observations if o["source"] == "easyocr")
        lines.append(f"easyocr observations logged: {easy_n} (acceptance path unchanged)")
        lines.append("======================================================")
        text = "\n".join(lines)
        print(text)
        logger.info(text)
        return reports

    def _save_stable_tracks(self, reports: list[dict]) -> Optional[str]:
        """Save crops + CSV for tracks stable across frames (still UNVERIFIED).

        Stable = >=3 fastplate observations with a most-common string seen
        >=3 times.  Up to 3 highest-confidence crops per stable track
        (max 5 tracks).  Images come from the diagnostic fastplate set.
        """
        import csv
        from collections import Counter

        stable = []
        for rep in reports:
            if rep["n_observations"] < 3:
                continue
            top, cnt = Counter(t for t, _, _ in rep["predictions"]).most_common(1)[0]
            if cnt >= 3:
                stable.append((rep, top, cnt))
        stable = stable[:5]
        if not stable:
            logger.info("[STABLE] no stable tracks")
            return None
        out_dir = (self.debug_crops_dir or (Path("data") / "debug_ocr")) / "stable_tracks"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            for rep, top, cnt in stable:
                tid = rep["track_key"][1] if len(rep["track_key"]) > 1 else rep["track_key"]
                cand = sorted(
                    (e for e in self._fast_top_images if e["track_id"] == tid),
                    key=lambda e: e["conf"], reverse=True,
                )[:3]
                for i, e in enumerate(cand):
                    cv2.imwrite(
                        str(out_dir / f"track{tid}_{i + 1}_conf{e['conf']:.3f}"
                                       f"_f{e['frame']}.jpg"),
                        e["image"],
                    )
            csv_path = out_dir / "stable_tracks.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=["track_id", "n_observations", "top_string",
                                   "top_count", "fused", "fused_confidence",
                                   "regex_pass", "verdict", "verified"],
                    extrasaction="ignore",
                )
                w.writeheader()
                for rep, top, cnt in stable:
                    tid = rep["track_key"][1] if len(rep["track_key"]) > 1 else rep["track_key"]
                    w.writerow({"track_id": tid, "n_observations": rep["n_observations"],
                                "top_string": top, "top_count": cnt,
                                "fused": rep["fused"],
                                "fused_confidence": rep["fused_confidence"],
                                "regex_pass": rep["regex_pass"],
                                "verdict": rep["verdict"], "verified": False})
            logger.info("[STABLE] saved %d stable track(s) to %s", len(stable), out_dir)
            return str(csv_path)
        except Exception:
            import traceback
            traceback.print_exc()
            logger.exception("[STABLE] save failed")
            return None

    def _print_ledger_summary(self) -> None:
        """Print det/OCR confidence distributions + gate survival + top reasons."""
        from collections import Counter

        d = self._diag
        det_confs = [r["det_conf"] for r in self._ledger if r.get("det_conf") is not None]
        ocr_confs = [r["accept_conf"] for r in self._ledger if r.get("accept_text")]
        reasons = Counter(r["final_reason"] for r in self._ledger)

        def _dist(xs: list[float]) -> str:
            if not xs:
                return "n/a (no samples)"
            s = sorted(xs)
            return (f"n={len(s)} min={s[0]:.3f} mean={sum(s)/len(s):.3f} "
                    f"max={s[-1]:.3f} values={[round(v, 3) for v in s]}")

        lines = [
            "",
            "===== CANDIDATE LEDGER SUMMARY (thresholds UNCHANGED) =====",
            f"Plate-detector threshold         : {self.detector.plate_conf_threshold}",
            f"OCR confidence threshold         : {self.min_ocr_confidence} "
            "(below_gate = non-empty text with conf BELOW this)",
            f"gate_cleared requires            : conf >= {self.min_ocr_confidence} AND regex pass",
            f"format_rejected requires         : conf >= {self.min_ocr_confidence} AND regex FAIL",
            f"Plate-detector conf distribution : {_dist(det_confs)}",
            f"OCR accept-conf distribution     : {_dist(ocr_confs)}",
            "Gate survival:",
            f"  candidate_boxes={d['candidate_boxes']} aspect_rejected={d['aspect_rejected']} "
            f"position_rejected={d['position_rejected']} tiny_rejected={d['tiny_rejected']} "
            f"detector_conf_rejected={d['detector_conf_rejected']}",
            f"  ocr_attempted={d['ocr_attempted']} ocr_empty={d['ocr_empty']} "
            f"ocr_low_conf={d['ocr_low_conf']} regex_rejected={d['regex_rejected']} "
            f"accepted={d['accepted']}",
            "Top rejection reasons:",
        ]
        for reason, n in reasons.most_common():
            lines.append(f"  {reason}: {n}")
        lines.append("===========================================================")
        text = "\n".join(lines)
        print(text)
        logger.info(text)

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