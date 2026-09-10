"""Per-camera ANPR pipeline runner.

Processes video frames through the full detection → OCR → tracking →
fusion → DB pipeline for a single camera feed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.detection.detector import PlateDetector
from src.ocr.reader import PlateOCR
from src.tracking.tracker import VehicleTracker
from src.fusion.reid import CrossCameraFusion
from src.alerts.engine import AlertEngine
from src.db.schema import get_connection, init_db
from src.analytics.engine import TrafficAnalytics

logger = logging.getLogger(__name__)

_FORMAT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"


class PipelineRunner:
    """Runs the full ANPR pipeline for a single camera.

    Args:
        camera_id: Identifier for this camera (e.g. ``"cam_1"``).
        video_path: Path to the input video file.
        camera_config: Dict with at least ``gps_lat`` and ``gps_lon`` keys.
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
        self.camera_config = camera_config
        self.device = device
        self.min_ocr_confidence = min_ocr_confidence
        self.min_vehicle_height_px = min_vehicle_height_px

        self.gps_lat: float = camera_config.get("gps_lat", 0.0)
        self.gps_lon: float = camera_config.get("gps_lon", 0.0)

        logger.info(
            "Initialising pipeline runner for camera %s (video=%s)",
            self.camera_id,
            self.video_path,
        )

        self.detector = PlateDetector(device=self.device)
        self.ocr = PlateOCR()
        self.tracker = VehicleTracker()
        self.fusion = CrossCameraFusion()
        self.alert_engine = AlertEngine()
        self.analytics = TrafficAnalytics()

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
    ) -> dict[str, Any]:
        """Open the video and process frames through the full pipeline.

        Args:
            speed_factor: Playback speed multiplier.  ``1.0`` attempts
                real-time pacing; ``0`` disables sleeping.
            max_frames: Stop after this many frames.  *None* means
                process until end-of-video.

        Returns:
            Summary dict with keys ``frames_processed``, ``vehicles_detected``,
            ``plates_read``, ``alerts_triggered``.
        """
        if not self.video_path.exists():
            logger.error("Video file not found: %s", self.video_path)
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        init_db()

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
        """Process a single frame through detection, OCR, tracking, and alerts.

        Args:
            frame: BGR ``uint8`` image (e.g. from ``cv2.VideoCapture.read``).
            frame_number: Current frame index in the video.

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

        vehicles_detected = len(vehicles)
        plates_read = 0
        alerts_triggered = 0

        ocr_plates: list[dict[str, Any]] = []

        # 2. OCR on each detected plate
        for plate_det in plates:
            x1, y1, x2, y2 = plate_det["bbox"]
            h, w = frame.shape[:2]
            cx1 = max(0, int(x1))
            cy1 = max(0, int(y1))
            cx2 = min(w, int(x2))
            cy2 = min(h, int(y2))

            if cx2 <= cx1 or cy2 <= cy1:
                logger.debug("Skipping degenerate plate crop at frame %d", frame_number)
                continue

            crop = frame[cy1:cy2, cx1:cx2]
            ocr_result = self.ocr.read_plate(crop)
            plate_text = ocr_result["text"]
            plate_confidence = ocr_result["confidence"]

            plate_entry = dict(plate_det)
            plate_entry["plate_text"] = plate_text
            plate_entry["ocr_confidence"] = plate_confidence
            ocr_plates.append(plate_entry)

            if plate_text:
                plates_read += 1

                sighting = {
                    "plate": plate_text,
                    "camera_id": self.camera_id,
                    "gps_lat": self.gps_lat,
                    "gps_lon": self.gps_lon,
                    "timestamp": ts,
                    "confidence": plate_confidence,
                    "vehicle_bbox": vehicles[plate_det["vehicle_idx"]]["bbox"]
                    if plate_det.get("vehicle_idx", -1) >= 0
                    and plate_det["vehicle_idx"] < len(vehicles)
                    else None,
                    "plate_bbox": plate_det["bbox"],
                    "frame_number": frame_number,
                }

                self._insert_sighting(sighting)
                self.fusion.add_sighting(
                    plate=plate_text,
                    camera_id=self.camera_id,
                    gps_lat=self.gps_lat,
                    gps_lon=self.gps_lon,
                    timestamp=ts,
                    confidence=plate_confidence,
                )

                # 3. Check alerts
                alerts = self.alert_engine.check_plate(
                    plate=plate_text,
                    camera_id=self.camera_id,
                    gps_lat=self.gps_lat,
                    gps_lon=self.gps_lon,
                    timestamp=ts,
                )
                alerts_triggered += len(alerts)
                self._alerts.extend(alerts)

        # 2b. Plate-crop fallback: when no plate model exists, crop the
        # lower-center region of each vehicle bbox (where plates sit) and OCR.
        if not plates and vehicles:
            crop_reads = self._ocr_vehicle_plate_crops(frame, vehicles, frame_number)
            ocr_plates.extend(crop_reads["plate_entries"])
            plates_read += crop_reads["read_count"]

            for sighting in crop_reads["sightings"]:
                self._insert_sighting(sighting)
                self.fusion.add_sighting(
                    plate=sighting["plate"],
                    camera_id=self.camera_id,
                    gps_lat=self.gps_lat,
                    gps_lon=self.gps_lon,
                    timestamp=sighting["timestamp"],
                    confidence=sighting["confidence"],
                )
                alerts = self.alert_engine.check_plate(
                    plate=sighting["plate"],
                    camera_id=self.camera_id,
                    gps_lat=self.gps_lat,
                    gps_lon=self.gps_lon,
                    timestamp=sighting["timestamp"],
                )
                alerts_triggered += len(alerts)
                self._alerts.extend(alerts)

        # 4. Tracking
        tracked = self.tracker.update(vehicles, frame=frame)

        # 5. Analytics
        self.analytics.update_camera_count(
            self.camera_id, ts, len(vehicles), frame_shape=frame.shape
        )
        self._analytics_entries.append(
            {
                "camera_id": self.camera_id,
                "timestamp": ts,
                "vehicle_count": len(vehicles),
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

    def _ocr_vehicle_plate_crops(
        self, frame: np.ndarray, vehicles: list[dict], frame_number: int
    ) -> dict[str, Any]:
        """OCR the lower-center crop of each vehicle bbox (plate location).

        No dedicated plate-detection model is available, so we crop the
        region of each vehicle where number plates typically sit (bottom
        ~20% of the bbox) and run OCR on the real pixels.
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

            # Upscale 2x so OCR can see small plate glyphs
            upscale = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            ocr_result = self.ocr.read_plate(upscale)
            text = ocr_result["text"]
            conf = ocr_result["confidence"]

            entry = {
                "bbox": [cx1, cy1, cx2, cy2],
                "confidence": 0.0,
                "plate_text": text,
                "ocr_confidence": conf,
                "vehicle_idx": idx,
                "source": "vehicle_crop_fallback",
            }
            plate_entries.append(entry)

            # Only accept confident plate readings (anti-hallucination gate)
            if text and conf >= self.min_ocr_confidence:
                read_count += 1
                ts = datetime.now().timestamp()
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
                        "class_name": veh["class_name"],
                    }
                )
            elif text:
                logger.debug(
                    "Rejected low-confidence plate reading '%s' (conf=%.2f < %.2f)",
                    text,
                    conf,
                    self.min_ocr_confidence,
                )

        if read_count:
            logger.info(
                "Frame %d: plate-crop fallback read %d plate(s) from %d vehicle crop(s) "
                "(%d candidates)",
                frame_number,
                read_count,
                len(plate_entries),
                candidates,
            )

        return {"plate_entries": plate_entries, "sightings": sightings, "read_count": read_count}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_sighting(sighting: dict[str, Any]) -> None:
        """Write a sighting row to the database."""
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO sightings
                    (plate, plate_confidence, camera_id, gps_lat, gps_lon,
                     timestamp, vehicle_class, vehicle_bbox, plate_bbox, frame_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            conn.commit()
            logger.debug("Sighting inserted: %s on %s", sighting["plate"], sighting["camera_id"])
        finally:
            conn.close()

    def _flush_analytics(self) -> None:
        """Persist accumulated analytics to the database."""
        if not self._analytics_entries:
            return

        summary = self.analytics.get_analytics_summary()
        cam_summary = summary.get(self.camera_id, {})

        avg_speed = cam_summary.get("avg_speed_kmh", 0.0)
        density = cam_summary.get("density", 0.0)
        congestion = cam_summary.get("congestion", {})
        level = congestion.get("congestion_level", "low")
        is_congested = congestion.get("is_congested", False)

        if self._analytics_entries:
            last = self._analytics_entries[-1]
            self.analytics.store_analytics(
                camera_id=self.camera_id,
                timestamp=last["timestamp"],
                vehicle_count=last["vehicle_count"],
                avg_speed=avg_speed,
                density_level=level,
                congestion_flag=is_congested,
            )
            logger.info(
                "Analytics stored for %s: vehicles=%d avg_speed=%.1f density=%s",
                self.camera_id,
                last["vehicle_count"],
                avg_speed,
                level,
            )


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
