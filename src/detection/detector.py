from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs for vehicles
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_VEHICLE_CLASS_IDS = set(VEHICLE_CLASSES.keys())

_DEFAULT_VEHICLE_MODEL = "yolov8n.pt"

# Pretrained license-plate detector fetched from Hugging Face.
# Koushim/yolov8-license-plate-detection is a YOLOv8n fine-tuned for a single
# class ("license_plate"); the weights file is ``best.pt`` (confirmed on the
# model card / repo file listing).
HF_PLATE_MODEL_REPO_ID = "Koushim/yolov8-license-plate-detection"
HF_PLATE_MODEL_FILENAME = "best.pt"


def resolve_plate_model_from_hub(
    repo_id: str = HF_PLATE_MODEL_REPO_ID,
    filename: str = HF_PLATE_MODEL_FILENAME,
) -> Path:
    """Download (or fetch cached) plate-detection weights from Hugging Face.

    Uses ``huggingface_hub.hf_hub_download`` so the weights are cached by hub
    and only fetched once.  Returns the local path to the weights file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to fetch the license-plate detector. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info("License-plate detector weights fetched from %s/%s (%s)",
                repo_id, filename, path)
    return Path(path)


class PlateDetector:
    """YOLOv8-based detector for vehicles and license plates.

    Plate localization uses a dedicated plate model (fetched from Hugging Face
    by default) run on each vehicle crop via :meth:`detect_plates_in_crop`.
    There is no heuristic fallback: when no plate box clears the plate
    confidence threshold, the caller must skip OCR for that vehicle.
    """

    def __init__(
        self,
        vehicle_model_path: Optional[str | Path] = None,
        plate_model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        vehicle_conf_threshold: float = 0.25,
        plate_conf_threshold: float = 0.25,
        vehicle_iou_threshold: float = 0.45,
        plate_iou_threshold: float = 0.45,
    ) -> None:
        """Initialise detector and load models.

        Parameters
        ----------
        vehicle_model_path:
            Path to the YOLO weights for vehicle detection.  When *None*
            the bundled ``yolov8n.pt`` is used.
        plate_model_path:
            Optional path to a YOLO weights file fine-tuned for plate
            detection.  When *None* plate detection is skipped at this
            stage.
        device:
            Compute device, e.g. ``"cpu"``, ``"cuda:0"``.  When *None*
            the best available device is selected automatically.
        vehicle_conf_threshold:
            Minimum confidence for a vehicle detection.
        plate_conf_threshold:
            Minimum confidence for a plate detection.
        vehicle_iou_threshold:
            IoU threshold for NMS on vehicle detections.
        plate_iou_threshold:
            IoU threshold for NMS on plate detections.
        """
        self._vehicle_conf = vehicle_conf_threshold
        self._plate_conf = plate_conf_threshold
        self._vehicle_iou = vehicle_iou_threshold
        self._plate_iou = plate_iou_threshold

        # ------------------------------------------------------------------
        # Resolve device
        # ------------------------------------------------------------------
        self._device = self._resolve_device(device)
        logger.info("Using device: %s", self._device)

        # ------------------------------------------------------------------
        # Load vehicle model
        # ------------------------------------------------------------------
        vehicle_path = self._resolve_model_path(vehicle_model_path, _DEFAULT_VEHICLE_MODEL)
        self._vehicle_model = self._load_yolo(vehicle_path)
        logger.info("Vehicle model loaded from %s", vehicle_path)

        # ------------------------------------------------------------------
        # Load plate model (optional)
        # ------------------------------------------------------------------
        self._plate_model = None
        if plate_model_path is not None:
            resolved_plate = self._resolve_model_path(plate_model_path, None)
            self._plate_model = self._load_yolo(resolved_plate)
            logger.info("Plate model loaded from %s", resolved_plate)
        else:
            logger.info("No plate model provided; plate detection disabled.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def plate_conf_threshold(self) -> float:
        """Minimum confidence for a plate detection (the "plate box" gate)."""
        return self._plate_conf

    def detect_vehicles(self, frame: np.ndarray) -> list[dict]:
        """Detect vehicles in *frame*.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``bbox``: ``[x1, y1, x2, y2]`` pixel coordinates.
            - ``confidence``: detection confidence.
            - ``class_name``: human-readable class label.
            - ``class_id``: COCO class id.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to detect_vehicles; returning empty list.")
            return []

        t0 = time.perf_counter()
        results = self._vehicle_model.predict(
            source=frame,
            conf=self._vehicle_conf,
            iou=self._vehicle_iou,
            device=self._device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Vehicle inference took %.1f ms", elapsed_ms)

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in _VEHICLE_CLASS_IDS:
                    continue
                xyxy = boxes.xyxy[i].cpu().numpy().astype(float).tolist()
                conf = float(boxes.conf[i].item())
                detections.append(
                    {
                        "bbox": [int(round(v)) for v in xyxy],
                        "confidence": conf,
                        "class_name": VEHICLE_CLASSES[cls_id],
                        "class_id": cls_id,
                    }
                )
        logger.info(
            "Detected %d vehicle(s) in %.1f ms",
            len(detections),
            elapsed_ms,
        )
        return detections

    def detect_plates(
        self,
        frame: np.ndarray,
        vehicle_detections: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Detect license plates in *frame*.

        When a dedicated plate model is loaded it is used directly.  When
        no plate model is available the method returns an empty list;
        downstream code should attempt OCR on vehicle crops instead.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``bbox``: ``[x1, y1, x2, y2]`` pixel coordinates.
            - ``confidence``: detection confidence.
            - ``plate_text``: always ``""`` at detection stage.
            - ``vehicle_idx``: index into the *vehicle_detections* list
              that this plate is associated with, or ``-1`` when
              association is not possible.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to detect_plates; returning empty list.")
            return []

        detections: list[dict] = []

        if self._plate_model is not None:
            detections = self._detect_plates_model(frame)

            # Attempt to associate each plate with a vehicle via IoU.
            if vehicle_detections:
                for plate in detections:
                    plate["vehicle_idx"] = self._best_vehicle_match(
                        plate["bbox"], vehicle_detections
                    )
            else:
                for plate in detections:
                    plate["vehicle_idx"] = -1

            logger.info("Detected %d plate(s) via plate model.", len(detections))
        else:
            logger.debug("No plate model available; skipping plate detection.")

        return detections

    def detect_plates_in_crop(
        self,
        frame: np.ndarray,
        vehicle_bbox: list[int],
        pad_ratio: float = 0.10,
    ) -> list[dict]:
        """Localize the license plate within a single vehicle crop.

        The dedicated plate model is run on the crop of *vehicle_bbox* (not
        the full frame) so the plate box is found within the vehicle.  The
        vehicle box is expanded by ``pad_ratio`` on each side before cropping
        so plates near the edge of the vehicle detection are not lost.

        There is **no heuristic fallback**: if the model cannot localize a
        plate above the configured confidence threshold (0.25) an empty list
        is returned and the caller must skip OCR for that vehicle.

        Returns
        -------
        list[dict]
            Plate boxes in **full-frame** pixel coordinates; each dict is the
            same shape as :meth:`detect_plates` (``bbox``, ``confidence``,
            ``plate_text``, ``vehicle_idx``).
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to detect_plates_in_crop.")
            return []
        if self._plate_model is None:
            raise RuntimeError(
                "detect_plates_in_crop requires a dedicated plate model; "
                "none is loaded."
            )

        h_max, w_max = frame.shape[:2]
        x1, y1, x2, y2 = vehicle_bbox
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)

        cx1 = max(0, int(x1) - pad_x)
        cy1 = max(0, int(y1) - pad_y)
        cx2 = min(w_max, int(x2) + pad_x)
        cy2 = min(h_max, int(y2) + pad_y)
        if cx2 <= cx1 or cy2 <= cy1:
            logger.debug("Degenerate vehicle crop; skipping plate detection.")
            return []

        crop = frame[cy1:cy2, cx1:cx2]
        dets = self._detect_plates_model(crop)
        for det in dets:
            bx1, by1, bx2, by2 = det["bbox"]
            # Offset the crop-relative box back into full-frame coordinates.
            det["bbox"] = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]
        return dets

    def detect(self, frame: np.ndarray) -> dict:
        """Run full pipeline: vehicles then plates.

        Returns
        -------
        dict
            ``{"vehicles": [...], "plates": [...]}``
        """
        t0 = time.perf_counter()
        vehicles = self.detect_vehicles(frame)
        plates = self.detect_plates(frame, vehicle_detections=vehicles)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Frame processed: %d vehicle(s), %d plate(s) — %.1f ms total",
            len(vehicles),
            len(plates),
            elapsed_ms,
        )
        return {"vehicles": vehicles, "plates": plates}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_plates_model(self, frame: np.ndarray) -> list[dict]:
        """Run the plate YOLO model on *frame*."""
        assert self._plate_model is not None
        t0 = time.perf_counter()
        results = self._plate_model.predict(
            source=frame,
            conf=self._plate_conf,
            iou=self._plate_iou,
            device=self._device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Plate inference took %.1f ms", elapsed_ms)

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(float).tolist()
                conf = float(boxes.conf[i].item())
                detections.append(
                    {
                        "bbox": [int(round(v)) for v in xyxy],
                        "confidence": conf,
                        "plate_text": "",
                        "vehicle_idx": -1,
                    }
                )
        return detections

    @staticmethod
    def _best_vehicle_match(
        plate_bbox: list[int],
        vehicle_detections: list[dict],
        iou_threshold: float = 0.01,
    ) -> int:
        """Return index of the vehicle that best overlaps *plate_bbox*.

        A very low IoU threshold is used because plates are typically
        much smaller than their parent vehicle bounding box.
        """
        best_idx = -1
        best_iou = 0.0
        px1, py1, px2, py2 = plate_bbox
        plate_area = max(px2 - px1, 0) * max(py2 - py1, 0)
        if plate_area == 0:
            return -1

        for idx, veh in enumerate(vehicle_detections):
            vx1, vy1, vx2, vy2 = veh["bbox"]
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
            if inter == 0:
                continue
            veh_area = max(vx2 - vx1, 0) * max(vy2 - vy1, 0)
            union = plate_area + veh_area - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold:
            return best_idx
        return -1

    # ------------------------------------------------------------------
    # Model / device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Return a valid device string, auto-detecting when necessary."""
        if device is not None:
            return device
        try:
            import torch

            if torch.cuda.is_available():
                dev = "cuda:0"
                logger.info("CUDA available – using %s", dev)
                return dev
        except ImportError:
            logger.debug("torch not installed; falling back to CPU.")
        return "cpu"

    @staticmethod
    def _resolve_model_path(
        user_path: Optional[str | Path],
        fallback_filename: Optional[str],
    ) -> Path:
        """Resolve a model path, searching common locations.

        Resolution order:
        1. Absolute *user_path* if it exists.
        2. ``models/detection/<user_path or fallback>`` relative to the
           project root.
        3. ``<user_path or fallback>`` as-is (let ultralytics handle it
           – it will download if the name is a known model like
           ``yolov8n.pt``).
        """
        src_dir = Path(__file__).resolve().parent          # src/detection/
        project_root = src_dir.parent.parent.parent        # anpr-pipeline/
        models_dir = project_root / "models" / "detection"

        filename = str(user_path) if user_path is not None else fallback_filename
        if filename is None:
            raise ValueError("No model path provided and no fallback defined.")

        candidates: list[Path] = []

        # Absolute path from caller
        if user_path is not None:
            candidates.append(Path(user_path))

        # Relative to models/detection/
        candidates.append(models_dir / filename)

        # Bare name – let ultralytics resolve / download
        candidates.append(Path(filename))

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        # Return the bare name so ultralytics can attempt a download.
        logger.warning(
            "Model file not found locally; ultralytics will attempt download: %s",
            filename,
        )
        return Path(filename)

    @staticmethod
    def _load_yolo(path: Path):
        """Load and return a YOLO model from *path*."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required.  Install it with: pip install ultralytics"
            ) from exc

        logger.info("Loading YOLO model from %s …", path)
        model = YOLO(str(path))
        return model
