"""Second OCR backend: fast-plate-ocr (ONNX, plate-specific recognizer).

Diagnostic / A-B comparison use only — EasyOCR remains the primary path and
no detector, tracker, fusion, DB, gate, or threshold code is touched by this
module.  Tested against the RAW plate crop first (no aggressive
preprocessing: the model is designed for cropped plates).

Verified API (fast-plate-ocr, signatures inspected, smoke-tested):
    from fast_plate_ocr import LicensePlateRecognizer
    LicensePlateRecognizer(hub_ocr_model='cct-s-v2-global-model',
                           device='auto' | 'cuda' | 'cpu', ...)
    recognizer.run(rgb_uint8_array, return_confidence=True)
        -> [PlatePrediction(.plate, .char_probs, .has_confidence, ...)]
Note: RGB arrays are assumed (NOT BGR) — BGR crops are converted here.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "cct-s-v2-global-model"


def package_version() -> str:
    """Return the installed fast-plate-ocr version (pip-show equivalent)."""
    try:
        from importlib.metadata import version
        return version("fast-plate-ocr")
    except Exception:
        return "unknown"


def available_providers() -> list[str]:
    """ONNX Runtime execution providers actually present (never invented)."""
    try:
        import onnxruntime
        return list(onnxruntime.get_available_providers())
    except Exception:
        traceback.print_exc()
        return []


class FastPlateOCR:
    """Lazy fast-plate-ocr backend for A/B diagnostics.

    Args:
        model_name: hub model (default ``cct-s-v2-global-model``).
        device: ``'auto'`` (deduced from ONNX providers), ``'cuda'`` or
            ``'cpu'`` — the exact literals from the verified constructor.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._rec = None
        self._sig_printed = False
        self.version = package_version()

    @property
    def recognizer(self):
        """Lazily build LicensePlateRecognizer after signature verification."""
        if self._rec is None:
            import inspect

            from fast_plate_ocr import LicensePlateRecognizer

            if not self._sig_printed:
                print("LicensePlateRecognizer sig:",
                      inspect.signature(LicensePlateRecognizer.__init__))
                print("run sig:", inspect.signature(LicensePlateRecognizer.run))
                print("fast-plate-ocr version:", self.version)
                print("onnx providers:", available_providers())
                self._sig_printed = True
            self._rec = LicensePlateRecognizer(
                self.model_name, device=self.device
            )
            logger.info("FastPlateOCR ready (model=%s device=%s)",
                        self.model_name, self.device)
        return self._rec

    @staticmethod
    def to_rgb(crop_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR crop to uint8 RGB as the model requires."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("crop is empty or None")
        if crop_bgr.ndim == 2:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2RGB)
        elif crop_bgr.ndim == 3 and crop_bgr.shape[2] == 3:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected crop shape: {crop_bgr.shape}")
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    @staticmethod
    def parse_prediction(pred: Any) -> tuple[str, Optional[float]]:
        """Extract (text, confidence) from a PlatePrediction.

        Confidence = mean(char_probs) when the API provides it, else None
        (recorded, never fabricated).
        """
        text = str(getattr(pred, "plate", "") or "").strip()
        conf: Optional[float] = None
        try:
            if getattr(pred, "has_confidence", False):
                probs = getattr(pred, "char_probs", None)
                if probs is not None and len(probs) > 0:
                    conf = float(np.mean(np.asarray(probs, dtype=float)))
        except Exception:
            traceback.print_exc()
            conf = None
        return text, conf

    def read_crop(self, crop_bgr: np.ndarray) -> dict[str, Any]:
        """Run the model on a RAW BGR crop.  Raises on failure (loud)."""
        rgb = self.to_rgb(crop_bgr)
        preds = self.recognizer.run(rgb, return_confidence=True)
        if not preds:
            return {"text": "", "conf": None, "status": "empty"}
        text, conf = self.parse_prediction(preds[0])
        return {"text": text, "conf": conf, "status": "ok"}
