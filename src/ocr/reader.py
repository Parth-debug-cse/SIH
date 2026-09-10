"""
EasyOCR-based license plate reader with preprocessing and multi-frame voting.

Uses EasyOCR for text recognition with CLAHE contrast enhancement and
majority voting across consecutive frames for stable plate readings.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateOCR:
    """License plate OCR engine with preprocessing and multi-frame voting.

    Args:
        lang: Language for EasyOCR (default ``'en'``).
        use_gpu: Whether to run EasyOCR on GPU.
        voting_window: Number of recent readings to consider for majority voting.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        voting_window: int = 5,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.voting_window = voting_window
        self._reader = None
        self._history: list[str] = []

    @property
    def reader(self):
        """Lazy-initialise EasyOCR on first use."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                [self.lang],
                gpu=self.use_gpu,
                verbose=False,
            )
            logger.info(
                "EasyOCR initialised (lang=%s, gpu=%s)", self.lang, self.use_gpu
            )
        return self._reader

    def preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to a plate crop.

        Steps:
            1. Convert to grayscale.
            2. Apply CLAHE contrast enhancement.
            3. Resize to standard height (32 px) preserving aspect ratio.
            4. Convert back to 3-channel BGR for OCR.
        """
        if plate_image is None or plate_image.size == 0:
            raise ValueError("plate_image is empty or None")

        img = plate_image.copy()

        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            gray = img
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        target_h = 32
        h, w = enhanced.shape[:2]
        if h <= 0:
            raise ValueError(f"Invalid image height: {h}")
        scale = target_h / h
        target_w = max(int(w * scale), 1)
        resized = cv2.resize(
            enhanced, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )

        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """Normalise raw OCR output to a plausible plate string."""
        text = text.strip().upper()
        text = re.sub(r"[^A-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_text(self, ocr_result: list) -> tuple[str, float]:
        """Pull the best text + confidence from an EasyOCR result list."""
        if not ocr_result:
            return "", 0.0

        best_text = ""
        best_conf = 0.0

        for item in ocr_result:
            if item is None or len(item) < 3:
                continue
            text = str(item[1])
            conf = float(item[2])
            if conf > best_conf:
                best_conf = conf
                best_text = text

        return self._clean_plate_text(best_text), best_conf

    def read_plate(self, plate_image: np.ndarray) -> dict:
        """Run OCR on a single plate crop with majority voting."""
        preprocessed = self.preprocess_plate(plate_image)

        try:
            ocr_result = self.reader.readtext(preprocessed)
            text, confidence = self._extract_text(ocr_result)
        except Exception:
            logger.exception("OCR inference failed")
            text, confidence = "", 0.0

        if text:
            self._history.append(text)
            if len(self._history) > self.voting_window:
                self._history = self._history[-self.voting_window:]
            text = self._majority_vote()

        return {"text": text, "confidence": confidence}

    def read_plate_batch(self, plate_images: list[np.ndarray]) -> list[dict]:
        """Run OCR on a batch of plate crops."""
        results: list[dict] = []
        for img in plate_images:
            try:
                results.append(self.read_plate(img))
            except Exception:
                logger.exception("Batch OCR failed for one image")
                results.append({"text": "", "confidence": 0.0})
        return results

    def clear_history(self) -> None:
        """Reset the multi-frame voting history."""
        self._history.clear()

    def _majority_vote(self) -> str:
        """Return the most frequent plate string from recent history."""
        if not self._history:
            return ""
        counter = Counter(self._history)
        most_common_text, most_common_count = counter.most_common(1)[0]
        if most_common_count >= 2:
            return most_common_text
        return self._history[-1]
