"""
EasyOCR-based license plate reader with preprocessing and track-aware voting.

A plate reading is produced by EasyOCR + CLAHE preprocessing.  Multi-frame
majority voting is **isolated per logical vehicle identity** - keyed by the
caller-supplied ``track_key`` (normally ``(camera_id, track_id)``) - so that
readings from different vehicles are never pooled into one vote.  Histories
expire after a TTL and the set of tracked histories is bounded.

Confidence semantics
--------------------
``read_plate()`` returns:

* ``text``          - the voted (or most recent) plate string.
* ``confidence``    - the confidence of exactly that string: the mean
  EasyOCR confidence of the readings that produced the voted text.
* ``raw_text`` / ``raw_confidence`` - the current single reading, so
  callers can distinguish a raw read from an aggregated vote.
* ``voted``         - True when the returned text came from a multi-reading
  majority vote.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, deque
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_KEY = ("__default__",)

# Permissive Indian plate grammar: two-letter state code, 1-2 digit district
# number, optional 0-3 letter series, then the 3-4 digit unique number.
# Examples that match: KA01AB1234, MH12CD5678, DL01EF9012, TN09ABX1234.
PLATE_PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}$")


def looks_like_plate(text: str) -> bool:
    """Permissive check that *text* is a plausible Indian plate string.

    Whitespace is ignored and the input is upper-cased before matching, so
    fragments such as ``"KA 15 F 1502"`` still validate.  Region names/branding
    (``"BMTC"``, ``"BENGALURU"``) and bare digit runs (``"1502"``) do not.
    """
    if not text:
        return False
    return bool(PLATE_PATTERN.match(text.replace(" ", "").upper()))


class PlateOCR:
    """License plate OCR engine with preprocessing and multi-frame voting.

    Args:
        lang: Language for EasyOCR (default ``'en'``).
        use_gpu: Whether to run EasyOCR on GPU.
        voting_window: Number of recent readings per track to consider.
        history_ttl_seconds: Max age (s) of a track history before expiry.
        max_track_histories: Bound on the number of live track histories.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        voting_window: int = 5,
        history_ttl_seconds: float = 30.0,
        max_track_histories: int = 256,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.voting_window = voting_window
        self.history_ttl_seconds = history_ttl_seconds
        self.max_track_histories = max_track_histories
        self._reader = None
        # track_key -> deque of (text, confidence, timestamp)
        self._histories: dict[tuple, deque] = {}
        # track_key -> last-access time (for LRU eviction)
        self._last_access: dict[tuple, float] = {}
        # Observability for the preprocessing stage (fix: was defined but
        # unverifiable in the real path).
        self.preprocess_count: int = 0
        self.last_preprocessed: Optional[np.ndarray] = None

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

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_plate(
        self,
        plate_image: np.ndarray,
        upscale_factor: float = 3.0,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid_size: int = 8,
        unsharp_sigma: float = 1.5,
        unsharp_amount: float = 1.0,
    ) -> np.ndarray:
        """Preprocess a plate crop before EasyOCR.

        This is the single isolated, tunable preprocessing stage applied only
        to the plate crop.  Steps (in order):

            1. Convert to grayscale.
            2. Apply CLAHE contrast enhancement.
            3. Upscale ``upscale_factor``-fold (3-4x) with ``cv2.INTER_CUBIC``
               so the plate glyphs render large enough for EasyOCR.
            4. Unsharp-mask sharpen to restore edge contrast.

        Returns a 3-channel BGR image ready for OCR.
        """
        if plate_image is None or plate_image.size == 0:
            raise ValueError("plate_image is empty or None")

        img = plate_image.copy()

        # 1. Grayscale.
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            gray = img
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # 2. CLAHE contrast enhancement.
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )
        enhanced = clahe.apply(gray)

        # 3. 3-4x upscale with cubic interpolation.
        scale = max(float(upscale_factor), 1.0)
        h, w = enhanced.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {enhanced.shape}")
        target_w = max(int(w * scale), 1)
        target_h = max(int(h * scale), 1)
        upscaled = cv2.resize(
            enhanced,
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC,
        )

        # 4. Unsharp-mask sharpen.
        blurred = cv2.GaussianBlur(upscaled, (0, 0), unsharp_sigma)
        sharpened = cv2.addWeighted(
            upscaled, 1.0 + unsharp_amount, blurred, -unsharp_amount, 0.0
        )

        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        # Observability: log every call so the preprocess stage is provably
        # exercised in the live detection -> OCR path, and stash the output
        # (the caller may save one crop per run for manual inspection).
        logger.debug(
            "[PREPROCESS] plate crop preprocessed (%dx%d -> %dx%d)",
            w, h, target_w, target_h,
        )
        self.preprocess_count += 1
        self.last_preprocessed = result

        return result

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

    # ------------------------------------------------------------------
    # Track-aware voting
    # ------------------------------------------------------------------

    def _expire_histories(self, now: float) -> None:
        """Drop histories that have not been touched within the TTL."""
        expired = [
            key for key, last in self._last_access.items()
            if now - last > self.history_ttl_seconds
        ]
        for key in expired:
            self._histories.pop(key, None)
            self._last_access.pop(key, None)

        # Bound the number of live histories (stale tracks must not accumulate
        # forever) by evicting the least-recently-used entries.
        while len(self._histories) > self.max_track_histories:
            oldest = min(self._last_access, key=self._last_access.get)
            self._histories.pop(oldest, None)
            self._last_access.pop(oldest, None)

    def _history_for(self, track_key, now: float) -> deque:
        key = tuple(track_key) if track_key is not None else _DEFAULT_KEY
        hist = self._histories.get(key)
        if hist is None:
            hist = deque(maxlen=self.voting_window)
            self._histories[key] = hist
        self._last_access[key] = now
        return hist

    def record_vote(
        self,
        text: str,
        confidence: float,
        track_key=None,
        now: Optional[float] = None,
    ) -> dict:
        """Record one reading for a track and return the current vote.

        This is the single place where history is updated, so unit tests can
        exercise voting without running EasyOCR.
        """
        now = time.time() if now is None else now
        if not text:
            return {"text": "", "confidence": 0.0, "raw_text": "", "raw_confidence": 0.0,
                    "voted_count": 0, "voted": False}
        hist = self._history_for(track_key, now)
        # Expire AFTER the current key is (re)registered so the bound is
        # respected exactly and the just-touched history is never evicted.
        self._expire_histories(now)
        hist.append((text, float(confidence)))

        voted_text, voted_conf, count = self._aggregate(hist)
        return {
            "text": voted_text,
            "confidence": voted_conf,
            "raw_text": text,
            "raw_confidence": float(confidence),
            "voted_count": count,
            "voted": count >= 2,
        }

    @staticmethod
    def _aggregate(hist: deque) -> tuple[str, float, int]:
        """Aggregate a track's readings into (text, confidence, vote size).

        The text is the most frequent reading; confidence is the mean
        confidence across the readings that produced that text.  With no
        majority, the most recent reading (and its confidence) is returned.
        """
        if not hist:
            return "", 0.0, 0

        confs_by_text: dict[str, list[float]] = {}
        for text, conf in hist:
            confs_by_text.setdefault(text, []).append(conf)

        most_common_text, confs = max(
            confs_by_text.items(), key=lambda kv: (len(kv[1]), sum(kv[1]))
        )
        count = len(confs)
        if count >= 2:
            return most_common_text, float(sum(confs) / len(confs)), count
        # No majority yet - return the most recent reading.
        last_text, last_conf = hist[-1]
        return last_text, float(last_conf), 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_plate(self, plate_image: np.ndarray, track_key=None) -> dict:
        """Run OCR on a single plate crop with track-aware majority voting.

        Args:
            plate_image: BGR crop containing the plate.
            track_key: logical vehicle identity (e.g. ``(camera_id, track_id)``).
                Defaults to a shared key (legacy single-stream behaviour).

        Returns:
            Dict with ``text``/``confidence`` (matching each other) plus
            ``raw_text``/``raw_confidence``/``voted``/``voted_count``.
        """
        preprocessed = self.preprocess_plate(plate_image)

        try:
            ocr_result = self.reader.readtext(preprocessed)
            text, confidence = self._extract_text(ocr_result)
        except Exception:
            logger.exception("OCR inference failed")
            text, confidence = "", 0.0

        if not text:
            return {"text": "", "confidence": 0.0, "raw_text": "", "raw_confidence": 0.0,
                    "voted_count": 0, "voted": False}

        return self.record_vote(text, confidence, track_key=track_key)

    def read_plate_batch(self, plate_images: list[np.ndarray], track_key=None) -> list[dict]:
        """Run OCR on a batch of plate crops (shared track identity)."""
        results: list[dict] = []
        for img in plate_images:
            try:
                results.append(self.read_plate(img, track_key=track_key))
            except Exception:
                logger.exception("Batch OCR failed for one image")
                results.append({"text": "", "confidence": 0.0, "raw_text": "",
                                "raw_confidence": 0.0, "voted_count": 0, "voted": False})
        return results

    def clear_history(self, track_key=None) -> None:
        """Reset voting history.

        With *track_key*: clear only that track's history.  Without it: clear
        all track histories.
        """
        if track_key is None:
            self._histories.clear()
            self._last_access.clear()
            return
        key = tuple(track_key)
        self._histories.pop(key, None)
        self._last_access.pop(key, None)