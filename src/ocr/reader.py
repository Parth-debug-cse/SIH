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

# Common EasyOCR character confusions for automotive glyphs.  Used only to
# legalise a near-miss string against the Indian plate grammar - the raw OCR
# read is never mutated unless it then matches the exact plate format.
PLATE_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "O": ("0",), "D": ("0",), "Q": ("0",),
    "I": ("1",), "L": ("1",), "B": ("8",),
    "S": ("5",), "Z": ("2",), "G": ("6",),
    "0": ("O",), "1": ("I",), "5": ("S",),
    "8": ("B",), "2": ("Z",), "6": ("G",),
}

# Restrict EasyOCR to glyphs that can appear on an Indian plate.  Without an
# allowlist EasyOCR spends probability mass on punctuation/symbols and the
# confidence of the registration line is dragged down.
OCR_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Digit that may legally appear in a LETTER segment (and its fix), and
# vice versa.  Derived from PLATE_CONFUSIONS — no invented mappings.
_DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G",
}
_LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1", "B": "8",
    "S": "5", "Z": "2", "G": "6",
}


def position_correct(text: str) -> str:
    """Position-aware confusion fix using the Indian plate layout.

    Layout: 2 state LETTERS + 1-2 district DIGITS + 0-3 series LETTERS +
    3-4 registration DIGITS.  A character that violates its segment's
    class (e.g. ``O`` inside the digit block) is mapped to the matching
    class; characters already in-class are never touched — unlike a
    global replace-all, this cannot corrupt correct characters.

    Every valid (district_len, series_len, number_len) split is tried in
    fixed order; the first split needing a mapped fix and yielding a
    string that matches the plate grammar is returned.  If no split
    works, the input is returned unchanged (never a hallucinated value).
    """
    s = re.sub(r"[^A-Z0-9]", "", text.upper())
    if looks_like_plate(s):
        return s
    n = len(s)
    best: str | None = None
    best_changes = 10 ** 9
    for dd in (1, 2):
        for se in (0, 1, 2, 3):
            for nu in (3, 4):
                if 2 + dd + se + nu != n:
                    continue
                parts = [s[0:2], s[2:2 + dd], s[2 + dd:2 + dd + se], s[2 + dd + se:]]
                fixed: list[str] = []
                changes = 0
                ok = True
                for seg_idx, seg in enumerate(parts):
                    want_digit = seg_idx in (1, 3)
                    for ch in seg:
                        if want_digit and ch.isalpha():
                            rep = _LETTER_TO_DIGIT.get(ch)
                            if rep is None:
                                ok = False
                                break
                            fixed.append(rep)
                            changes += 1
                        elif not want_digit and ch.isdigit():
                            rep = _DIGIT_TO_LETTER.get(ch)
                            if rep is None:
                                ok = False
                                break
                            fixed.append(rep)
                            changes += 1
                        else:
                            fixed.append(ch)
                    if not ok:
                        break
                if not ok or changes == 0 or changes >= best_changes:
                    continue
                candidate = "".join(fixed)
                if looks_like_plate(candidate):
                    best, best_changes = candidate, changes
    return best if best is not None else s


def fuse_track_observations(obs: list[tuple[str, float]]) -> dict:
    """Fuse one track's OCR observations into a canonical plate.

    Steps: normalize -> keep plate-shaped readings if any -> group by
    length (only structurally compatible strings vote together) ->
    per-position confidence-weighted vote.  Falls back to the
    whole-string (count, summed-confidence) winner when no position
    consensus is possible.  Returns ``canonical_plate``,
    ``aggregate_confidence``, ``observation_count`` and ``method``.
    """
    norm = [(re.sub(r"[^A-Z0-9]", "", t.upper()), float(c)) for t, c in obs if t]
    norm = [(t, c) for t, c in norm if t]
    if not norm:
        return {"canonical_plate": "", "aggregate_confidence": 0.0,
                "observation_count": 0, "method": "empty"}
    shaped = [(t, c) for t, c in norm if looks_like_plate(t)]
    pool = shaped if shaped else norm
    by_len: dict[int, list[tuple[str, float]]] = {}
    for t, c in pool:
        by_len.setdefault(len(t), []).append((t, c))
    # Largest group wins; tie-break by summed confidence (deterministic).
    group = max(by_len.values(), key=lambda g: (len(g), sum(c for _, c in g)))
    if len(group) == 1:
        t, c = group[0]
        return {"canonical_plate": t, "aggregate_confidence": c,
                "observation_count": 1, "method": "single"}
    L = len(group[0][0])
    winners: list[str] = []
    pos_scores: list[float] = []
    for i in range(L):
        weights: dict[str, float] = {}
        total = 0.0
        for t, c in group:
            weights[t[i]] = weights.get(t[i], 0.0) + c
            total += c
        ch, w = max(weights.items(), key=lambda kv: (kv[1], kv[0]))
        winners.append(ch)
        pos_scores.append(w / total if total > 0 else 0.0)
    canonical = "".join(winners)
    mean_conf = sum(c for _, c in group) / len(group)
    consistency = sum(pos_scores) / len(pos_scores)
    if pool is shaped and not looks_like_plate(canonical):
        # Position vote produced a non-plate: fall back to whole-string winner.
        best, confs = max(
            {t: [c for tt, c in group if tt == t] for t, _ in group}.items(),
            key=lambda kv: (len(kv[1]), sum(kv[1])),
        )
        return {"canonical_plate": best,
                "aggregate_confidence": sum(confs) / len(confs),
                "observation_count": len(group), "method": "whole_string_fallback"}
    return {"canonical_plate": canonical,
            "aggregate_confidence": round(mean_conf * consistency, 4),
            "observation_count": len(group), "method": "per_position_weighted"}


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

    def _normalize_confusions(self, text: str) -> str:
        """Attempt bounded character substitutions to legalise a near-miss.

        Each position is tried against ``PLATE_CONFUSIONS``; the first variant
        that matches the Indian plate grammar is returned.  If none matches,
        the original text is returned unchanged — no hallucinated values are
        added.
        """
        upper = text.upper()
        if looks_like_plate(upper):
            return upper
        for i, ch in enumerate(upper):
            for rep in PLATE_CONFUSIONS.get(ch, ()):
                candidate = upper[:i] + rep + upper[i + 1:]
                if looks_like_plate(candidate):
                    return candidate
        return upper

    @staticmethod
    def _row_candidate_score(text: str, conf: float) -> tuple[float, float]:
        """Return (score, conf) for a candidate plate string.

        The score ranks candidates; conf is the raw OCR value that gates the
        pipeline.  ``looks_like_plate`` gets a large bonus but conf is *never*
        artificially inflated — the anti-hallucination gate still applies.
        """
        n_digits = sum(ch.isdigit() for ch in text)
        n_alpha = sum(ch.isalpha() for ch in text)
        if looks_like_plate(text):
            return conf + 1.0, conf
        if n_alpha >= 1 and n_digits >= 1 and 5 <= len(text) <= 11:
            return conf + 0.30, conf
        if n_alpha >= 1 and 5 <= len(text) <= 11:
            return conf + 0.05, conf
        return conf, conf

    def _extract_text(self, ocr_result: list) -> tuple[str, float]:
        """Select the best plate string from one OCR pass.

        Multi-row assembly: bus/truck plates have the state/brand on a top
        row and the registration number below.  EasyOCR boxes both rows as
        independent detections.  We group boxes into rows by y-centre
        proximity, then assemble row-texts top-down and bottom-up to recover
        the full plate string before running the format/confusion pass.
        """
        if not ocr_result:
            return "", 0.0

        # 1. Collect boxes with geometric info
        boxes: list[dict] = []
        for item in ocr_result:
            if item is None or len(item) < 3:
                continue
            text = self._clean_plate_text(str(item[1]))
            if not text:
                continue
            pts = item[0]
            ys = [p[1] for p in pts]
            xs = [p[0] for p in pts]
            cy = sum(ys) / len(ys)
            h = max(ys) - min(ys) or 1.0
            boxes.append({
                "text": text,
                "conf": float(item[2]),
                "cy": cy,
                "cx": sum(xs) / len(xs),
                "h": h,
            })

        if not boxes:
            return "", 0.0

        # 2. Group into horizontal rows (within ~18 px on the 3×-upscaled image)
        boxes.sort(key=lambda b: b["cy"])
        rows: list[list[dict]] = []
        for b in boxes:
            if rows and abs(b["cy"] - rows[-1][-1]["cy"]) <= max(18.0, 0.5 * (rows[-1][-1]["h"] + b["h"])):
                rows[-1].append(b)
            else:
                rows.append([b])
        for r in rows:
            r.sort(key=lambda b: b["cx"])

        # 3. Build candidates: each row alone + assembled top-down / bottom-up
        candidates: list[tuple[float, float, str]] = []

        def _add(text: str, conf: float) -> None:
            norm = self._normalize_confusions(text)
            score, _ = self._row_candidate_score(norm, conf)
            candidates.append((score, conf, norm))
            logger.debug(
                "OCR candidate text='%s' conf=%.2f score=%.2f", norm, conf, score,
            )
            # Position-aware layout fix as an *additional* candidate: it can
            # only add a plate-shaped variant, never remove or mutate `norm`.
            pos = position_correct(text)
            if pos != norm:
                pscore, _ = self._row_candidate_score(pos, conf)
                candidates.append((pscore, conf, pos))
                logger.debug(
                    "OCR candidate text='%s' conf=%.2f score=%.2f", pos, conf, pscore,
                )

        for row in rows:
            row_text = "".join(b["text"] for b in row)
            row_conf = sum(b["conf"] for b in row) / len(row)
            _add(row_text, row_conf)

        if len(rows) >= 2:
            agg_conf = sum(b["conf"] for b in boxes) / len(boxes)
            full_tb = "".join("".join(b["text"] for b in r) for r in rows)
            full_bt = "".join("".join(b["text"] for b in r) for r in reversed(rows))
            _add(full_tb, agg_conf)
            _add(full_bt, agg_conf)

        if not candidates:
            return "", 0.0

        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][2], candidates[0][1]

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
            ocr_result = self.reader.readtext(
                preprocessed, allowlist=OCR_ALLOWLIST,
            )
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