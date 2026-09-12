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


# Direct-recognition preprocessing variants (diagnostic experiment).
# A: raw crop | B: 3x INTER_CUBIC | C: gray+CLAHE+upscale |
# D: gray+OTSU+upscale | E: mild sharpen+upscale |
# F: perspective warp (requires a quadrilateral; skipped otherwise).
VARIANT_NAMES = ("A_raw", "B_3x", "C_clahe", "D_otsu", "E_sharpen", "F_persp")


def build_variant_images(
    crop: np.ndarray,
    quad: Optional[list] = None,
    upscale: float = 3.0,
) -> dict[str, Optional[np.ndarray]]:
    """Build greyscale OCR variants A–F from a plate crop (pure cv2, no model).

    Args:
        crop: BGR or greyscale plate crop.
        quad: optional 4 corner points for perspective correction (variant F).
            Without a quadrilateral, F is returned as None ("skipped").
    """
    if crop is None or crop.size == 0:
        raise ValueError("crop is empty or None")
    if crop.ndim == 3 and crop.shape[2] == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    elif crop.ndim == 2:
        gray = crop.copy()
    else:
        raise ValueError(f"Unexpected image shape: {crop.shape}")

    def _up(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        s = float(upscale) if max(h, w) * float(upscale) <= 1200 else 1200.0 / max(h, w)
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    variants: dict[str, Optional[np.ndarray]] = {
        "A_raw": gray,
        "B_3x": _up(gray),
    }
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variants["C_clahe"] = _up(clahe.apply(gray))
    # Threshold AFTER upscaling so the variant stays truly binary
    # (cubic interpolation would otherwise reintroduce grey fringes).
    _, otsu = cv2.threshold(_up(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["D_otsu"] = otsu
    base = _up(gray)
    blur = cv2.GaussianBlur(base, (0, 0), 1.0)
    variants["E_sharpen"] = cv2.addWeighted(base, 1.5, blur, -0.5, 0.0)

    if quad is None:
        variants["F_persp"] = None
    else:
        try:
            import numpy as _np

            pts = _np.asarray(quad, dtype=float).reshape(4, 2)
            (tl, tr, br, bl) = pts
            w_a = _np.linalg.norm(br - bl)
            w_b = _np.linalg.norm(tr - tl)
            h_a = _np.linalg.norm(tr - br)
            h_b = _np.linalg.norm(tl - bl)
            dst_w, dst_h = max(int(max(w_a, w_b)), 8), max(int(max(h_a, h_b)), 8)
            dst = _np.array(
                [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                dtype=float,
            )
            m = cv2.getPerspectiveTransform(pts.astype(_np.float32), dst.astype(_np.float32))
            warped = cv2.warpPerspective(gray, m, (dst_w, dst_h), flags=cv2.INTER_CUBIC)
            variants["F_persp"] = _up(warped)
        except Exception:
            variants["F_persp"] = None
    return variants


def looks_like_plate(text: str) -> bool:
    """Permissive check that *text* is a plausible Indian plate string.

    Whitespace is ignored and the input is upper-cased before matching, so
    fragments such as ``"KA 15 F 1502"`` still validate.  Region names/branding
    (``"BMTC"``, ``"BENGALURU"``) and bare digit runs (``"1502"``) do not.
    """
    if not text:
        return False
    return bool(PLATE_PATTERN.match(text.replace(" ", "").upper()))


# Real Indian state/UT RTO codes (whitelist).  Format-valid strings with a
# fake code — 'AA05440', 'GD75000' — passed the old permissive regex and must
# now be rejected.  'GD' is not a state code; neither is 'AA'.
INDIA_STATE_CODES = frozenset({
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HR', 'HP', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UK', 'UP',
    'WB',
})

PLATE_FORMAT_RE = re.compile(r'^([A-Z]{2})\d{1,2}[A-Z]{0,3}\d{3,4}$')


def is_valid_india_plate(text: str) -> tuple[bool, str]:
    """Two-stage Indian plate check: format first, then state-code whitelist.

    Returns ``(is_valid, reason)`` where reason is one of ``'ok'``,
    ``'empty'``, ``'format_rejected'``, ``'invalid_state_code'``.
    """
    if not text:
        return False, 'empty'
    s = text.replace(" ", "").upper()
    m = PLATE_FORMAT_RE.match(s)
    if not m:
        return False, 'format_rejected'
    if m.group(1) not in INDIA_STATE_CODES:
        return False, 'invalid_state_code'
    return True, 'ok'


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
        # Source-tagged observation log (diagnostic/integration stream).
        # Appended by log_observation(); NEVER read by the voting path, so
        # _aggregate/record_vote semantics are byte-identical with or
        # without logging.  Each entry: text/confidence/track_key/source/
        # frame/timestamp.
        self._observations: list[dict] = []
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

    def log_observation(
        self,
        text: str,
        confidence: float,
        track_key=None,
        source: str = "easyocr",
        frame: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Append one source-tagged observation to the track-level log.

        This stream is independent of the voting deques: logging never
        changes ``record_vote``/``_aggregate`` behavior.  Empty texts are
        ignored (nothing to fuse).
        """
        if not text:
            return
        key = tuple(track_key) if track_key is not None else _DEFAULT_KEY
        self._observations.append({
            "text": str(text),
            "confidence": float(confidence),
            "track_key": key,
            "source": str(source),
            "frame": frame,
            "timestamp": time.time() if timestamp is None else float(timestamp),
        })

    def observations_for(self, track_key=None, source: Optional[str] = None) -> list[dict]:
        """Return logged observations, optionally filtered by track/source."""
        key = tuple(track_key) if track_key is not None else None
        return [
            o for o in self._observations
            if (key is None or o["track_key"] == key)
            and (source is None or o["source"] == source)
        ]

    def track_fusion_report(
        self,
        track_key,
        source: str = "fastplate",
        min_confidence: float = 0.50,
    ) -> dict:
        """Fuse one track's logged observations with the EXISTING fusion.

        Runs :func:`fuse_track_observations` (position-aware,
        confidence-weighted — unchanged) over the track's observations for
        *source* and reports the verdict.  ``verdict`` mirrors the
        acceptance gates for reporting ONLY: nothing here writes sightings.
        Stability is never labeled correctness (see ``verified`` = False).
        """
        obs = self.observations_for(track_key, source=source)
        pairs = [(o["text"], o["confidence"]) for o in obs]
        fused = fuse_track_observations(pairs)
        plate = fused["canonical_plate"]
        # Validity (format + real state code) — the fusion math above is
        # unchanged; only the verdict gate is tightened.
        valid, vreason = is_valid_india_plate(plate) if plate else (False, 'empty')
        regex = valid
        if not plate:
            verdict = "no_observations"
        elif fused["aggregate_confidence"] < min_confidence:
            verdict = "fused_conf_below_gate"
        elif not regex:
            verdict = vreason  # 'format_rejected' or 'invalid_state_code'
        else:
            verdict = "would_accept_UNVERIFIED"
        return {
            "track_key": tuple(track_key),
            "source": source,
            "n_observations": len(obs),
            "predictions": [(o["text"], o["confidence"], o["frame"]) for o in obs],
            "fused": plate,
            "fused_confidence": fused["aggregate_confidence"],
            "method": fused["method"],
            "regex_pass": regex,
            "validity_reason": vreason,
            "verdict": verdict,
            "verified": False,
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

    def recognize_image(self, grey_img: np.ndarray) -> tuple[str, float]:
        """Direct whole-crop recognition, bypassing EasyOCR's text detector.

        Verified against installed EasyOCR (1.7.2,
        ``easyocr.py:353-375``): ``recognize(grey, horizontal_list=None)``
        treats the whole image as one region ``[[0, x_max, 0, y_max]]`` and
        returns ``[(box, text, conf)]``.  Only the verified ``allowlist``
        kwarg is passed (guarded by signature inspection for robustness
        across EasyOCR versions).  Raises on failure — the caller logs it.
        """
        import inspect
        import traceback

        try:
            params = inspect.signature(self.reader.recognize).parameters
        except (ValueError, TypeError):
            traceback.print_exc()
            params = {}
        kwargs: dict = {}
        if "allowlist" in params or not params:
            kwargs["allowlist"] = OCR_ALLOWLIST
        res = self.reader.recognize(grey_img, **kwargs)
        if not res:
            return "", 0.0
        _box, text, conf = res[0]
        return self._clean_plate_text(str(text)), float(conf)

    def recognize_variants(
        self,
        crop: np.ndarray,
        quad: Optional[list] = None,
    ) -> list[dict]:
        """Run direct recognition over variants A–F (diagnostic experiment).

        Touches no vote history — pure function of the crop.  Each trial
        carries ``variant/text/conf/normalized/regex_pass/status`` plus the
        variant ``image`` (used for top-5 Drive saves; excluded from CSVs).
        Failures are logged with tracebacks and recorded, never swallowed.
        """
        import traceback

        trials: list[dict] = []
        try:
            images = build_variant_images(crop, quad=quad)
        except Exception:
            traceback.print_exc()
            logger.exception("variant build failed")
            return trials
        for name in VARIANT_NAMES:
            img = images.get(name)
            if img is None:
                trials.append({"variant": name, "text": "", "conf": 0.0,
                               "normalized": "", "regex_pass": False,
                               "status": "skipped_no_quad", "image": None})
                continue
            try:
                text, conf = self.recognize_image(img)
                norm = self._normalize_confusions(text) if text else ""
                # Display validity (format + state code); internal ranking
                # math (_row_candidate_score) is unchanged.
                valid, _ = is_valid_india_plate(norm) if norm else (False, 'empty')
                trials.append({"variant": name, "text": text, "conf": conf,
                               "normalized": norm,
                               "regex_pass": valid,
                               "status": "ok", "image": img})
            except Exception:
                traceback.print_exc()
                logger.exception("direct recognize failed for variant %s", name)
                trials.append({"variant": name, "text": "", "conf": 0.0,
                               "normalized": "", "regex_pass": False,
                               "status": "failed", "image": img})
        return trials

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


class HybridPlateOCR:
    """
    Routes plate crops to fast-plate-ocr (primary) or EasyOCR (fallback) based on
    crop height, because fast-plate-ocr produces confident WRONG reads on crops
    under ~65px tall (verified: cam_1, 43-51px crops, hallucinated 'GD' state-code
    pattern at 0.77-0.84 conf). EasyOCR is strictly worse on adequately-sized crops
    (verified: cam_2, mean accept_conf 0.25 vs fast-plate-ocr's 0.5-0.99) but at
    least fails toward low-confidence/empty rather than confident-wrong on tiny crops.

    The numeric acceptance gate (0.50) stays in the caller (pipeline_runner):
    this class reports text/confidence/engine/validity and NEVER thresholds.
    """

    MIN_HEIGHT_FOR_FASTPLATE = 65  # px; below this, fast-plate-ocr is unreliable

    def __init__(
        self,
        use_gpu: bool = True,
        hub_ocr_model: str = 'cct-s-v2-global-model',
        easy_ocr: Optional["PlateOCR"] = None,
        fast_backend=None,
    ) -> None:
        # EasyOCR fallback: shared PlateOCR (vote history preserved) or own.
        self.easy: PlateOCR = easy_ocr if easy_ocr is not None else PlateOCR(
            use_gpu=use_gpu
        )
        self._fast = fast_backend  # FastPlateOCR or None (lazy)
        self.hub_ocr_model = hub_ocr_model
        self.use_gpu = use_gpu

    @property
    def fast(self):
        """Lazily build the fast-plate-ocr backend (model downloads once)."""
        if self._fast is None:
            from src.ocr.fastplate import FastPlateOCR

            # 'auto' is deduced from onnxruntime providers (verified
            # constructor literal); 'cpu' when no GPU requested.
            self._fast = FastPlateOCR(
                model_name=self.hub_ocr_model,
                device='auto' if self.use_gpu else 'cpu',
            )
        return self._fast

    def read_plate(
        self,
        crop_bgr: np.ndarray,
        crop_height_px: Optional[int] = None,
        track_key=None,
    ) -> dict:
        """Route one crop to fast-plate-ocr or the EasyOCR fallback.

        Returns dict with keys: text, confidence (None when the backend
        provides none — never fabricated), engine_used
        ('fastplate'/'easyocr'), regex_pass (via :func:`is_valid_india_plate`),
        validity_reason, reason.

        Engine failures are LOUD (traceback printed) but non-fatal: they
        return ``reason='engine_failed'`` with empty text so one bad crop
        cannot kill a video run — the traceback, not silence, is the record.
        """
        import traceback

        h = int(crop_height_px) if crop_height_px else int(crop_bgr.shape[0])
        if h >= self.MIN_HEIGHT_FOR_FASTPLATE:
            try:
                res = self.fast.read_crop(crop_bgr)
            except Exception:
                traceback.print_exc()
                logger.exception("HybridPlateOCR fastplate path failed")
                return {"text": "", "confidence": None, "engine_used": "fastplate",
                        "regex_pass": False, "validity_reason": "engine_failed",
                        "reason": "engine_failed"}
            valid, vreason = is_valid_india_plate(res["text"]) if res["text"] else (False, 'empty')
            return {"text": res["text"], "confidence": res["conf"],
                    "engine_used": "fastplate", "regex_pass": valid,
                    "validity_reason": vreason,
                    "reason": "ok" if res["text"] else "fastplate_empty"}
        try:
            res = self.easy.read_plate(crop_bgr, track_key=track_key)
        except Exception:
            traceback.print_exc()
            logger.exception("HybridPlateOCR easyocr path failed")
            return {"text": "", "confidence": 0.0, "engine_used": "easyocr",
                    "regex_pass": False, "validity_reason": "engine_failed",
                    "reason": "engine_failed"}
        if not res["text"]:
            return {"text": "", "confidence": 0.0, "engine_used": "easyocr",
                    "regex_pass": False, "validity_reason": "empty",
                    "reason": "below_resolution_floor"}
        valid, vreason = is_valid_india_plate(res["text"])
        return {"text": res["text"], "confidence": res["confidence"],
                "engine_used": "easyocr", "regex_pass": valid,
                "validity_reason": vreason, "reason": "easyocr_fallback"}