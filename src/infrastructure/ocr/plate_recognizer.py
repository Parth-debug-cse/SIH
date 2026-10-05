"""Plate recognition engine: MLX LPRNet primary, Vision fallback.

Backend resolution order (first usable wins):
  1. ``mlx_lprnet`` — LPRNet graph on MLX (needs ``mlx`` + ``.npz`` weights).
  2. ``vision`` — ``VNRecognizeTextRequest`` via pyobjc (macOS only).
  3. ``stub`` — deterministic canned output, explicit opt-in for tests ONLY.

If none is usable the engine raises :class:`OCREngineUnavailable` instead
of inventing text. EasyOCR is not referenced anywhere in this module.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

import numpy as np

from src.domain.ocr_schemas import OCRResult
from src.infrastructure.ocr.lprnet_mlx import (
    LPRNetMLX,
    ctc_greedy_decode,
    preprocess_for_lprnet,
)
from src.infrastructure.ocr.plate_preprocessor import PlatePreprocessor
from src.infrastructure.ocr.vision_backend import (
    recognize_with_vision,
    vision_available,
)

logger = logging.getLogger(__name__)

_ALNUM = re.compile(r"[^0-9A-Z]")


class OCREngineUnavailable(RuntimeError):
    """No recognition backend is usable on this machine."""


@dataclass
class RecognizerConfig:
    backend: str = "auto"  # auto | mlx | vision | stub (stub = tests only)
    mlx_weights: str = "models/ocr/lprnet_indian.npz"
    stub_text: str = "MH12AB1234"
    stub_confidence: float = 0.99


def normalize_plate_text(raw: str) -> str:
    """Uppercase alphanumeric only — shared by every backend."""
    return _ALNUM.sub("", (raw or "").upper())


class PlateRecognizer:
    def __init__(
        self,
        config: RecognizerConfig | None = None,
        preprocessor: PlatePreprocessor | None = None,
    ) -> None:
        self.config = config or RecognizerConfig()
        self.preprocessor = preprocessor or PlatePreprocessor()
        self.active_backend: str = "unresolved"
        self._lprnet: LPRNetMLX | None = None

    # ------------------------------------------------------------------
    def _resolve(self) -> str:
        if self.active_backend != "unresolved":
            return self.active_backend
        want = self.config.backend
        if want in ("auto", "mlx") and self._try_init_mlx():
            self.active_backend = "mlx_lprnet"
            return self.active_backend
        if want in ("auto", "vision") and vision_available():
            self.active_backend = "vision"
            return self.active_backend
        if want == "stub":
            # Explicit opt-in only (unit tests). Never a silent fallback —
            # silent fake text would be worse than an honest error.
            self.active_backend = "stub"
            return self.active_backend
        raise OCREngineUnavailable(
            "No OCR backend usable: MLX weights/model missing and "
            "pyobjc Vision unavailable. Provide models/ocr/lprnet_indian.npz "
            "or install pyobjc-framework-Vision on macOS."
        )

    def _try_init_mlx(self) -> bool:
        from pathlib import Path

        if not Path(self.config.mlx_weights).exists():
            return False
        try:
            net = LPRNetMLX()
            net.load_weights(self.config.mlx_weights)
            net.eval()
            self._lprnet = net
            logger.info("PlateRecognizer: MLX LPRNet ready (%s)", self.config.mlx_weights)
            return True
        except Exception:
            logger.warning("MLX LPRNet init failed; trying next backend", exc_info=True)
            self._lprnet = None
            return False

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._resolve()

    def recognize(
        self,
        crop_bgr: np.ndarray,
        bbox: list[float] | None = None,
        quad: np.ndarray | None = None,
    ) -> OCRResult:
        backend = self._resolve()
        processed, info = self.preprocessor.process(np.asarray(crop_bgr), quad=quad)
        box = [float(v) for v in (bbox or [0, 0, crop_bgr.shape[1], crop_bgr.shape[0]])]

        if backend == "mlx_lprnet":
            return self._recognize_mlx(processed, box, info)
        if backend == "vision":
            return self._recognize_vision(processed, box, info)
        return self._recognize_stub(box, info)

    async def arecognize(
        self,
        crop_bgr: np.ndarray,
        bbox: list[float] | None = None,
        quad: np.ndarray | None = None,
    ) -> OCRResult:
        """Async entry — blocking inference stays off the event loop."""
        return await asyncio.to_thread(self.recognize, np.asarray(crop_bgr), bbox, quad)

    # ------------------------------------------------------------------
    def _recognize_mlx(self, processed: np.ndarray, box: list[float], info: dict) -> OCRResult:
        assert self._lprnet is not None
        t0 = time.perf_counter()
        x = preprocess_for_lprnet(processed)
        logprobs = self._lprnet(x)
        mlx_ms = (time.perf_counter() - t0) * 1000.0
        text, conf = ctc_greedy_decode(logprobs)
        text = normalize_plate_text(text)
        return OCRResult(
            bbox=box, text=text, confidence=float(conf), backend="mlx_lprnet",
            inference_ms=mlx_ms, mlx_inference_ms=mlx_ms,
            preprocess_backend=info["upscale_backend"],
            deskew_applied=info["deskew_applied"],
        )

    def _recognize_vision(self, processed: np.ndarray, box: list[float], info: dict) -> OCRResult:
        t0 = time.perf_counter()
        try:
            raw, conf = recognize_with_vision(processed)
        except Exception as exc:
            raise OCREngineUnavailable(f"Vision backend failed at runtime: {exc}") from exc
        ms = (time.perf_counter() - t0) * 1000.0
        return OCRResult(
            bbox=box, text=normalize_plate_text(raw), confidence=float(conf),
            backend="vision", inference_ms=ms, mlx_inference_ms=None,
            preprocess_backend=info["upscale_backend"],
            deskew_applied=info["deskew_applied"],
        )

    def _recognize_stub(self, box: list[float], info: dict) -> OCRResult:
        logger.warning("PlateRecognizer using TEST stub backend — not for production")
        return OCRResult(
            bbox=box, text=self.config.stub_text,
            confidence=float(self.config.stub_confidence), backend="stub",
            inference_ms=0.0, mlx_inference_ms=None,
            preprocess_backend=info["upscale_backend"],
            deskew_applied=info["deskew_applied"],
        )
