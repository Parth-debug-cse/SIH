"""YOLO11 inference engine with Apple Neural Engine (ANE) fast path.

Priority: CoreML ``.mlpackage`` (ANE) > torch MPS > torch CPU.
All blocking inference runs in ``asyncio.to_thread`` so the event loop —
and the decoder producer — never stalls.

Export (one-time, on Apple Silicon)::
    from ultralytics import YOLO
    m = YOLO("yolo11n.pt")
    m.export(format="coreml", nms=True, imgsz=640)  # -> yolo11n.mlpackage

Then point this engine at the ``.mlpackage``. On non-Apple hardware, or when
coremltools/ANE is unavailable, the engine falls back to MPS/CPU torch with
the original ``.pt`` — no exception to the caller for backend selection.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.domain.m5_schemas import Detection, InferenceBackend, M5DeviceInfo

logger = logging.getLogger(__name__)

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}


def probe_m5_device() -> M5DeviceInfo:
    """Detect the best inference backend. Never raises."""
    plat, arch = platform.system(), platform.machine()
    mps, mlx = False, False
    try:
        import torch

        mps = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        logger.debug("torch MPS probe failed", exc_info=True)
    try:
        import mlx.core  # noqa: F401

        mlx = True
    except Exception:
        mlx = False
    if plat == "Darwin" and arch == "arm64" and _coreml_importable():
        backend = InferenceBackend.COREML_ANE
    elif mps:
        backend = InferenceBackend.MPS
    else:
        backend = InferenceBackend.CPU
    return M5DeviceInfo(
        platform=plat, arch=arch, mps_available=mps,
        mlx_available=mlx, backend=backend,
    )


def _coreml_importable() -> bool:
    try:
        import coremltools  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class YOLO11Config:
    weights_pt: str = "yolo11n.pt"  # torch fallback / export source
    coreml_package: str = ""  # e.g. "models/detection/yolo11n.mlpackage"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    device: str = "auto"  # auto | mps | cpu (torch fallback only)
    export_if_missing: bool = False  # set True to auto-export on Apple Silicon


class YOLO11ANEEngine:
    """Thin async wrapper around Ultralytics YOLO11 with ANE preference."""

    def __init__(self, config: YOLO11Config | None = None) -> None:
        self.config = config or YOLO11Config()
        self.device_info = probe_m5_device()
        self.backend = self.device_info.backend
        self._model: Any = None
        self._lock: asyncio.Lock | None = None  # created lazily (py3.9-safe)

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    async def load(self) -> InferenceBackend:
        """Load the best available model. Returns the active backend."""
        async with self._get_lock():
            if self._model is not None:
                return self.backend
            return await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> InferenceBackend:
        from ultralytics import YOLO

        # 1. ANE path: existing .mlpackage on Apple Silicon.
        pkg = self.config.coreml_package
        if (
            self.device_info.backend == InferenceBackend.COREML_ANE
            and pkg
            and Path(pkg).exists()
        ):
            try:
                self._model = YOLO(pkg)  # Ultralytics runs .mlpackage on ANE
                self.backend = InferenceBackend.COREML_ANE
                logger.info("YOLO11 loaded on ANE: %s", pkg)
                return self.backend
            except Exception:
                logger.exception("CoreML package load failed; falling back")

        # 2. Optional one-time export (Apple Silicon only, explicit opt-in).
        if (
            self.config.export_if_missing
            and self.device_info.backend == InferenceBackend.COREML_ANE
            and pkg and not Path(pkg).exists()
        ):
            try:
                src = YOLO(self.config.weights_pt)
                exported = src.export(
                    format="coreml", nms=True, imgsz=self.config.imgsz
                )
                logger.info("YOLO11 exported to CoreML: %s", exported)
                self._model = YOLO(str(exported))
                self.backend = InferenceBackend.COREML_ANE
                return self.backend
            except Exception:
                logger.exception("CoreML export failed; falling back to torch")

        # 3. Torch fallback: MPS when available, else CPU.
        device = self.config.device
        if device == "auto":
            device = "mps" if self.device_info.mps_available else "cpu"
        try:
            self._model = YOLO(self.config.weights_pt)
            # Ultralytics predict(device=...) handles mps/cpu placement.
            self._model.to(device)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("YOLO .to(device) failed; using default device")
            self._model = YOLO(self.config.weights_pt)
            device = "cpu"
        self.backend = (
            InferenceBackend.MPS if device == "mps" else InferenceBackend.CPU
        )
        logger.info("YOLO11 loaded on torch backend=%s", self.backend)
        return self.backend

    # ------------------------------------------------------------------
    async def predict(
        self, frame_bgr: np.ndarray, conf: float | None = None
    ) -> tuple[list[Detection], float]:
        """Run inference without blocking the loop. Returns (dets, ms)."""
        if self._model is None:
            await self.load()
        return await asyncio.to_thread(
            self._predict_blocking, frame_bgr, conf or self.config.conf
        )

    def _predict_blocking(
        self, frame_bgr: np.ndarray, conf: float
    ) -> tuple[list[Detection], float]:
        import cv2  # Ultralytics expects RGB; convert once, on the thread.

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        device = (
            "mps"
            if self.backend == InferenceBackend.MPS
            else ("cpu" if self.backend == InferenceBackend.CPU else None)
        )
        kwargs: dict[str, Any] = {
            "imgsz": self.config.imgsz,
            "conf": conf,
            "iou": self.config.iou,
            "verbose": False,
        }
        if device is not None:
            kwargs["device"] = device
        results = self._model.predict(rgb, **kwargs)
        dets: list[Detection] = []
        for r in results:
            names = r.names
            for b in (r.boxes or []):
                cls_id = int(b.cls[0])
                name = str(names.get(cls_id, cls_id))
                if name not in VEHICLE_CLASS_NAMES:
                    continue  # vehicle-only for Phase 1
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                dets.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=float(b.conf[0]),
                        class_id=cls_id,
                        class_name=name,
                    )
                )
        ms = (time.perf_counter() - t0) * 1000.0
        return dets, ms
