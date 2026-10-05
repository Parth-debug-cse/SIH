"""Phase-1 async pipeline: VideoToolbox decode -> YOLO11-ANE detect.

Decode and inference run as independent asyncio tasks joined only by the
bounded decoder queue — a slow inference step applies backpressure (file) or
drops oldest (live) but NEVER blocks the decoder thread, and a slow decoder
never blocks inference (consumer just awaits).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from src.domain.m5_schemas import FrameResult
from src.infrastructure.detection.yolo11_ane import YOLO11ANEEngine, YOLO11Config
from src.infrastructure.video.m5_decoder import M5DecoderConfig, M5VideoDecoder

logger = logging.getLogger(__name__)


@dataclass
class M5PipelineConfig:
    decoder: M5DecoderConfig
    detector: YOLO11Config | None = None
    log_every: int = 100


class M5Pipeline:
    def __init__(self, config: M5PipelineConfig) -> None:
        self.config = config
        self.decoder = M5VideoDecoder(config.decoder)
        self.engine = YOLO11ANEEngine(config.detector or YOLO11Config())
        self.frames_processed = 0
        self.detections_total = 0

    async def run(self) -> dict[str, float | str | int]:
        """Consume the whole stream. Returns summary stats."""
        decode_backend = await self.decoder.start()
        infer_backend = await self.engine.load()
        logger.info("M5 pipeline up: decode=%s infer=%s",
                    decode_backend, infer_backend)
        try:
            async for frame in self.decoder.frames():
                dets, ms = await self.engine.predict(frame.array)
                self.frames_processed += 1
                self.detections_total += len(dets)
                if self.frames_processed % self.config.log_every == 0:
                    logger.info(
                        "M5 progress: frame=%d dets=%d infer_ms=%.1f",
                        frame.frame_number, len(dets), ms,
                    )
        finally:
            await self.decoder.stop()
        return {
            "decode_backend": str(decode_backend),
            "infer_backend": str(infer_backend),
            "frames_processed": self.frames_processed,
            "detections_total": self.detections_total,
            "frames_dropped": self.decoder.frames_dropped,
        }

    async def run_once(self, frame_bgr: np.ndarray, frame_number: int = 0) -> FrameResult:
        """Single-frame path (tests / API) — no decoder involved."""
        dets, ms = await self.engine.predict(frame_bgr)
        return FrameResult(
            stream_id=self.config.decoder.stream_id,
            frame_number=frame_number,
            backend=self.engine.backend,
            detections=dets,
            inference_ms=ms,
        )
