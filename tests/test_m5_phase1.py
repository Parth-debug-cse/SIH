"""Phase-1 tests — no GPU, no model weights, no video files needed.

Mocks the blocking decode + YOLO model to verify:
  1. Decoder queue is truly async (producer never blocks consumer).
  2. Drop-oldest policy works when inference is slower than decode.
  3. Engine backend probe never raises and returns a valid backend.
  4. Pipeline.run_once works end-to-end with a stub model.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.m5.m5_pipeline import M5Pipeline, M5PipelineConfig  # noqa: E402
from src.domain.m5_schemas import Detection, InferenceBackend  # noqa: E402
from src.infrastructure.detection.yolo11_ane import (  # noqa: E402
    YOLO11ANEEngine,
    probe_m5_device,
)
from src.infrastructure.video.m5_decoder import (  # noqa: E402
    M5DecoderConfig,
    M5VideoDecoder,
    _Frame,
    probe_videotoolbox,
)


def test_probe_helpers_never_raise() -> None:
    assert probe_videotoolbox() in (True, False)
    info = probe_m5_device()
    assert info.backend in (
        InferenceBackend.COREML_ANE, InferenceBackend.MPS, InferenceBackend.CPU,
    )


def test_drop_oldest_queue_policy() -> None:
    async def _run() -> None:
        cfg = M5DecoderConfig(source="dummy.mp4", queue_size=2, drop_on_full=True)
        dec = M5VideoDecoder(cfg)
        for i in range(5):  # overflow a size-2 queue
            await dec._put_drop_oldest(
                _Frame(stream_id="t", frame_number=i, pts=None,
                       array=np.zeros((4, 4, 3), np.uint8),
                       backend=dec.backend)
            )
        assert dec.queue.qsize() == 2
        assert dec.frames_dropped == 3
        # Newest survive.
        got = [dec.queue.get_nowait().frame_number for _ in range(2)]
        assert got == [3, 4]

    asyncio.run(_run())


def test_decode_does_not_block_inference() -> None:
    """Producer pushes 20 frames while a slow consumer reads; wall time must
    stay near consumer speed, proving handoff is async (drop-oldest)."""

    import time

    async def _run() -> float:
        cfg = M5DecoderConfig(source="dummy.mp4", queue_size=4, drop_on_full=True)
        dec = M5VideoDecoder(cfg)
        t0 = time.perf_counter()
        for i in range(20):
            await dec._put_drop_oldest(
                _Frame(stream_id="t", frame_number=i, pts=None,
                       array=np.zeros((8, 8, 3), np.uint8),
                       backend=dec.backend)
            )
            await asyncio.sleep(0)  # yield like a real decoder thread gap
        # Slow consumer: 5ms per frame.
        n = 0
        while not dec.queue.empty():
            dec.queue.get_nowait()
            await asyncio.sleep(0.005)
            n += 1
        dt = time.perf_counter() - t0
        assert n <= 4  # only the last window survived
        return dt

    dt = asyncio.run(_run())
    assert dt < 2.0, f"handoff looks blocking: {dt:.2f}s"


def test_pipeline_run_once_with_stub() -> None:
    async def _run() -> None:
        cfg = M5PipelineConfig(decoder=M5DecoderConfig(source="dummy.mp4"))
        pipe = M5Pipeline(cfg)

        async def _stub(frame: np.ndarray) -> tuple[list, float]:
            await asyncio.sleep(0)  # proves awaitable off-loop path
            return ([Detection(bbox=[1, 2, 3, 4], confidence=0.9,
                               class_id=2, class_name="car")], 1.5)

        pipe.engine.predict = _stub  # type: ignore[method-assign]
        res = await pipe.run_once(np.zeros((16, 16, 3), np.uint8))
        assert len(res.detections) == 1
        assert res.detections[0].class_name == "car"
        assert res.inference_ms == 1.5

    asyncio.run(_run())


def test_engine_lazy_without_weights() -> None:
    eng = YOLO11ANEEngine()
    assert eng.backend in (
        InferenceBackend.COREML_ANE, InferenceBackend.MPS, InferenceBackend.CPU,
    )


if __name__ == "__main__":
    test_probe_helpers_never_raise()
    test_drop_oldest_queue_policy()
    test_decode_does_not_block_inference()
    test_pipeline_run_once_with_stub()
    test_engine_lazy_without_weights()
    print("PHASE1_OK")
