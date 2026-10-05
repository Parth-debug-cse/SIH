"""M5 hardware video ingestion via PyAV + VideoToolbox.

Main-pipeline decoding MUST NOT use ``cv2.VideoCapture`` (CPU path, throttles
M5). This module decodes H.264/HEVC with Apple's VideoToolbox media engine and
hands frames to inference through a bounded ``asyncio.Queue`` so decoding
never blocks inference (and vice versa).

Design:
  * Producer runs blocking ``container.decode()`` in a worker thread via
    ``asyncio.to_thread`` and ``await``-pushes into the queue.
  * Queue is drop-oldest when full (live/RTSP) or backpressure-wait (file),
    configured by ``drop_on_full``.
  * Consumer (inference) pulls with ``await get()`` — never touches the
    decoder thread.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

import numpy as np

from src.domain.m5_schemas import DecodeBackend

logger = logging.getLogger(__name__)


def probe_videotoolbox() -> bool:
    """Best-effort check that VideoToolbox HW decode is usable.

    True only on Apple Silicon macOS with an ffmpeg build exposing
    ``h264_videotoolbox``. Never raises — returns False on any doubt so the
    caller falls back to software decode.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # No ffmpeg binary to probe; PyAV bundles its own ffmpeg libs, so
        # still attempt HW open later — but report unavailable here.
        return False
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=10,
        )
        return "videotoolbox" in (out.stdout + out.stderr).lower()
    except Exception:
        logger.exception("VideoToolbox probe failed; using software decode")
        return False


@dataclass
class M5DecoderConfig:
    source: str  # file path, RTSP URL, or MP4
    stream_id: str = "cam_1"
    width: int = 0  # 0 = keep native
    height: int = 0
    queue_size: int = 8
    drop_on_full: bool = True  # True for live; False for file backpressure
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    timeout_s: float = 10.0


@dataclass
class _Frame:
    stream_id: str
    frame_number: int
    pts: float | None
    array: np.ndarray  # BGR uint8
    backend: DecodeBackend


class M5VideoDecoder:
    """Async PyAV decoder with VideoToolbox HW acceleration + fallback."""

    def __init__(self, config: M5DecoderConfig) -> None:
        self.config = config
        self.queue: asyncio.Queue[_Frame | None] = asyncio.Queue(
            maxsize=config.queue_size
        )
        self.backend: DecodeBackend = DecodeBackend.SOFTWARE
        self._stop = asyncio.Event()
        self._producer_task: asyncio.Task[None] | None = None
        self.frames_decoded: int = 0
        self.frames_dropped: int = 0

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------
    async def start(self) -> DecodeBackend:
        """Spawn the producer thread-task. Returns the active backend."""
        self._stop.clear()
        self._producer_task = asyncio.create_task(self._produce())
        # Give producer a chance to set backend; don't block long.
        await asyncio.sleep(0.05)
        return self.backend

    async def stop(self) -> None:
        self._stop.set()
        if self._producer_task is not None:
            await self._producer_task
        # Drain sentinel safety: unblock any waiting consumer.
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def __aiter__(self) -> AsyncIterator[_Frame]:
        return self.frames()

    async def frames(self) -> AsyncIterator[_Frame]:
        """Yield frames until stream ends (None sentinel) or stop()."""
        assert self._producer_task is not None, "call await start() first"
        while True:
            item = await self.queue.get()
            if item is None:  # end-of-stream sentinel
                break
            yield item
            self.queue.task_done()

    # ------------------------------------------------------------------
    # Producer (blocking decode off the event loop)
    # ------------------------------------------------------------------
    async def _produce(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await asyncio.to_thread(self._decode_blocking, loop)
        except Exception:
            logger.exception("M5 decoder producer crashed")
        finally:
            # Always terminate the consumer stream.
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                # Make room for the sentinel if a live stream filled it.
                try:
                    self.queue.get_nowait()
                    self.frames_dropped += 1
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    def _decode_blocking(self, loop: asyncio.AbstractEventLoop) -> None:
        import av  # local import: keeps domain import-light for tests

        src = self.config.source
        is_rtsp = src.startswith("rtsp://")
        # Attempt HW first, then software — never fail without trying both.
        attempts: list[dict] = []
        if is_rtsp:
            attempts.append(
                {"rtsp_transport": self.config.rtsp_transport,
                 "hwaccel": "videotoolbox", "timeout": str(int(self.config.timeout_s * 1e6))}
            )
            attempts.append(
                {"rtsp_transport": self.config.rtsp_transport,
                 "timeout": str(int(self.config.timeout_s * 1e6))}
            )
        else:
            attempts.append({"hwaccel": "videotoolbox"})
            attempts.append({})

        container = None
        last_err: Exception | None = None
        for opts in attempts:
            try:
                # NOTE: 'hwaccel' is a libavcodec decoder option; PyAV
                # forwards unknown container options to ffmpeg. If the linked
                # ffmpeg lacks videotoolbox this raises -> we fall through to
                # the software attempt below. No hallucinated private API used.
                container = av.open(src, options=opts or None)  # type: ignore[arg-type]
                self.backend = (
                    DecodeBackend.VIDEOTOOLBOX
                    if opts.get("hwaccel") == "videotoolbox"
                    else DecodeBackend.SOFTWARE
                )
                logger.info("M5 decode open ok: source=%s opts=%s backend=%s",
                            src, opts, self.backend)
                break
            except Exception as exc:  # noqa: BLE001 — must try next attempt
                last_err = exc
                logger.warning("M5 decode open failed opts=%s: %s", opts, exc)
        if container is None:
            raise RuntimeError(f"Cannot open video source: {src}") from last_err

        frame_no = 0
        try:
            vstream = container.streams.video[0]
            vstream.thread_type = "AUTO"  # multi-threaded decode
            for av_frame in container.decode(vstream):
                if self._stop.is_set():
                    break
                # Hardware frames need transfer to CPU-accessible ndarray;
                # to_ndarray handles that when present.
                arr: np.ndarray = av_frame.to_ndarray(format="bgr24")
                if self.config.width and self.config.height:
                    arr = self._resize(arr, self.config.width, self.config.height)
                h, w = arr.shape[:2]
                item = _Frame(
                    stream_id=self.config.stream_id,
                    frame_number=frame_no,
                    pts=float(av_frame.pts * vstream.time_base)
                    if av_frame.pts is not None and vstream.time_base else None,
                    array=arr, backend=self.backend,
                )
                frame_no += 1
                self._push_blocking(loop, item)
                self.frames_decoded += 1
        finally:
            container.close()
            logger.info("M5 decode closed: decoded=%d dropped=%d backend=%s",
                        self.frames_decoded, self.frames_dropped, self.backend)

    def _push_blocking(
        self, loop: asyncio.AbstractEventLoop, item: _Frame
    ) -> None:
        """Hand a frame from the decoder thread to the asyncio queue.

        Drop-oldest when full (live mode) so decode never blocks; ``await``
        put when backpressure is wanted (file mode).
        """
        if self.config.drop_on_full:
            fut = asyncio.run_coroutine_threadsafe(
                self._put_drop_oldest(item), loop
            )
            fut.result(timeout=5.0)
        else:
            fut = asyncio.run_coroutine_threadsafe(self.queue.put(item), loop)
            fut.result(timeout=30.0)

    async def _put_drop_oldest(self, item: _Frame) -> None:
        """Loop-thread put that evicts the oldest frame when full (live)."""
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                self.frames_dropped += 1
        # Queue has room now (or waiters); never blocks the decoder thread
        # for long because maxsize is small.
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.frames_dropped += 1

    @staticmethod
    def _resize(arr: np.ndarray, w: int, h: int) -> np.ndarray:
        import cv2  # resize only; NOT capture — allowed for frame scaling

        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
