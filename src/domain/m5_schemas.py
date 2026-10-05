"""Phase-1 domain schemas — pure pydantic, no I/O, no hardware deps."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DecodeBackend(str, Enum):
    VIDEOTOOLBOX = "videotoolbox"
    SOFTWARE = "software"


class InferenceBackend(str, Enum):
    COREML_ANE = "coreml_ane"
    MPS = "mps"
    CPU = "cpu"


class FramePacket(BaseModel):
    """Single decoded frame travelling decoder -> inference."""

    model_config = {"arbitrary_types_allowed": True}

    stream_id: str = Field(description="Camera / stream identifier")
    frame_number: int = Field(ge=0)
    pts: Optional[float] = Field(default=None, description="Presentation timestamp (s)")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    decode_backend: DecodeBackend = DecodeBackend.SOFTWARE

    # ndarray payload excluded from pydantic validation for speed;
    # stored as opaque object. Use .payload to access.


class Detection(BaseModel):
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int
    class_name: str


class FrameResult(BaseModel):
    stream_id: str
    frame_number: int
    backend: InferenceBackend
    detections: list[Detection] = Field(default_factory=list)
    inference_ms: float = Field(ge=0.0)


class M5DeviceInfo(BaseModel):
    platform: str
    arch: str
    mps_available: bool
    mlx_available: bool
    backend: InferenceBackend
    source: Literal["probe", "fallback"] = "probe"
