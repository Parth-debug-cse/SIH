"""Phase-3 OCR data schemas — pure pydantic, no hardware deps."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """Single plate-read result. EasyOCR-free by design."""

    bbox: List[float] = Field(
        ..., min_length=4, max_length=4,
        description="Plate box [x1, y1, x2, y2] in source-frame pixels",
    )
    text: str = Field(..., description="Normalized plate string (e.g. MH12AB1234)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    backend: str = Field(
        ..., description="Engine that produced this read: mlx_lprnet | vision | stub"
    )
    inference_ms: float = Field(..., ge=0.0, description="Recognizer wall time (ms)")
    mlx_inference_ms: Optional[float] = Field(
        default=None,
        description="Time spent inside the MLX graph (ms); None when the "
        "read did not go through MLX (e.g. Vision fallback)",
    )
    preprocess_backend: Optional[str] = Field(
        default=None, description="Upscaler that fed the recognizer"
    )
    deskew_applied: bool = Field(default=False)


class PreprocessInfo(BaseModel):
    scale: float = Field(..., gt=0.0)
    output_size: List[int] = Field(..., min_length=2, max_length=2)  # [w, h]
    deskew_applied: bool = False
    deskew_angle_deg: Optional[float] = None
    upscale_backend: str = "unknown"
