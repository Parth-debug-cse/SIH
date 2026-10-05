"""Phase-3 tests — OCR pipeline, no weights / GPU / mlx / pyobjc needed.

Covers: schema validation, CTC greedy decode, LPRNet preprocessing shape,
20px->100px upscale, homography quad deskew, stub recognizer output shape,
honest OCREngineUnavailable when no backend exists, async path.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain.ocr_schemas import OCRResult  # noqa: E402
from src.infrastructure.ocr.lprnet_mlx import (  # noqa: E402
    NUM_CLASSES,
    ctc_greedy_decode,
    preprocess_for_lprnet,
)
from src.infrastructure.ocr.plate_preprocessor import (  # noqa: E402
    PlatePreprocessor,
)
from src.infrastructure.ocr.plate_recognizer import (  # noqa: E402
    OCREngineUnavailable,
    PlateRecognizer,
    RecognizerConfig,
    normalize_plate_text,
)


def _plate_crop(w: int = 100, h: int = 20, text: str = "MH12AB1234") -> np.ndarray:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([1, 1, w - 2, h - 2], outline="black")
    d.text((6, 2), text[:8])
    return np.asarray(img)[:, :, ::-1].copy()  # BGR


# --- schemas ------------------------------------------------------------
def test_ocr_schema_validates() -> None:
    r = OCRResult(bbox=[1, 2, 3, 4], text="MH12AB1234", confidence=0.9,
                  backend="mlx_lprnet", inference_ms=3.2, mlx_inference_ms=2.8)
    assert r.mlx_inference_ms == 2.8 and r.inference_ms == 3.2
    with pytest.raises(Exception):
        OCRResult(bbox=[1, 2, 3], text="X", confidence=1.5,
                  backend="stub", inference_ms=0.0)


def test_normalize_plate_text() -> None:
    assert normalize_plate_text("mh-12 ab 1234!") == "MH12AB1234"


# --- CTC decode ---------------------------------------------------------
def test_ctc_greedy_decode_collapses_repeats_and_blanks() -> None:
    T = 6
    logits = np.full((T, NUM_CLASSES), -10.0, dtype=np.float32)
    a = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".index("A") + 1
    b = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".index("B") + 1
    logits[0, 0] = 10.0   # blank
    logits[1, a] = 10.0   # A
    logits[2, a] = 10.0   # A repeat -> collapsed
    logits[3, 0] = 10.0   # blank (separator)
    logits[4, a] = 10.0   # A again -> kept (blank separated)
    logits[5, b] = 10.0   # B
    text, conf = ctc_greedy_decode(logits)
    assert text == "AAB", text
    assert 0.0 < conf <= 1.0


def test_ctc_empty_on_all_blank() -> None:
    logits = np.zeros((4, NUM_CLASSES), dtype=np.float32)
    logits[:, 0] = 5.0
    text, conf = ctc_greedy_decode(logits)
    assert text == "" and conf == 0.0


def test_lprnet_preprocess_shape() -> None:
    x = preprocess_for_lprnet(_plate_crop())
    assert x.shape == (1, 24, 94, 1)
    assert x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0


# --- preprocessor -------------------------------------------------------
def test_upscale_20px_to_100px() -> None:
    pp = PlatePreprocessor()
    crop = _plate_crop(w=100, h=20)
    out, backend = pp.upscale(crop)
    assert out.shape[0] == 100, out.shape  # 20px -> 100px tall
    assert out.shape[1] == 500
    assert backend.startswith("lanczos_unsharp")


def test_process_returns_info_dict() -> None:
    pp = PlatePreprocessor()
    out, info = pp.process(_plate_crop(w=80, h=20))
    assert out.shape[0] == 100
    assert info["scale"] == pytest.approx(5.0)
    assert info["output_size"] == [400, 100]
    assert info["deskew_applied"] is False  # no cv2 here; graceful


def test_quad_deskew_rectifies() -> None:
    pp = PlatePreprocessor()
    crop = _plate_crop(w=120, h=30)
    h, w = crop.shape[:2]
    # Simulate a slanted view: top edge shifted right by 20px.
    quad = np.array([[20, 0], [w, 0], [w - 20, h], [0, h]], dtype=np.float64)
    warped, angle, applied = pp.deskew(crop, quad=quad)
    assert applied is True
    # Canonical size follows the quad's own edge lengths (~100 x ~36:
    # the slanted sides are longer than the 30px crop height).
    assert warped.shape[0] == 36 and warped.shape[1] == 100, warped.shape


def test_degenerate_input_raises_valueerror() -> None:
    pp = PlatePreprocessor()
    with pytest.raises(ValueError):
        pp.process(np.zeros((0, 0, 3), dtype=np.uint8))


# --- recognizer ---------------------------------------------------------
def test_stub_recognizer_output_schema() -> None:
    rec = PlateRecognizer(RecognizerConfig(backend="stub"))
    assert rec.backend == "stub"
    res = rec.recognize(_plate_crop(), bbox=[0.0, 0.0, 100.0, 20.0])
    assert isinstance(res, OCRResult)
    assert res.text == "MH12AB1234"
    assert res.bbox == [0.0, 0.0, 100.0, 20.0]
    assert res.mlx_inference_ms is None  # stub never claims MLX time
    assert res.preprocess_backend.startswith("lanczos_unsharp")
    assert 0.0 <= res.confidence <= 1.0


def test_no_backend_raises_honest_error() -> None:
    rec = PlateRecognizer(RecognizerConfig(backend="auto", mlx_weights="nope.npz"))
    with pytest.raises(OCREngineUnavailable):
        rec.backend  # noqa: B018 — property access must raise


def test_async_recognize_stub() -> None:
    async def _run() -> None:
        rec = PlateRecognizer(RecognizerConfig(backend="stub", stub_text="KA01ZZ9999"))
        res = await rec.arecognize(_plate_crop())
        assert res.text == "KA01ZZ9999" and res.backend == "stub"

    asyncio.run(_run())


if __name__ == "__main__":
    test_ocr_schema_validates()
    test_normalize_plate_text()
    test_ctc_greedy_decode_collapses_repeats_and_blanks()
    test_ctc_empty_on_all_blank()
    test_lprnet_preprocess_shape()
    test_upscale_20px_to_100px()
    test_process_returns_info_dict()
    test_quad_deskew_rectifies()
    test_degenerate_input_raises_valueerror()
    test_stub_recognizer_output_schema()
    try:
        test_no_backend_raises_honest_error()
    except OCREngineUnavailable:
        pass
    test_async_recognize_stub()
    print("PHASE3_OK")
