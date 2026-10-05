"""Zero-latency fallback: Apple's native Vision text recognition.

Uses ``VNRecognizeTextRequest`` via ``pyobjc-framework-Vision`` — on-device,
no model download, no server. Accuracy is below a trained LPRNet on distant
plates (hence fallback, not primary), but latency is near-zero and it works
offline on any Apple Silicon Mac.

Requires: macOS 10.15+, ``pip install pyobjc-framework-Vision``.
All Apple imports are lazy; without them the backend reports unavailable
instead of raising at import time.
"""

from __future__ import annotations

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


def vision_available() -> bool:
    try:
        import Vision  # noqa: F401
        import Foundation  # noqa: F401

        return True
    except Exception:
        return False


def _to_jpeg_bytes(crop_bgr: np.ndarray) -> bytes:
    from PIL import Image

    rgb = Image.fromarray(crop_bgr[:, :, ::-1])
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def recognize_with_vision(crop_bgr: np.ndarray) -> tuple[str, float]:
    """Run VNRecognizeTextRequest on a BGR crop -> (text, confidence).

    Raises:
        ImportError: pyobjc Vision bindings missing.
        RuntimeError: the request itself failed.
    """
    try:
        import Foundation
        import Vision
    except Exception as exc:
        raise ImportError(
            "pyobjc-framework-Vision is required for the Vision fallback: "
            "pip install pyobjc-framework-Vision (macOS only)."
        ) from exc

    jpeg = _to_jpeg_bytes(np.asarray(crop_bgr))
    nsdata = Foundation.NSData.dataWithBytes_length_(jpeg, len(jpeg))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(nsdata, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()

    accurate = getattr(Vision, "VNRequestTextRecognitionLevelAccurate", 1)
    try:
        request.setRecognitionLevel_(accurate)
    except Exception:
        pass
    try:
        request.setUsesLanguageCorrection_(False)  # plates aren't prose
    except Exception:
        pass
    if request.respondsToSelector_("setRecognitionLanguages:"):
        try:
            request.setRecognitionLanguages_(["en-US"])
        except Exception:
            pass

    performed = handler.performRequests_error_([request], None)
    # pyobjc maps NSError** to a (bool, error) tuple; be liberal in parsing.
    ok = performed[0] if isinstance(performed, tuple) else bool(performed)
    err = performed[1] if isinstance(performed, tuple) and len(performed) > 1 else None
    if not ok:
        raise RuntimeError(f"VNRecognizeTextRequest failed: {err}")

    best_text, best_conf = "", 0.0
    for obs in (request.results() or []):
        try:
            cands = obs.topCandidates_(1)
        except Exception:
            continue
        if not cands:
            continue
        cand = cands[0]
        try:
            text, conf = str(cand.string()), float(cand.confidence())
        except Exception:
            continue
        if conf >= best_conf:
            best_text, best_conf = text, conf
    return best_text, best_conf
