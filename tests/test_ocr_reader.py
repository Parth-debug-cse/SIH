"""pytest suite: HybridPlateOCR routing + is_valid_india_plate.

No model downloads, no GPU: both backends are stubbed/spied.  Proves the
height gate routes correctly and the state-code whitelist rejects the exact
false positives seen on real footage ('AA05440', 'GD75000').
"""
import numpy as np

from src.ocr.reader import HybridPlateOCR, is_valid_india_plate


class _FastSpy:
    """Stub fast-plate-ocr backend counting invocations."""

    def __init__(self, text="KA03NS5132", conf=0.955):
        self.call_count = 0
        self._text = text
        self._conf = conf

    def read_crop(self, crop_bgr):
        self.call_count += 1
        return {"text": self._text, "conf": self._conf, "status": "ok"}


class _EasyStub:
    """Stub EasyOCR fallback returning a canned voted read."""

    def __init__(self, text="", conf=0.0):
        self._text = text
        self._conf = conf

    def read_plate(self, crop_bgr, track_key=None):
        return {"text": self._text, "confidence": self._conf,
                "raw_text": self._text, "raw_confidence": self._conf,
                "voted_count": 1, "voted": False}


def _crop(h, w=120):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_short_crop_never_calls_fastplate():
    fast = _FastSpy()
    hybrid = HybridPlateOCR(easy_ocr=_EasyStub(), fast_backend=fast)
    res = hybrid.read_plate(_crop(40), crop_height_px=40)
    assert res["engine_used"] == "easyocr"
    assert fast.call_count == 0


def test_tall_crop_uses_fastplate():
    fast = _FastSpy()
    hybrid = HybridPlateOCR(easy_ocr=_EasyStub(), fast_backend=fast)
    res = hybrid.read_plate(_crop(80), crop_height_px=80)
    assert res["engine_used"] == "fastplate"
    assert fast.call_count == 1
    assert res["text"] == "KA03NS5132"
    assert res["confidence"] == 0.955
    assert res["regex_pass"] is True


def test_short_crop_empty_easyocr_reports_resolution_floor():
    hybrid = HybridPlateOCR(easy_ocr=_EasyStub(text="", conf=0.0),
                            fast_backend=_FastSpy())
    res = hybrid.read_plate(_crop(43), crop_height_px=43)
    assert res["reason"] == "below_resolution_floor"
    assert res["text"] == ""


def test_aa05440_rejected_state_code():
    assert is_valid_india_plate("AA05440") == (False, "invalid_state_code")


def test_gd75000_rejected_state_code():
    assert is_valid_india_plate("GD75000") == (False, "invalid_state_code")


def test_ka03ns5132_accepted():
    assert is_valid_india_plate("KA03NS5132") == (True, "ok")


def test_every_return_path_carries_voted_count():
    # Guards the pipeline_runner accepted-branch log line, which reads
    # hybrid_res.get("voted_count", 0): every dict below must carry the key.
    fast = _FastSpy()
    hybrid = HybridPlateOCR(easy_ocr=_EasyStub(), fast_backend=fast)
    r1 = hybrid.read_plate(_crop(80), crop_height_px=80)
    assert r1["voted_count"] == 0
    hybrid2 = HybridPlateOCR(
        easy_ocr=_EasyStub(text="KA01AB1234", conf=0.9),
        fast_backend=_FastSpy(),
    )
    r2 = hybrid2.read_plate(_crop(40), crop_height_px=40)
    assert r2["voted_count"] == 1  # real EasyOCR vote depth passes through
    r3 = hybrid.read_plate(_crop(40), crop_height_px=40)
    assert r3["voted_count"] == 0
