"""Regression test: fast-plate-ocr backend adapter (no model download).

Proves with a stubbed recognizer (deterministic, offline):
1. BGR crops are converted to RGB (model requirement).
2. PlatePrediction parsing: text stripped, conf = mean(char_probs),
   conf None when the API provides none (never fabricated).
3. `read_crop` returns the contracted schema.
Also asserts the installed package version is detectable.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.ocr.fastplate import (
    FastPlateOCR,
    available_providers,
    package_version,
)


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


# 1. BGR -> RGB conversion (R and B channels must swap).
bgr = np.zeros((10, 20, 3), np.uint8)
bgr[:, :, 0] = 255  # pure blue in BGR
rgb = FastPlateOCR.to_rgb(bgr)
check("rgb shape/dtype", rgb.shape == (10, 20, 3) and rgb.dtype == np.uint8)
check("channels swapped", rgb[:, :, 0].max() == 0 and rgb[:, :, 2].max() == 255)

# 2. Prediction parsing.
pred = SimpleNamespace(plate=" KA01AB1234 ", has_confidence=True,
                       char_probs=[0.9, 0.8, 1.0])
text, conf = FastPlateOCR.parse_prediction(pred)
check("text stripped", text == "KA01AB1234")
check("conf is mean", abs(conf - 0.9) < 1e-9)
pred_nc = SimpleNamespace(plate="MH12CD5678", has_confidence=False,
                           char_probs=None)
text2, conf2 = FastPlateOCR.parse_prediction(pred_nc)
check("conf None when absent", text2 == "MH12CD5678" and conf2 is None)

# 3. read_crop with stubbed recognizer (no download, no model).
be = FastPlateOCR()
be._rec = SimpleNamespace(run=lambda img, return_confidence=True: [pred])
out = be.read_crop(np.zeros((30, 100, 3), np.uint8))
check("read_crop schema",
      all(k in out for k in ("text", "conf", "status")))
check("read_crop values", out["text"] == "KA01AB1234"
      and abs(out["conf"] - 0.9) < 1e-9 and out["status"] == "ok")

# 4. Environment facts.
check("package version known", package_version() not in ("", "unknown"))
print("   fast-plate-ocr version:", package_version())
print("   onnx providers:", available_providers())

print("\nFASTPLATE BACKEND TESTS PASSED")
