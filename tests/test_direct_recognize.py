"""Regression test: direct whole-crop recognition path (diagnostic experiment).

Proves, without guessing the EasyOCR API:
1. `build_variant_images` yields greyscale variants A–F (F skipped w/o quad).
2. `Reader.recognize` treats the whole crop as one region (box ~= full image).
3. `recognize_image` returns (str, conf-in-[0,1]).
4. `recognize_variants` returns the full trial schema incl. regex verdicts.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.ocr.reader import (
    PlateOCR,
    VARIANT_NAMES,
    build_variant_images,
    looks_like_plate,
)


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


# Synthetic plate crop: black text on white, like a real plate.
crop = np.full((120, 400, 3), 255, np.uint8)
cv2.putText(crop, "KA01AB1234", (20, 85), cv2.FONT_HERSHEY_SIMPLEX,
            2.2, (0, 0, 0), 5)

# 1. Variant builder: pure cv2, no model.
imgs = build_variant_images(crop)
check("6 variants", tuple(imgs.keys()) == VARIANT_NAMES)
for v in ("A_raw", "B_3x", "C_clahe", "D_otsu", "E_sharpen"):
    check(f"{v} greyscale uint8", imgs[v].ndim == 2 and imgs[v].dtype == np.uint8)
check("F skipped without quad", imgs["F_persp"] is None)
check("D_otsu is binary", set(np.unique(imgs["D_otsu"]).tolist()) <= {0, 255})
check("B_3x is 3x", imgs["B_3x"].shape == (360, 1200))
h, w = 120, 400
quad = [[0, 0], [w, 0], [w, h], [0, h]]
imgs_q = build_variant_images(crop, quad=quad)
check("F warps with quad", imgs_q["F_persp"] is not None)

# 2-4. Direct recognition (loads EasyOCR once; CPU).
ocr = PlateOCR()
box, _, _ = ocr.reader.recognize(
    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
    allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
)[0]
xs = [p[0] for p in box]
ys = [p[1] for p in box]
check("whole crop is one region",
      min(xs) <= 1 and max(xs) >= 399 and min(ys) <= 1 and max(ys) >= 119)

text, conf = ocr.recognize_image(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
check("recognize_image returns str", isinstance(text, str) and len(text) > 0)
check("conf in [0,1]", 0.0 <= conf <= 1.0)
print(f"   direct read: '{text}' @ {conf:.3f}")

trials = ocr.recognize_variants(crop)
check("6 trials", len(trials) == 6)
for t in trials:
    check(f"trial {t['variant']} schema",
          all(k in t for k in ("variant", "text", "conf", "normalized",
                               "regex_pass", "status", "image")))
ok = [t for t in trials if t["status"] == "ok" and t["text"]]
check(">=1 ok trial with text", len(ok) >= 1)
best = max(ok, key=lambda t: t["conf"])
print(f"   best variant: {best['variant']} '{best['text']}' @ {best['conf']:.3f} "
      f"regex={best['regex_pass']}")
check("regex verdict is bool", isinstance(best["regex_pass"], bool))

print("\nDIRECT-RECOGNITION TESTS PASSED")
