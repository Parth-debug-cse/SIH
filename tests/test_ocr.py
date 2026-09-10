"""Smoke test 5.2: OCR engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.ocr.reader import PlateOCR
import cv2
import numpy as np

ocr = PlateOCR(lang="en", use_gpu=False)
print("[OK] OCR engine initialized")

# Create synthetic plate images with text
plate_texts = ["KA01AB1234", "MH12CD5678", "DL01EF9012"]
results = []

for text in plate_texts:
    img = np.ones((50, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    result = ocr.read_plate(img)
    results.append({"expected": text, "got": result["text"], "conf": result["confidence"]})
    print(f"  Expected: {text} -> Got: '{result['text']}' (conf={result['confidence']:.3f})")

correct = sum(1 for r in results if r["expected"].replace(" ", "") in r["got"].replace(" ", ""))
print(f"[OK] OCR accuracy on synthetic plates: {correct}/{len(results)}")

# Also test on a real frame crop
cap = cv2.VideoCapture("data/raw_videos/camera_1.mp4")
ret, frame = cap.read()
cap.release()
if ret:
    # Crop center region as a test
    h, w = frame.shape[:2]
    crop = frame[h//2-25:h//2+25, w//2-100:w//2+100]
    result = ocr.read_plate(crop)
    print(f"[OK] OCR on frame crop: '{result['text']}' (conf={result['confidence']:.3f})")

ocr.clear_history()
print("[OK] History cleared")

print("SMOKE TEST 5.2 PASSED")
