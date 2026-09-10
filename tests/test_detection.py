"""Smoke test 5.1: Plate + vehicle detection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.detection.detector import PlateDetector
import cv2

detector = PlateDetector(
    vehicle_model_path="models/detection/yolov8n.pt",
    device="cpu"
)
print("[OK] Detector initialized")

cap = cv2.VideoCapture("data/raw_videos/camera_1.mp4")
ret, frame = cap.read()
cap.release()

assert ret, "FAIL: Could not read frame from test video"
print(f"[OK] Frame read: {frame.shape}")

result = detector.detect(frame)
vehicles = result["vehicles"]
plates = result["plates"]

print(f"[OK] Detection complete: {len(vehicles)} vehicles, {len(plates)} plates")
for v in vehicles:
    print(f"  Vehicle: {v['class_name']} conf={v['confidence']:.3f} bbox={v['bbox']}")
for p in plates:
    print(f"  Plate: conf={p['confidence']:.3f} bbox={p['bbox']}")

annotated = frame.copy()
for v in vehicles:
    x1, y1, x2, y2 = v["bbox"]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{v['class_name']} {v['confidence']:.2f}"
    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite("tests/smoke_detection.jpg", annotated)
print("[OK] Saved annotated frame: tests/smoke_detection.jpg")
print("SMOKE TEST 5.1 PASSED")
