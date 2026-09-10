"""Test detection on real footage."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.detection.detector import PlateDetector

cap = cv2.VideoCapture("data/raw_videos/camera_2.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("FAIL: could not read frame")
    sys.exit(1)

detector = PlateDetector(vehicle_model_path="models/detection/yolov8n.pt", device="cpu")
print("Running detection on real frame...")
t0 = time.time()
result = detector.detect(frame)
elapsed = time.time() - t0
print(f"Detection took {elapsed:.1f}s on CPU")
print(f"Vehicles detected: {len(result['vehicles'])}")
for v in result["vehicles"][:5]:
    print(f"  {v['class_name']}: conf={v['confidence']:.2f} bbox={v['bbox']}")

annotated = frame.copy()
for v in result["vehicles"]:
    x1, y1, x2, y2 = v["bbox"]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{v['class_name']} {v['confidence']:.2f}"
    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite("tests/real_footage_detection.jpg", annotated)
print("Saved: tests/real_footage_detection.jpg")