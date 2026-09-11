"""Test OCR plate reads on 1080p dashcam footage.

Reports reads that pass the same 0.50 confidence gate used by the live
pipeline (anti-hallucination contract).  Reads below the gate are dropped.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from src.detection.detector import PlateDetector
from src.ocr.reader import PlateOCR

MIN_OCR_CONFIDENCE = 0.50

detector = PlateDetector(vehicle_model_path="models/detection/yolov8n.pt", device="cpu")
ocr = PlateOCR()

cap = cv2.VideoCapture("data/raw_videos/dash_high1.MOV")
reads = []
veh_count = 0
max_h = 0
for frame_i in range(0, 750, 15):  # every 0.5s
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
    ret, frame = cap.read()
    if not ret:
        break
    res = detector.detect(frame)
    for v in res["vehicles"]:
        x1, y1, x2, y2 = v["bbox"]
        veh_h = y2 - y1
        max_h = max(max_h, veh_h)
        if veh_h < 60:
            continue
        veh_count += 1
        # Try multiple crop regions of the vehicle
        for sname, (fr_top, fr_bot) in {
            "bottom30": (0.70, 1.0),
            "mid": (0.55, 0.80),
        }.items():
            crop = frame[int(y1 + veh_h * fr_top):int(y1 + veh_h * fr_bot), x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 20:
                continue
            up = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            r = ocr.read_plate(up)
            if r["text"] and r["confidence"] >= MIN_OCR_CONFIDENCE:
                reads.append((frame_i, v["class_name"], int(veh_h), sname, r["text"], r["confidence"]))
                print(f"READ @f{frame_i} [{sname}] h={int(veh_h)}px: '{r['text']}' conf={r['confidence']:.2f}")
            ocr.clear_history()
cap.release()
print(f"\nVehicles >=60px: {veh_count}, max_h={int(max_h)}px, "
      f"reads at conf>={MIN_OCR_CONFIDENCE}: {len(reads)}")