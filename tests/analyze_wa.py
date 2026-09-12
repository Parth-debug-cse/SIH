"""Analyze the user's WhatsApp test video: detect vehicles, plate regions, run OCR."""
import sys, os, cv2, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.detector import PlateDetector
from src.ocr.reader import PlateOCR

PATH = "data/raw_videos/WhatsApp Video 2026-09-11 at 1.35.19 PM.mp4"
det = PlateDetector(vehicle_model_path="models/detection/yolov8n.pt", device="cpu")
ocr = PlateOCR()

cap = cv2.VideoCapture(PATH)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
max_h = 0
best = []
sample = list(range(0, n, max(1, n // 24)))
for i in sample:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    res = det.detect_vehicles(frame)
    for v in res:
        h = v["bbox"][3] - v["bbox"][1]
        if h >= 60:
            best.append({"frame": i, "h": h, "bbox": v["bbox"], "cls": v["class_name"]})
cap.release()

best.sort(key=lambda d: -d["h"])
print(f"vehicles>=60px across {len(sample)} samples: {len(best)}")
for d in best[:10]:
    print(f"  f{d['frame']} {d['cls']} h={int(d['h'])}px bbox={d['bbox']}")

# OCR the largest vehicle plates in the top frames
cap = cv2.VideoCapture(PATH)
seen_frames = set()
TOP_N = 4
for d in best:
    fi = d["frame"]
    if fi in seen_frames:
        continue
    if len(seen_frames) >= TOP_N:
        break
    seen_frames.add(fi)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue
    x1, y1, x2, y2 = d["bbox"]
    veh = frame[y1:y2+10, x1:x2]
    cv2.imwrite(f"tests/wa_vehicle_f{fi}.jpg", veh)
    # multi-band OCR
    hh, ww = veh.shape[:2]
    for tf in (0.60, 0.70, 0.80, 0.90):
        band = veh[int(hh*tf):min(int(hh*(tf+0.15)), hh), :]
        if band.shape[0] < 8 or band.shape[1] < 20:
            continue
        up = cv2.resize(band, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        r = ocr.read_plate(up)
        if r["text"]:
            print(f"  f{fi} band{tf:.2f}: '{r['text']}' conf={r['confidence']:.2f}")
        ocr.clear_history()
cap.release()