"""Use dedicated plate model to localize + OCR plates on the user's WhatsApp video."""
import sys, os, cv2, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.detector import PlateDetector
from src.ocr.reader import PlateOCR

PATH = "data/raw_videos/WhatsApp Video 2026-09-11 at 1.35.19 PM.mp4"
plate_path = r"C:\Users\dellc\.cache\huggingface\hub\models--Koushim--yolov8-license-plate-detection\snapshots\9aaa5cd490abe0c165882ba87f4f62658ab54d01\best.pt"
det = PlateDetector(
    vehicle_model_path="models/detection/yolov8n.pt",
    plate_model_path=plate_path,
    plate_conf_threshold=0.05,
    device="cpu",
)
ocr = PlateOCR()

cap = cv2.VideoCapture(PATH)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample = list(range(0, n, 4))  # every 4th frame = one per 0.17s
total_plate_boxes = 0
reads = []

for i in sample:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    plates = det.detect_plates(frame)
    if not plates:
        continue
    for p in plates:
        ox1, oy1, ox2, oy2 = p["bbox"]
        total_plate_boxes += 1
        crop = frame[max(0,oy1):oy2, max(0,ox1):ox2]
        if crop.shape[0] < 8 or crop.shape[1] < 16:
            continue
        up = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        r = ocr.read_plate(up)
        if r["text"] and r["confidence"] >= 0.10:
            reads.append((r["text"], r["confidence"], i, crop.shape, p["bbox"]))
        ocr.clear_history()
cap.release()

print(f"sampled {len(sample)} frames, plate boxes found: {total_plate_boxes}")
reads.sort(key=lambda r: -r[1])
print(f"reads with conf>=0.10: {len(reads)}")
for t, c, fi, shp, bb in reads[:25]:
    print(f"  f{fi:4d} '{t}' conf={c:.3f} crop={shp[1]}x{shp[0]} box={bb}")