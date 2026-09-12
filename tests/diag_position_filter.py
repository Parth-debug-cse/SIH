"""Diagnostic: how many plate candidates pass vs fail the posision filter?"""
import sys, os, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.detector import PlateDetector

PATH = "data/raw_videos/WhatsApp Video 2026-09-11 at 1.35.19 PM.mp4"
plate_path = r"C:\Users\dellc\.cache\huggingface\hub\models--Koushim--yolov8-license-plate-detection\snapshots\9aaa5cd490abe0c165882ba87f4f62658ab54d01\best.pt"
det = PlateDetector(
    vehicle_model_path="models/detection/yolov8n.pt",
    plate_model_path=plate_path,
    plate_conf_threshold=0.05,
    plate_iou_threshold=0.45,
    device="cpu",
)

cap = cv2.VideoCapture(PATH)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample = list(range(0, n, 4))

total_dets = 0
passing = 0
rejected_pos = 0
rejected_aspect = 0
unassoc = 0
examples_fail = []
examples_pass = []

for i in sample:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    vehicles = det.detect_vehicles(frame)
    plates = det.detect_plates(frame, vehicle_detections=vehicles)
    # raw candidates ignoring filters (re-run raw detect for counting):
    raw = det._detect_plates_model(frame)
    total_dets += len(raw)
    passing += len(plates)
    for p in raw:
        idx = det._best_vehicle_match(p["bbox"], vehicles)
        if idx < 0:
            unassoc += 1
            continue
        veh = vehicles[idx]["bbox"]
        if not det._plate_sits_in_bumper_zone(p["bbox"], veh):
            rejected_pos += 1
            if len(examples_fail) < 6:
                examples_fail.append((i, p["bbox"], veh))
        else:
            if len(examples_pass) < 6:
                examples_pass.append((i, p["bbox"], veh))
    det.reset_plate_aspect_ratio_counter()
    rejected_aspect += det.plate_aspect_ratio_rejects
cap.release()

print(f"sampled {len(sample)} frames")
print(f"raw plate-model detections: {total_dets}")
print(f"rejected by bumper-position filter: {rejected_pos}")
print(f"rejected by aspect filter (applied in _detect_plates_model): {rejected_aspect}")
print(f"unassociated (no parent vehicle): {unassoc}")
print(f"kept after filters: {passing}")
print("\n-- FAIL (rejected by position) --")
for i, pb, vb in examples_fail:
    print(f"  f{i} plate_box={pb} inside vehicle={vb}")
print("\n-- PASS --")
for i, pb, vb in examples_pass:
    print(f"  f{i} plate_box={pb} inside vehicle={vb}")