"""Debug dashcam plate-model run with explicit error handling."""
import sys, os, cv2, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO
plate_model = YOLO(r"C:\Users\dellc\.cache\huggingface\hub\models--Koushim--yolov8-license-plate-detection\snapshots\9aaa5cd490abe0c165882ba87f4f62658ab54d01\best.pt")
from src.detection.detector import PlateDetector

print("loading vehicle detector...", flush=True)
det = PlateDetector(vehicle_model_path="models/detection/yolov8n.pt", device="cpu")
print("loading video...", flush=True)
cap = cv2.VideoCapture("data/raw_videos/dash_2.mp4")
print("video opened:", cap.isOpened(), flush=True)
for fi in (240, 2460):
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        print(f"f{fi} read={ret} shape={frame.shape if ret else 'NA'}", flush=True)
        res = det.detect_vehicles(frame)
        print(f"f{fi} vehicles={len(res)}", flush=True)
        big = sorted([v for v in res if v["bbox"][3]-v["bbox"][1] > 90], key=lambda v: -(v["bbox"][3]-v["bbox"][1]))
        print(f"f{fi} big={len(big)}", flush=True)
        for v in big[:2]:
            x1, y1, x2, y2 = v["bbox"]
            crop = frame[y1:y2, x1:x2]
            up = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            r2 = plate_model.predict(up, conf=0.05, verbose=False)[0]
            nb = len(r2.boxes) if r2.boxes is not None else 0
            print(f"dash_2 f{fi} h={int(y2-y1)}px crop={up.shape[1]}x{up.shape[0]} plate_dets={nb}", flush=True)
            if r2.boxes is not None:
                for b in r2.boxes:
                    c = b.xyxy[0].cpu().numpy().astype(int)
                    print(f"   box {c.tolist()} conf={float(b.conf[0].item()):.2f}", flush=True)
    except Exception:
        traceback.print_exc()
cap.release()