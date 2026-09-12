"""Dump EasyOCR per-row reads inside chosen plate crops to design a fix."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr.reader import PlateOCR
ocr = PlateOCR()
reader = ocr.reader

PATH = "data/raw_videos/WhatsApp Video 2026-09-11 at 1.35.19 PM.mp4"

# (frame, plate_box) picked from the run diagnostic + known good reads
targets = [
    (4, [38, 350, 202, 406]),      # 'F7502' earlier
    (308, [203, 366, 331, 404]),   # 'KAOZHUZE99'
    (76, [178, 140, 307, 204]),    # 'BMTC' 1.0 - brand box
    (112, [152, 185, 336, 245]),   # 'BENGALURU'
    (160, [173, 499, 305, 535]),   # bumper pass example
]

cap = cv2.VideoCapture(PATH)
for fi, box in targets:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue
    x1, y1, x2, y2 = box
    crop = frame[max(0,y1-4):min(frame.shape[0],y2+4), max(0,x1-4):min(frame.shape[1],x2+4)]
    if crop.shape[0] < 8 or crop.shape[1] < 16:
        print(f"f{fi} box={box}: crop too small", flush=True)
        continue
    cv2.imwrite(f"tests/wa_crop_f{fi}.jpg", crop)
    up = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
    res = reader.readtext(up, detail=1, paragraph=False)
    print(f"\n=== f{fi} box={box} crop={crop.shape[1]}x{crop.shape[0]} ===", flush=True)
    for _b, t, c in res:
        t2 = "".join(ch for ch in t.upper() if ch.isalnum())
        if t2:
            print(f"  '{t2}' conf={c:.2f}", flush=True)
cap.release()