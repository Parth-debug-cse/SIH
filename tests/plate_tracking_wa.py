"""Track one plate across a frame window, OCR with multi-scale variants, vote."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH = "data/raw_videos/WhatsApp Video 2026-09-11 at 1.35.19 PM.mp4"
from src.ocr.reader import PlateOCR
ocr = PlateOCR()
reader = ocr.reader

def variants(gray):
    yield "raw", gray
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    yield "clahe", clahe.apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu", otsu
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    yield "denoise", blur

def best_registration(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    best = ("", 0.0)
    for vname, g in variants(gray):
        for scale in (2.0, 3.0, 4.0, 6.0):
            up = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            res = reader.readtext(up, detail=1, paragraph=False)
            for _b, t, c in res:
                t = "".join(ch for ch in t.upper() if ch.isalnum())
                n_d = sum(ch.isdigit() for ch in t)
                n_a = sum(ch.isalpha() for ch in t)
                if n_d >= 1 and n_a >= 1 and 5 <= len(t) <= 10:
                    if c > best[1]:
                        best = (t, float(c))
                        print(f"    {vname}/{scale}x: {t!r} conf={c:.2f}")
    return best

keyboxes = [(308, [203, 366, 331, 404]), (4, [38, 350, 202, 406])]
cap = cv2.VideoCapture(PATH)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for anchor, box in keyboxes:
    print(f"\n=== plate box {box} around f{anchor} ===")
    x1, y1, x2, y2 = box
    for win in range(max(0, anchor-20), min(n, anchor+21), 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, win)
        ret, frame = cap.read()
        if not ret:
            continue
        crop = frame[max(0,y1-5):y2+5, max(0,x1-5):x2+5]
        if crop.shape[0] < 6 or crop.shape[1] < 10:
            continue
        # try a few horizontal offsets in case the box shifted slightly
        placements = [crop]
        for dx in (-6, 6):
            c = frame[max(0,y1-5):y2+5, max(0,x1-5+dx):x2+5+dx]
            if c.shape == crop.shape:
                placements.append(c)
        for pc in placements:
            b = best_registration(pc)
            if b[1] > 0.35 and b[0]:
                print(f"  f{win}: best '{b[0]}' conf={b[1]:.2f}")
cap.release()