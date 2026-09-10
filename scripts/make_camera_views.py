"""Split real highway footage into 3 distinct camera views for demo.

cam_1: seconds 0-20, full frame (highway overpass view)
cam_2: seconds 20-40, full frame (later traffic, day) 
cam_3: seconds 40-60, bottom crop (closer road view for plate legibility)
"""
import cv2
import os

SRC = "data/raw_videos/camera_2.full.mp4"
OUT_DIR = "data/raw_videos"

def extract_segment(src, out_path, start_s, end_s, crop=None):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if crop:
        w = crop[2] - crop[0]
        h = crop[3] - crop[1]
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_n / fps
        if start_s <= t < end_s:
            if crop:
                frame = frame[crop[1]:crop[3], crop[0]:crop[2]]
            out.write(frame)
        frame_n += 1
        if t >= end_s:
            break
    cap.release()
    out.release()
    print(f"Created {out_path} ({int((end_s-start_s))}s)")

os.makedirs(OUT_DIR, exist_ok=True)

extract_segment(SRC, f"{OUT_DIR}/camera_1.mp4", 0, 20)
extract_segment(SRC, f"{OUT_DIR}/camera_2.mp4", 20, 40)
extract_segment(SRC, f"{OUT_DIR}/camera_3.mp4", 40, 60, crop=(0, 300, 1920, 1080))

# Clean up source extra
extra = f"{OUT_DIR}/camera_extra.mp4"
if os.path.exists(extra):
    os.remove(extra)

print("All 3 camera views created from real footage")