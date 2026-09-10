"""Smoke test 5.3: Per-camera tracking."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.tracking.tracker import VehicleTracker
import cv2
import numpy as np

tracker = VehicleTracker(max_age=30, n_init=2)
print("[OK] Tracker initialized")

cap = cv2.VideoCapture("data/raw_videos/camera_1.mp4")
frame_shape = (480, 640)

track_ids_seen = set()
stable_tracks = {}

for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    # Simulate detections (rectangle moving across frame)
    detections = [
        {"bbox": [50 + i*3, 250, 180 + i*3, 320], "confidence": 0.85, "class_name": "car"},
    ]
    tracked = tracker.update(detections, frame=frame)
    for t in tracked:
        tid = t["track_id"]
        track_ids_seen.add(tid)
        if tid not in stable_tracks:
            stable_tracks[tid] = 0
        stable_tracks[tid] += 1

cap.release()

print(f"[OK] Processed 30 frames")
print(f"[OK] Unique track IDs seen: {len(track_ids_seen)}")
for tid, count in stable_tracks.items():
    print(f"  Track {tid}: seen in {count}/30 frames")

assert len(track_ids_seen) <= 3, f"FAIL: Too many track IDs ({len(track_ids_seen)}), tracks not stable"
assert all(c >= 10 for c in stable_tracks.values()), "FAIL: Track IDs not stable across frames"
print("[OK] Track IDs stable across frames")

tracker.reset()
print("[OK] Tracker reset")

print("SMOKE TEST 5.3 PASSED")
