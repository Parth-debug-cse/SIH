"""Smoke test 5.4: Cross-camera fusion / re-identification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.fusion.reid import CrossCameraFusion
import json

fusion = CrossCameraFusion(
    cameras_config_path="data/calibration/cameras.json",
    max_edit_distance=2,
    max_speed_kmh=150.0,
)
print(f"[OK] Fusion initialized with {len(fusion.cameras)} cameras")
for cam_id, cam in fusion.cameras.items():
    print(f"  {cam_id}: lat={cam.gps_lat}, lon={cam.gps_lon}")

# Test haversine distance
d = fusion.compute_distance_km(12.9758, 77.6082, 12.9768, 77.6050)
print(f"[OK] Distance between cam_1 and cam_2: {d:.3f} km")

# Add sightings for same plate from different cameras
import time
now = time.time()

fusion.add_sighting("KA01AB1234", "cam_1", 12.9758, 77.6082, now, 0.95)
fusion.add_sighting("KA01AB1234", "cam_2", 12.9768, 77.6050, now + 30, 0.88)  # 30s later
fusion.add_sighting("KA01AB1234", "cam_3", 12.9745, 77.6095, now + 60, 0.91)  # 60s later

# Test spatiotemporal plausibility
s1 = {"gps_lat": 12.9758, "gps_lon": 77.6082, "timestamp": now}
s2 = {"gps_lat": 12.9768, "gps_lon": 77.6050, "timestamp": now + 30}
plausible = fusion.is_spatiotemporally_plausible(s1, s2)
print(f"[OK] Spatiotemporal plausibility (30s gap): {plausible}")

# Test plate matching
matches = fusion.match_plates(fusion.sightings)
print(f"[OK] Plate matches found: {len(matches)}")
for m in matches:
    print(f"  {m['plate_a']} <-> {m['plate_b']}: sim={m['similarity']:.1f}, plausible={m['plausible']}")

# Test trajectory fusion
trajectories = fusion.fuse_trajectories()
print(f"[OK] Fused trajectories: {len(trajectories)}")
assert len(trajectories) == 1, f"FAIL: expected 1 trajectory, got {len(trajectories)}"
traj = trajectories[0]
sightings = traj["sightings"]
cameras = [s["camera_id"] for s in sightings]
print(f"  '{traj['canonical_plate']}' ({traj['trajectory_id']}): route = {' -> '.join(cameras)}")
assert traj["trajectory_id"] == "trajectory_0"
assert traj["canonical_plate"] == "KA01AB1234"
assert len(sightings) == 3, f"FAIL: Expected 3 sightings, got {len(sightings)}"
assert cameras == ["cam_1", "cam_2", "cam_3"], f"FAIL: Route order wrong: {cameras}"

print("SMOKE TEST 5.4 PASSED")
