"""Smoke test 5.6: Macro traffic analytics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.analytics.engine import TrafficAnalytics
import time

analytics = TrafficAnalytics(cameras_config_path="data/calibration/cameras.json")
print(f"[OK] Analytics initialized with {len(analytics.cameras)} cameras")

now = time.time()

# Simulate camera counts over time
for i in range(20):
    ts = now + i
    analytics.update_camera_count("cam_1", ts, 3 + (i % 5))
    analytics.update_camera_count("cam_2", ts, 2 + (i % 3))

# Test density
density = analytics.compute_density("cam_1", time_window_seconds=600)
print(f"[OK] Density for cam_1: {density:.2f}")

# Test congestion detection
congestion = analytics.detect_congestion("cam_1", time_window_seconds=600)
print(f"[OK] Congestion for cam_1: level={congestion['congestion_level']}, congested={congestion['is_congested']}")

# Test speed computation
bbox1 = [100, 200, 230, 270]
bbox2 = [200, 200, 330, 270]
speed = analytics.compute_speed("cam_1", bbox1, bbox2, time_delta=1.0)
print(f"[OK] Speed estimate: {speed:.1f} km/h")

# Test OD patterns
trajectories = {
    "KA01AB1234": [
        {"camera_id": "cam_1", "timestamp": now},
        {"camera_id": "cam_2", "timestamp": now + 30},
    ],
    "KA02CD5678": [
        {"camera_id": "cam_1", "timestamp": now + 5},
        {"camera_id": "cam_3", "timestamp": now + 35},
    ],
}
od = analytics.compute_od_patterns(trajectories)
print(f"[OK] OD patterns: {len(od)} patterns")
for p in od:
    print(f"  {p['origin_camera']} -> {p['dest_camera']}: {p['count']} vehicles")

# Test summary
summary = analytics.get_analytics_summary()
print(f"[OK] Analytics summary for {len(summary)} cameras")
for cam_id, stats in summary.items():
    print(f"  {cam_id}: total={stats['total_vehicles']}, speed={stats['avg_speed_kmh']:.1f} km/h")

print("SMOKE TEST 5.6 PASSED")
