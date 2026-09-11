"""Populate DB with realistic test sightings for integration testing.

Only **sightings** are seeded here.  Trajectories are NOT inserted by hand:
the real fusion engine consumes the seeded sightings from the shared database
and writes fused trajectory rows, exactly as it does in the live pipeline.

Run: python tests/populate_test_data.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.schema import init_db, get_connection

init_db()
conn = get_connection()
now = time.time()

plates_data = [
    ("KA01AB1234", 0.95, "cam_1", 12.9758, 77.6082, now, "car", 11),
    ("KA01AB1234", 0.88, "cam_2", 12.9768, 77.6050, now + 45, "car", 21),
    ("KA01AB1234", 0.91, "cam_3", 12.9745, 77.6095, now + 90, "car", 31),
    ("MH12CD5678", 0.87, "cam_1", 12.9758, 77.6082, now + 5, "car", 12),
    ("MH12CD5678", 0.92, "cam_2", 12.9768, 77.6050, now + 50, "car", 22),
    ("DL01EF9012", 0.83, "cam_3", 12.9745, 77.6095, now + 10, "truck", 32),
]
conn.executemany(
    "INSERT INTO sightings (plate, plate_confidence, camera_id, gps_lat, gps_lon, timestamp, vehicle_class, track_id) VALUES (?,?,?,?,?,?,?,?)",
    plates_data,
)
conn.commit()

for i in range(5):
    conn.execute(
        "INSERT INTO analytics (camera_id, timestamp, vehicle_count, avg_speed, density_level, congestion_flag) VALUES (?,?,?,?,?,?)",
        ("cam_1", now + i, 3 + i, 25.0 + i * 2, "medium", 0),
    )
conn.commit()

conn.execute(
    "INSERT INTO alerts (plate, alert_type, camera_id, gps_lat, gps_lon, timestamp, details) VALUES (?,?,?,?,?,?,?)",
    ("MH12AB1234", "blacklist_exact", "cam_1", 12.9758, 77.6082, now, '{"matched":"MH12AB1234"}'),
)
conn.commit()
conn.close()
print("[OK] Test sightings populated")

# Run the real fusion engine so trajectories are produced by fusion, not manual seeding.
from src.fusion.engine import run_fusion_once
stats = run_fusion_once()
print(f"[OK] Fusion produced {stats['num_trajectories']} trajectory(ies) "
      f"from {stats['num_sightings']} sighting(s): {stats['trajectories']}")