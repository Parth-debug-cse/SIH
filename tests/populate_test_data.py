"""Populate DB with realistic test data for integration testing."""
import sys, os, time, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.schema import init_db

init_db()
conn = sqlite3.connect("data/anpr.db")
now = time.time()

plates_data = [
    ("KA01AB1234", 0.95, "cam_1", 12.9758, 77.6082, now),
    ("KA01AB1234", 0.88, "cam_2", 12.9768, 77.6050, now + 45),
    ("KA01AB1234", 0.91, "cam_3", 12.9745, 77.6095, now + 90),
    ("MH12CD5678", 0.87, "cam_1", 12.9758, 77.6082, now + 5),
    ("MH12CD5678", 0.92, "cam_2", 12.9768, 77.6050, now + 50),
    ("DL01EF9012", 0.83, "cam_3", 12.9745, 77.6095, now + 10),
]
for p in plates_data:
    conn.execute(
        "INSERT INTO sightings (plate, plate_confidence, camera_id, gps_lat, gps_lon, timestamp, frame_number) VALUES (?,?,?,?,?,?,?)",
        (p[0], p[1], p[2], p[3], p[4], p[5], 1),
    )
conn.commit()

trajs = [
    ("KA01AB1234", "cam_1", "cam_3", now, now + 90, 3, "cam_1->cam_2->cam_3"),
    ("MH12CD5678", "cam_1", "cam_2", now + 5, now + 50, 2, "cam_1->cam_2"),
]
for t in trajs:
    conn.execute(
        "INSERT INTO trajectories (plate, first_camera, last_camera, first_seen, last_seen, num_sightings, route) VALUES (?,?,?,?,?,?,?)",
        t,
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
print("[OK] Test data populated")
