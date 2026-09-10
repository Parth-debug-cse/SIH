"""Smoke test 5.5: Trajectory database."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.db.schema import init_db, get_connection, DB_PATH
import time, json

# Initialize fresh database
test_db = "tests/test_anpr.db"
if os.path.exists(test_db):
    os.remove(test_db)

init_db(test_db)
print(f"[OK] Database initialized at {test_db}")

# Insert test sightings
conn = get_connection(test_db)
now = time.time()

test_sightings = [
    ("KA01AB1234", 0.95, "cam_1", 12.9758, 77.6082, now, "car", None, None, 1),
    ("KA01AB1234", 0.88, "cam_2", 12.9768, 77.6050, now + 30, "car", None, None, 50),
    ("KA01AB1234", 0.91, "cam_3", 12.9745, 77.6095, now + 60, "car", None, None, 100),
]

for s in test_sightings:
    conn.execute(
        """INSERT INTO sightings
           (plate, plate_confidence, camera_id, gps_lat, gps_lon, timestamp, vehicle_class, vehicle_bbox, plate_bbox, frame_number)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        s,
    )
conn.commit()
print(f"[OK] Inserted {len(test_sightings)} test sightings")

# Query for specific plate
rows = conn.execute("SELECT * FROM sightings WHERE plate = ?", ("KA01AB1234",)).fetchall()
print(f"[OK] Query returned {len(rows)} rows for plate KA01AB1234")
assert len(rows) == 3, f"FAIL: Expected 3 rows, got {len(rows)}"

# Verify ordering
cameras = [row[3] for row in rows]  # camera_id column
print(f"[OK] Camera route: {' -> '.join(cameras)}")
assert cameras == ["cam_1", "cam_2", "cam_3"], f"FAIL: Route wrong: {cameras}"

# Insert trajectory
conn.execute(
    """INSERT INTO trajectories (plate, first_camera, last_camera, first_seen, last_seen, num_sightings, route)
       VALUES (?, ?, ?, ?, ?, ?, ?)""",
    ("KA01AB1234", "cam_1", "cam_3", now, now + 60, 3, "cam_1->cam_2->cam_3"),
)
conn.commit()

# Query trajectory
traj = conn.execute("SELECT * FROM trajectories WHERE plate = ?", ("KA01AB1234",)).fetchone()
print(f"[OK] Trajectory: plate={traj[1]} from {traj[2]} to {traj[3]}, num_sightings={traj[6]}")
assert traj[1] == "KA01AB1234"
assert traj[2] == "cam_1"
assert traj[3] == "cam_3"
assert traj[6] == 3

conn.close()
os.remove(test_db)
print("[OK] Test database cleaned up")

print("SMOKE TEST 5.5 PASSED")
