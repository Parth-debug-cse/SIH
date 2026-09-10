"""Smoke test 5.8: API layer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.db.schema import init_db
import time, json, sqlite3

# Init fresh DB with test data
test_db = "tests/test_api.db"
if os.path.exists(test_db):
    os.remove(test_db)
init_db(test_db)

import src.db.schema as schema_mod
original_db = schema_mod.DB_PATH
schema_mod.DB_PATH = test_db

# Insert test data
conn = sqlite3.connect(test_db)
now = time.time()
conn.execute(
    """INSERT INTO sightings (plate, plate_confidence, camera_id, gps_lat, gps_lon, timestamp, frame_number)
       VALUES (?, ?, ?, ?, ?, ?, ?)""",
    ("KA01AB1234", 0.95, "cam_1", 12.9758, 77.6082, now, 1),
)
conn.execute(
    """INSERT INTO sightings (plate, plate_confidence, camera_id, gps_lat, gps_lon, timestamp, frame_number)
       VALUES (?, ?, ?, ?, ?, ?, ?)""",
    ("KA01AB1234", 0.88, "cam_2", 12.9768, 77.6050, now + 30, 50),
)
conn.execute(
    """INSERT INTO trajectories (plate, first_camera, last_camera, first_seen, last_seen, num_sightings, route)
       VALUES (?, ?, ?, ?, ?, ?, ?)""",
    ("KA01AB1234", "cam_1", "cam_2", now, now + 30, 2, "cam_1->cam_2"),
)
conn.execute(
    """INSERT INTO alerts (plate, alert_type, camera_id, gps_lat, gps_lon, timestamp, details)
       VALUES (?, ?, ?, ?, ?, ?, ?)""",
    ("KA01AB1234", "blacklist_exact", "cam_1", 12.9758, 77.6082, now, '{"matched_plate":"KA01AB1234"}'),
)
conn.execute(
    """INSERT INTO analytics (camera_id, timestamp, vehicle_count, avg_speed, density_level, congestion_flag)
       VALUES (?, ?, ?, ?, ?, ?)""",
    ("cam_1", now, 5, 25.0, "medium", 0),
)
conn.commit()
conn.close()
print("[OK] Test data inserted")

# Import FastAPI test client
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

# Test health
resp = client.get("/health")
assert resp.status_code == 200
data = resp.json()
assert data["status"] == "ok"
print(f"[OK] GET /health: {data['status']}")

# Test trajectory
resp = client.get("/trajectory/KA01AB1234")
assert resp.status_code == 200
data = resp.json()
assert data["plate"] == "KA01AB1234"
assert len(data["sightings"]) == 2
print(f"[OK] GET /trajectory/KA01AB1234: {len(data['sightings'])} sightings, route={data['route_summary']['first_camera']}->{data['route_summary']['last_camera']}")

# Test 404 for unknown plate
resp = client.get("/trajectory/UNKNOWN")
assert resp.status_code == 404
print("[OK] GET /trajectory/UNKNOWN: 404 as expected")

# Test analytics endpoints
resp = client.get("/analytics/density")
assert resp.status_code == 200
data = resp.json()
print(f"[OK] GET /analytics/density: {len(data['cameras'])} cameras")

resp = client.get("/analytics/congestion")
assert resp.status_code == 200
data = resp.json()
print(f"[OK] GET /analytics/congestion: {len(data['cameras'])} cameras")

resp = client.get("/analytics/od-patterns")
assert resp.status_code == 200
data = resp.json()
print(f"[OK] GET /analytics/od-patterns: {len(data['patterns'])} patterns")

# Test alerts
resp = client.get("/alerts")
assert resp.status_code == 200
data = resp.json()
assert len(data["alerts"]) >= 1
print(f"[OK] GET /alerts: {len(data['alerts'])} alerts")

# Test ingest
resp = client.post("/ingest", json={
    "plate": "TEST123",
    "camera_id": "cam_1",
    "gps_lat": 12.9758,
    "gps_lon": 77.6082,
    "timestamp": now + 100,
})
assert resp.status_code == 200
data = resp.json()
assert data["status"] == "ok"
print(f"[OK] POST /ingest: id={data['id']}")

# Test sightings
resp = client.get("/sightings?plate=KA01AB1234")
assert resp.status_code == 200
data = resp.json()
assert len(data["sightings"]) == 2
print(f"[OK] GET /sightings?plate=KA01AB1234: {len(data['sightings'])} sightings")

# Clean up
schema_mod.DB_PATH = original_db
os.remove(test_db)
print("[OK] Test database cleaned up")

print("SMOKE TEST 5.8 PASSED")
