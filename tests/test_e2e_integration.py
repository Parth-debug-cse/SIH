"""End-to-end integration test: multi-camera sightings -> shared fusion -> API.

Runs exactly the code paths of the live demo:

1. Three "camera workers" write sightings to the shared SQLite database
   (same columns the pipeline persists: plate_confidence + track_id).
2. The shared fusion engine (``src.fusion.engine.run_fusion_once``) consumes
   those sightings and persists unified trajectories.
3. The FastAPI layer serves the fused trajectory from the DB.

No video / ML inference is involved: sightings are seeded exactly as the
camera workers persist them.
"""
import sys, os, time, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.db.schema import init_db
import src.db.schema as schema_mod

TEST_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_e2e.db")


def wipe(db_path):
    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)


def seed_sighting(conn, plate, camera_id, gps_lat, gps_lon, ts, conf, track, cls="car"):
    conn.execute(
        "INSERT INTO sightings (plate, plate_confidence, camera_id, gps_lat, gps_lon, "
        " timestamp, vehicle_class, track_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (plate, conf, camera_id, gps_lat, gps_lon, ts, cls, track),
    )


wipe(TEST_DB)
init_db(TEST_DB)
original_db = schema_mod.DB_PATH
schema_mod.DB_PATH = TEST_DB

now = time.time()
conn = sqlite3.connect(TEST_DB)
# Three camera workers observe the same vehicle in temporal order.
seed_sighting(conn, "KA01AB1234", "cam_1", 12.9758, 77.6082, now, 0.95, 11)
seed_sighting(conn, "KA01AB1234", "cam_2", 12.9768, 77.6050, now + 60, 0.88, 21)
seed_sighting(conn, "KA01AB1234", "cam_3", 12.9745, 77.6095, now + 180, 0.91, 31)
# A blacklisted plate observed once.
seed_sighting(conn, "KA01XX9999", "cam_3", 12.9745, 77.6095, now + 42, 0.85, 32)
conn.commit()
conn.close()
print("[OK] Seeded 4 sightings (3-camera vehicle + 1 blacklisted plate)")

# ---------------------------------------------------------------------------
# Shared fusion engine consumes sightings and persists trajectories
# ---------------------------------------------------------------------------
from src.fusion.engine import run_fusion_once

stats = run_fusion_once(db_path=TEST_DB)
assert stats["num_sightings"] == 4, stats
assert stats["num_trajectories"] >= 2, stats
assert "KA01AB1234" in stats["trajectories"], stats
print("[OK] Fusion: %d sightings -> %d trajectories: %s" % (
    stats["num_sightings"], stats["num_trajectories"], stats["trajectories"]))

conn = sqlite3.connect(TEST_DB)
conn.row_factory = sqlite3.Row
row = conn.execute(
    "SELECT trajectory_code, route, num_sightings, first_camera, last_camera, "
    "       total_distance_km, total_duration_seconds, trajectory_confidence "
    "FROM trajectories WHERE plate = 'KA01AB1234' ORDER BY id DESC LIMIT 1"
).fetchone()
assert row is not None
assert row["route"] == "cam_1->cam_2->cam_3", dict(row)
assert row["num_sightings"] == 3
assert row["total_duration_seconds"] == 180.0
assert row["trajectory_confidence"] >= 0.55, dict(row)
children = conn.execute(
    "SELECT camera_id FROM trajectory_sightings WHERE trajectory_id = "
    "(SELECT id FROM trajectories WHERE plate = 'KA01AB1234' ORDER BY id DESC LIMIT 1) "
    "ORDER BY seq"
).fetchall()
assert [c["camera_id"] for c in children] == ["cam_1", "cam_2", "cam_3"]
conn.close()
print("[OK] Trajectory persisted: cam_1->cam_2->cam_3 (3 ordered sightings, dur=180s)")

# ---------------------------------------------------------------------------
# API serves the fused trajectory
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

resp = client.get("/trajectory/KA01AB1234")
assert resp.status_code == 200
data = resp.json()
assert data["trajectory_code"].startswith("TRJ-")
assert data["route_summary"]["route"] == "cam_1->cam_2->cam_3"
assert data["sighting_source"] == "trajectory_sightings"
assert [s["camera_id"] for s in data["sightings"]] == ["cam_1", "cam_2", "cam_3"]
assert data["total_distance_km"] > 0
print("[OK] GET /trajectory/KA01AB1234 -> %s (%d sightings, source=%s)" % (
    data["route_summary"]["route"], len(data["sightings"]), data["sighting_source"]))

resp = client.get("/trajectory/KA01XX9999")
assert resp.status_code == 200
print("[OK] GET /trajectory/KA01XX9999 (single-camera sighting served)")

resp = client.get("/trajectory/NOPE0000")
assert resp.status_code == 404
print("[OK] GET /trajectory/NOPE0000 -> 404 (no fused trajectory)")

# ---------------------------------------------------------------------------
# O-D patterns come from real fused trajectories
# ---------------------------------------------------------------------------
resp = client.get("/analytics/od-patterns")
assert resp.status_code == 200
patterns = resp.json()["patterns"]
assert any(
    p["origin_camera"] == "cam_1" and p["dest_camera"] == "cam_3"
    and "KA01AB1234" in p["sample_plates"]
    for p in patterns
), patterns
print("[OK] GET /analytics/od-patterns: O-D cam_1->cam_3 present from fused trajectories")

# ---------------------------------------------------------------------------
# Raw sightings + alerts still served
# ---------------------------------------------------------------------------
resp = client.get("/sightings")
assert resp.status_code == 200 and len(resp.json()["sightings"]) == 4
print("[OK] GET /sightings: 4 raw rows")

conn = sqlite3.connect(TEST_DB)
conn.execute(
    "INSERT INTO alerts (plate, alert_type, camera_id, gps_lat, gps_lon, timestamp, details) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)",
    ("KA01XX9999", "blacklist_exact", "cam_3", 12.9745, 77.6095, now, '{"matched":"KA01XX9999"}'),
)
conn.commit()
conn.close()
resp = client.get("/alerts")
assert resp.status_code == 200 and len(resp.json()["alerts"]) >= 1
print("[OK] GET /alerts: blacklist alert served")

# ---------------------------------------------------------------------------
# /fusion/run is idempotent over the shared DB
# ---------------------------------------------------------------------------
resp = client.post("/fusion/run")
assert resp.status_code == 200
assert resp.json()["num_trajectories"] >= 2
print("[OK] POST /fusion/run: idempotent re-run over shared DB")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
schema_mod.DB_PATH = original_db
wipe(TEST_DB)
print("[OK] Test database cleaned up")

print("\nE2E INTEGRATION TEST PASSED")