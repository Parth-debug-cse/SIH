"""Print the live results of the last pipeline run from data/anpr.db.

Shows exactly: vehicles seen, plates read, and the plate numbers themselves.
"""

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "anpr.db"

conn = sqlite3.connect(str(DB))
cur = conn.cursor()

vehicles = cur.execute(
    "SELECT camera_id, vehicle_count FROM analytics ORDER BY id DESC LIMIT 100"
).fetchall()
plates = cur.execute(
    "SELECT plate, plate_confidence, camera_id, timestamp, vehicle_class "
    "FROM sightings ORDER BY timestamp"
).fetchall()

print("============================================")
print("            LIVE RESULTS")
print("============================================")

if vehicles:
    per_cam = {}
    for cam, count in vehicles:
        per_cam[cam] = max(per_cam.get(cam, 0), count)
    total = sum(per_cam.values())
    print(f"  Vehicles detected  : {total}")
    for cam in sorted(per_cam):
        print(f"    - {cam}: {per_cam[cam]}")
else:
    print("  Vehicles detected  : not recorded yet")

print(f"  Plates read        : {len(plates)}")
print("  Plate numbers      :")
if plates:
    seen = []
    for plate, conf, cam, ts, vclass in plates:
        if plate not in seen:
            seen.append(plate)
            print(f"    - {plate}  (conf {conf:.2f}, {cam})")
else:
    print("    (none)")

print("============================================")
conn.close()