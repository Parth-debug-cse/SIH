"""Full E2E pipeline test on REAL traffic footage (3 cameras)."""
import sys, os, json, time, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.schema import init_db

# Fresh DB
db_path = "data/anpr.db"
if os.path.exists(db_path):
    os.remove(db_path)
init_db(db_path)

from src.pipeline_runner import PipelineRunner

with open("data/calibration/cameras.json") as f:
    config = json.load(f)

results = {}
for cam in config["cameras"]:
    print(f"\n{'='*50}")
    print(f"Processing {cam['camera_id']} -> {cam['video_file']}")
    print(f"{'='*50}")
    runner = PipelineRunner(
        camera_id=cam["camera_id"],
        video_path="data/raw_videos/" + cam["video_file"],
        camera_config={"gps_lat": cam["gps_lat"], "gps_lon": cam["gps_lon"]},
        device="cpu",
    )
    t0 = time.time()
    summary = runner.run(speed_factor=0, max_frames=10)
    elapsed = time.time() - t0
    results[cam["camera_id"]] = summary
    print(f"  Time: {elapsed:.1f}s ({(elapsed/10):.1f}s/frame on CPU)")
    print(f"  Vehicles: {summary['vehicles_detected']}, Plates read: {summary['plates_read']}, Alerts: {summary['alerts_triggered']}")

print(f"\n{'='*60}")
print("E2E SUMMARY ON REAL FOOTAGE")
print(f"{'='*60}")

conn = sqlite3.connect(db_path)
sightings = conn.execute("SELECT plate, camera_id, gps_lat, gps_lon, timestamp, plate_confidence FROM sightings ORDER BY timestamp").fetchall()
print(f"\nSightings in DB: {len(sightings)}")
for s in sightings[:20]:
    print(f"  {s[0]}: cam={s[1]} lat={s[2]:.4f} lon={s[3]:.4f} conf={s[5]:.2f}")

alerts = conn.execute("SELECT plate, alert_type FROM alerts").fetchall()
print(f"\nAlerts in DB: {len(alerts)}")
for a in alerts[:10]:
    print(f"  {a[0]} ({a[1]})")

analytics = conn.execute("SELECT camera_id, vehicle_count, congestion_flag FROM analytics").fetchall()
print(f"\nAnalytics rows: {len(analytics)}")
for a in analytics[:10]:
    print(f"  {a[0]}: vehicles={a[1]} congested={a[2]}")
conn.close()

# Check if any plates were actually read
total_plates = sum(r["plates_read"] for r in results.values())
print(f"\nTOTAL plates read across all cameras: {total_plates}")
if total_plates == 0:
    print("WARN: No plates read yet. OCR on distant highway plates on CPU is challenging.")
else:
    print("OK: Pipeline read real plates end-to-end!")