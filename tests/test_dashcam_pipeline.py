"""Full pipeline test on real dashcam clips."""
import sys, os, json, time, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.schema import init_db

if os.path.exists("data/anpr.db"):
    os.remove("data/anpr.db")
init_db()

from src.pipeline_runner import PipelineRunner

cams = [
    ("cam_1", "dash_2.mp4"),
    ("cam_2", "dash_3.mp4"),
    ("cam_3", "dash_1.mp4"),
]

for cam_id, vid in cams:
    print(f"=== {cam_id} -> {vid} ===")
    runner = PipelineRunner(
        camera_id=cam_id,
        video_path=f"data/raw_videos/{vid}",
        camera_config={"gps_lat": 12.9758, "gps_lon": 77.6082},
        device="cpu",
        min_ocr_confidence=0.50,
        min_vehicle_height_px=60,
    )
    s = runner.run(speed_factor=0, max_frames=15)
    print(
        f'  frames={s["frames_processed"]} vehicles={s["vehicles_detected"]} '
        f'plates={s["plates_read"]} alerts={s["alerts_triggered"]}'
    )

print("\n=== DB CONTENT ===")
conn = sqlite3.connect("data/anpr.db")
rows = conn.execute(
    "SELECT plate, camera_id, plate_confidence FROM sightings ORDER BY timestamp"
).fetchall()
print(f"Sightings: {len(rows)}")
for r in rows[:25]:
    print(f"  {r[0]:20s} cam={r[1]} conf={r[2]:.2f}")
conn.close()
