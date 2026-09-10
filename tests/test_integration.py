"""Full integration test: pipeline runner + DB verification."""
import sys, os, json, time, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.schema import init_db
init_db()
from src.pipeline_runner import PipelineRunner

with open("data/calibration/cameras.json") as f:
    config = json.load(f)

cam = config["cameras"][0]
runner = PipelineRunner(
    camera_id=cam["camera_id"],
    video_path="data/raw_videos/" + cam["video_file"],
    camera_config={"gps_lat": cam["gps_lat"], "gps_lon": cam["gps_lon"]},
    device="cpu"
)
summary = runner.run(speed_factor=0, max_frames=30)

conn = sqlite3.connect("data/anpr.db")
sightings = conn.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
analytics_rows = conn.execute("SELECT COUNT(*) FROM analytics").fetchone()[0]
conn.close()

print()
print("=== INTEGRATION RESULT ===")
print("Frames processed:", summary["frames_processed"])
print("Vehicles detected:", summary["vehicles_detected"])
print("Plates read:", summary["plates_read"])
print("Alerts triggered:", summary["alerts_triggered"])
print("DB sightings rows:", sightings)
print("DB analytics rows:", analytics_rows)
print("=== ALL STAGES EXECUTED ===")
