"""Run full pipeline on the user's real Bengaluru WhatsApp video."""
import sys, os, json, time, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.schema import init_db, get_connection

db_path = "data/anpr.db"
if os.path.exists(db_path):
    os.remove(db_path)
init_db(db_path)

from src.pipeline_runner import PipelineRunner

PATH = "data/raw_videos/WhatsApp Video 2026-09-11 at 1.35.19 PM.mp4"
runner = PipelineRunner(
    camera_id="cam_wa",
    video_path=PATH,
    camera_config={"gps_lat": 12.9758, "gps_lon": 77.6082, "pixel_to_meter_ratio": 0.05, "compass_bearing": 45},
    device="cpu",
    min_ocr_confidence=0.50,
    min_vehicle_height_px=60,
)
t0 = time.time()
summary = runner.run(speed_factor=0)
elapsed = time.time() - t0
print(f"\nRun time: {elapsed:.1f}s for {summary['frames_processed']} frames")

print("\n--- diagnostic snapshot ---")
diag = summary.get("plate_diagnostics", {})
for k in ("vehicles_detected", "vehicle_too_small", "vehicles_with_plate_box",
          "no_plate_box_found", "plate_found_below_gate",
          "plate_confidence_gate_cleared", "plate_format_check_rejected",
          "plate_both_gates_cleared", "plate_aspect_ratio_rejected",
          "plate_crop_heights_px", "plate_crop_widths_px",
          "plate_format_rejected_samples"):
    if k in diag:
        print(f"  {k}: {diag[k]}")

conn = sqlite3.connect(db_path)
rows = conn.execute(
    "SELECT plate, camera_id, plate_confidence, frame_number FROM sightings ORDER BY timestamp"
).fetchall()
print(f"\nSIGHTINGS IN DB: {len(rows)}")
for r in rows:
    print(f"  plate={r[0]} cam={r[1]} conf={r[2]:.3f} frame={r[3]}")
tr = conn.execute("SELECT count(*) FROM trajectories").fetchone()[0]
print(f"TRAJECTORIES: {tr}")
conn.close()