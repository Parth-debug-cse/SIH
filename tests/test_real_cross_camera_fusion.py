"""Real footage cross-camera fusion integration test (SIH26127 demo path).

Requires two real video clips of the **same vehicle** filmed at two
different positions/times.  Each clip is processed by its own
``src.pipeline_runner`` instance (cam_1 and cam_2) into a fresh shared
SQLite DB, the shared fusion engine is triggered through the real
``POST /fusion/run`` API, and the fused trajectory is fetched with
``GET /trajectory/{plate}`` and asserted to span both cameras
chronologically.

Clip discovery (in priority order):
    1. CLI argument:  python tests/test_real_cross_camera_fusion.py CAM1 CLIP1 CLIP2
    2. Environment:   ANPR_CAM1_VIDEO / ANPR_CAM2_VIDEO
    3. Default names: data/raw_videos/cam_1.mp4 and data/raw_videos/cam_2.mp4

A wall-clock ``GAP_SECONDS`` (default 25, matching the ~20-30 s filming
discipline between the two positions) is inserted between the runs so the
cam_1 -> cam_2 transition is physically plausible at the calibrated
~0.36 km camera separation (~50 km/h implied).

Exit code 0 only when a genuine multi-camera trajectory is produced.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Calibrated camera positions (mirrors data/calibration/cameras.json).
CAM1_GPS = (12.9758, 77.6082)
CAM2_GPS = (12.9768, 77.6050)

GAP_SECONDS = float(os.environ.get("ANPR_FUSION_GAP_SECONDS", "25"))


def resolve_clips(argv) -> tuple[Path, Path]:
    """Resolve the two real clips, failing loudly when unavailable."""
    if len(argv) >= 3:
        return Path(argv[1]), Path(argv[2])
    env1 = os.environ.get("ANPR_CAM1_VIDEO")
    env2 = os.environ.get("ANPR_CAM2_VIDEO")
    if env1 and env2:
        return Path(env1), Path(env2)
    default1 = ROOT / "data" / "raw_videos" / "cam_1.mp4"
    default2 = ROOT / "data" / "raw_videos" / "cam_2.mp4"
    if default1.exists() and default2.exists():
        return default1, default2
    raise SystemExit(
        "REAL CROSS-CAMERA FUSION TEST – clips missing.  Provide two real "
        "clips of the same vehicle (two positions, ~20-30 s apart) as\n"
        "  python tests/test_real_cross_camera_fusion.py <cam1_clip> <cam2_clip>\n"
        "or via ANPR_CAM1_VIDEO / ANPR_CAM2_VIDEO.\n"
        "This part cannot be synthesised from code - it needs the second "
        "filmed position."
    )


def fresh_db() -> Path:
    """Reset the shared test database and point schema helpers at it."""
    db = ROOT / "tests" / "test_cross_camera_fusion.db"
    if db.exists():
        db.unlink()
    import src.db.schema as schema_mod
    from src.db.schema import init_db
    schema_mod.DB_PATH = db
    init_db(db)
    return db


def run_camera(camera_id: str, clip: Path, gps_lat: float, gps_lon: float) -> dict:
    """Run src.pipeline_runner for one camera over its full clip."""
    from src.pipeline_runner import PipelineRunner

    runner = PipelineRunner(
        camera_id=camera_id,
        video_path=clip,
        camera_config={"gps_lat": gps_lat, "gps_lon": gps_lon},
    )
    summary = runner.run(speed_factor=0.0, run_fusion=False)
    return {
        "summary": summary,
        "diag": summary["plate_diagnostics"],
    }


def load_sightings() -> list[dict]:
    """Read all sightings currently in the shared DB."""
    rows = sqlite3.connect(str(ROOT / "tests" / "test_cross_camera_fusion.db")).execute(
        "SELECT plate, plate_confidence, camera_id, timestamp, track_id "
        "FROM sightings"
    ).fetchall()
    return [
        {
            "plate": r[0].strip().upper(),
            "confidence": r[1],
            "camera_id": r[2],
            "timestamp": r[3],
            "track_id": r[4],
        }
        for r in rows
    ]


def plates_on_both_cameras(sightings: list[dict]) -> list[str]:
    """Plates (case-normalised) that were read on BOTH cam_1 and cam_2."""
    by_plate: dict[str, set[str]] = {}
    for s in sightings:
        by_plate.setdefault(s["plate"], set()).add(s["camera_id"])
    both = [
        plate for plate, cams in by_plate.items()
        if {"cam_1", "cam_2"} <= cams
    ]
    both.sort(key=lambda p: sum(1 for s in sightings if s["plate"] == p), reverse=True)
    return both


def api_client():
    from fastapi.testclient import TestClient
    from src.api.app import app
    return TestClient(app)


def print_fused_trajectory(payload: dict) -> None:
    """Pretty-print a /trajectory/{plate} response for the demo."""
    print("\n===== FUSED CROSS-CAMERA TRAJECTORY =====")
    print(f"plate               : {payload['plate']}")
    print(f"trajectory          : {payload['trajectory_code']} (id={payload['trajectory_id']})")
    print(f"confidence          : {payload['trajectory_confidence']}")
    rs = payload["route_summary"]
    print(f"route               : {rs['route']}")
    print(f"first -> last cam   : {rs['first_camera']} -> {rs['last_camera']}")
    print(f"num sightings       : {rs['num_sightings']}")
    print(f"duration (s)        : {rs['duration_seconds']}")
    print(f"distance (km)       : {rs['total_distance_km']}")
    print("sightings (chronological):")
    for i, s in enumerate(payload["sightings"]):
        print(
            f"  {i + 1:2d}. {s['camera_id']:<8s} "
            f"conf={s['confidence']:.2f} "
            f"ts={s['timestamp']:.1f} "
            f"dir={s['direction'] or 'n/a'} "
        )
    print("=========================================")


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    clip1, clip2 = resolve_clips(argv)
    if not (clip1.exists() and clip2.exists()):
        raise SystemExit(f"Clip not found: {clip1 if not clip1.exists() else clip2}")

    print(f"[SETUP] clip cam_1: {clip1}")
    print(f"[SETUP] clip cam_2: {clip2}")
    fresh_db()

    # 1. Run the two cameras independently (each into the shared DB).
    run1 = run_camera("cam_1", clip1, *CAM1_GPS)
    print(f"[RUN] cam_1 complete: {run1['summary']['plates_read']} plates read")
    time.sleep(GAP_SECONDS)  # reproduce the ~20-30 s filming gap between positions
    run2 = run_camera("cam_2", clip2, *CAM2_GPS)
    print(f"[RUN] cam_2 complete: {run2['summary']['plates_read']} plates read")

    # 2. Sightings BEFORE vs AFTER the format-plausibility gate.
    d1, d2 = run1["diag"], run2["diag"]
    before = d1["plate_confidence_gate_cleared"] + d2["plate_confidence_gate_cleared"]
    format_rejected = d1["plate_format_check_rejected"] + d2["plate_format_check_rejected"]
    after = d1["plate_both_gates_cleared"] + d2["plate_both_gates_cleared"]
    samples = list(dict.fromkeys(d1["plate_format_rejected_samples"] + d2["plate_format_rejected_samples"]))
    print(f"\n[GATE] sightings that cleared the CONFIDENCE gate only (would have "
          f"been written before): {before}")
    print(f"[GATE] ...then REJECTED by the format gate (false positives removed): {format_rejected}")
    print(f"[GATE] sightings stored (both gates): {after}")
    if samples:
        print(f"[GATE] example format-rejected reads: {', '.join(repr(s) for s in samples[:12])}")

    # 3. Trigger the SHARED fusion engine through the real API.
    client = api_client()
    resp = client.post("/fusion/run")
    assert resp.status_code == 200, f"/fusion/run failed: {resp.text}"
    fusion_stats = resp.json()
    print(f"\n[FUSION] {fusion_stats['num_sightings']} sightings -> "
          f"{fusion_stats['num_trajectories']} trajectories")

    # 4. Find a plate read on BOTH cameras and fetch its fused trajectory.
    sightings = load_sightings()
    candidates = plates_on_both_cameras(sightings)
    print(f"[CANDIDATES] plates read on both cam_1 and cam_2: {candidates or 'NONE'}")

    payload = None
    chosen = None
    for plate in candidates:
        resp = client.get(f"/trajectory/{plate}")
        if resp.status_code != 200:
            print(f"[TRAJ] plate {plate!r}: /trajectory returned {resp.status_code}; skipping")
            continue
        body = resp.json()
        route = [s["camera_id"] for s in body["sightings"]]
        if "cam_1" in route and "cam_2" in route:
            payload, chosen = body, plate
            break

    if payload is None:
        print("\n[CROSS-CAMERA] FAIL: no fused trajectory spanning both cameras was found.",
              file=sys.stderr)
        return 1

    # 5. Assert the returned route includes BOTH cameras in chronological order.
    route = [s["camera_id"] for s in payload["sightings"]]
    first_cam, last_cam = route[0], route[-1]
    timestamps = [s["timestamp"] for s in payload["sightings"]]
    assert first_cam == "cam_1" and last_cam == "cam_2", \
        f"route not cam_1..cam_2: {route}"
    assert timestamps == sorted(timestamps), "sightings not chronological"
    assert set(route) >= {"cam_1", "cam_2"}, f"route missing a camera: {route}"
    assert len(route) >= 2

    print_fused_trajectory(payload)
    print(f"\nREAL CROSS-CAMERA FUSION TEST PASSED — plate {chosen!r} fused "
          f"across cam_1 -> cam_2 on real footage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())