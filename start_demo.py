#!/usr/bin/env python3
"""
ANPR Pipeline Demo Launcher
Single command to bring up the entire system: backend + camera workers.
Usage: python start_demo.py [--keep-data] [--gpu] [--cpu] [--speed-factor N]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"

_child_processes = []


def log(tag, msg, status="OK"):
    symbols = {"OK": "[OK]", "WARN": "[!]", "FAIL": "[X]", "INFO": "[i]"}
    sym = symbols.get(status, "[-]")
    print(f"{sym} {tag}: {msg}")


def check_prerequisites():
    """Verify models, datasets, and video files exist."""
    issues = []

    model_path = PROJECT_ROOT / "models" / "detection" / "yolov8n.pt"
    if not model_path.exists():
        # Try project root (ultralytics default download location)
        model_root = PROJECT_ROOT / "yolov8n.pt"
        if model_root.exists():
            import shutil
            shutil.move(str(model_root), str(model_path))
            log("Prereq", f"Moved yolov8n.pt to {model_path}")
        else:
            issues.append(f"Vehicle model not found: {model_path}")

    videos_dir = PROJECT_ROOT / "data" / "raw_videos"
    video_files = sorted(videos_dir.glob("camera_*.mp4"))
    if not video_files:
        issues.append(f"No camera videos found in {videos_dir}")

    cameras_config = PROJECT_ROOT / "data" / "calibration" / "cameras.json"
    if not cameras_config.exists():
        issues.append(f"Camera config not found: {cameras_config}")

    if issues:
        for issue in issues:
            log("Prereq", issue, "FAIL")
        return False, []

    log("Prereq", f"Model: {model_path}")
    log("Prereq", f"Videos: {len(video_files)} files")
    for v in video_files:
        log("Prereq", f"  {v.name}")
    log("Prereq", f"Config: {cameras_config}")
    return True, video_files


def init_database(keep_data=False):
    """Initialize or reset the SQLite database."""
    db_path = PROJECT_ROOT / "data" / "anpr.db"
    if keep_data and db_path.exists():
        log("DB", f"Keeping existing database at {db_path}")
        return

    if db_path.exists():
        db_path.unlink()
        log("DB", "Removed old database")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.db.schema import init_db
    init_db(str(db_path))
    log("DB", f"Initialized at {db_path}")


def detect_device(requested):
    """Auto-detect GPU/CPU."""
    if requested == "gpu":
        return "cuda:0"
    if requested == "cpu":
        return "cpu"
    # auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            log("Device", "CUDA GPU detected", "OK")
            return "cuda:0"
    except ImportError:
        pass
    log("Device", "Using CPU (reduced speed)", "INFO")
    return "cpu"


def start_backend(device):
    """Start FastAPI backend and wait for health check."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        str(VENV_PYTHON), "-m", "uvicorn",
        "src.api.app:app",
        "--host", BACKEND_HOST,
        "--port", str(BACKEND_PORT),
    ]

    log("Backend", f"Starting uvicorn on {BACKEND_HOST}:{BACKEND_PORT}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _child_processes.append(proc)

    # Wait for health check
    for attempt in range(30):
        time.sleep(1)
        try:
            req = urllib.request.Request(HEALTH_URL)
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                log("Backend", f"Ready on :{BACKEND_PORT}", "OK")
                return proc
        except (urllib.error.URLError, OSError):
            pass

    log("Backend", "Failed to start within 30s", "FAIL")
    cleanup()
    sys.exit(1)


def start_fusion_worker():
    """Start the shared cross-camera fusion worker process.

    The fusion worker periodically consumes sightings from the shared
    database and persists unified trajectories.  Camera workers do NOT keep
    their own fusion state - this is the single logical fusion service.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        str(VENV_PYTHON), "-m", "src.fusion.worker",
        "--interval", "3",
        "--log-level", "INFO",
    ]

    log("Fusion", "Starting shared cross-camera fusion worker")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _child_processes.append(proc)
    return proc


def validate_camera_configs():
    """Validate data/calibration/cameras.json; warn on missing calibration."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from src.calibration import load_cameras_config, validate_cameras_config
        cameras = load_cameras_config()
        issues = validate_cameras_config(cameras)
        errors = [m for m in issues if m.startswith("[ERROR]")]
        warnings = [m for m in issues if m.startswith("[WARN]")]
        for w in warnings:
            log("Calibration", w.replace("[WARN] ", ""), "WARN")
        for e in errors:
            log("Calibration", e.replace("[ERROR] ", ""), "FAIL")
        return not errors
    except (FileNotFoundError, ValueError) as exc:
        log("Calibration", f"Config cannot be read: {exc}", "FAIL")
        return False


def start_camera_workers(video_files, device, speed_factor):
    """Start one pipeline runner per camera video."""
    sys.path.insert(0, str(PROJECT_ROOT))

    # Load camera configs
    config_path = PROJECT_ROOT / "data" / "calibration" / "cameras.json"
    with open(config_path) as f:
        config = json.load(f)

    cameras = {c["video_file"]: c for c in config.get("cameras", [])}

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    workers = []
    for vf in video_files:
        cam_config = cameras.get(vf.name, {})
        camera_id = cam_config.get("camera_id", f"cam_{len(workers)+1}")
        gps_lat = cam_config.get("gps_lat", 0.0)
        gps_lon = cam_config.get("gps_lon", 0.0)

        cmd = [
            str(VENV_PYTHON), "-m", "src.pipeline_runner",
            "--camera-id", camera_id,
            "--video", str(vf),
            "--gps-lat", str(gps_lat),
            "--gps-lon", str(gps_lon),
            "--device", device,
            "--speed-factor", str(speed_factor),
            "--log-level", "INFO",
        ]

        log("Worker", f"Starting {camera_id} -> {vf.name}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _child_processes.append(proc)
        workers.append((camera_id, proc))

    return workers


def cleanup(*args):
    """Stop all child processes cleanly."""
    log("Cleanup", "Stopping all processes...")
    for proc in _child_processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _child_processes.clear()
    log("Cleanup", "All processes stopped")


def print_example_commands():
    """Print curl commands the user can try immediately."""
    base = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
    print()
    print("=" * 60)
    print("  ANPR Pipeline is running!")
    print("=" * 60)
    print()
    print(f"  Health:       curl {base}/health")
    print(f"  Trajectory:   curl {base}/trajectory/KA01AB1234")
    print(f"  Density:      curl {base}/analytics/density")
    print(f"  Congestion:   curl {base}/analytics/congestion")
    print(f"  OD Patterns:  curl {base}/analytics/od-patterns")
    print(f"  Alerts:       curl {base}/alerts")
    print(f"  Sightings:    curl {base}/sightings")
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="ANPR Pipeline Demo Launcher")
    parser.add_argument("--keep-data", action="store_true", help="Keep existing database data")
    parser.add_argument("--gpu", action="store_true", help="Force GPU mode")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--speed-factor", type=float, default=0.5, help="Video playback speed (default: 0.5)")
    args = parser.parse_args()

    device_mode = "gpu" if args.gpu else ("cpu" if args.cpu else "auto")

    print()
    print("=" * 60)
    print("  ANPR City-Wide Trajectory Tracking Pipeline")
    print("=" * 60)
    print()

    # 1. Check prerequisites
    log("Startup", "Checking prerequisites...")
    ok, video_files = check_prerequisites()
    if not ok:
        log("Startup", "Missing prerequisites. Cannot continue.", "FAIL")
        sys.exit(1)

    # 2. Detect device
    device = detect_device(device_mode)

    # 3. Init database
    init_database(keep_data=args.keep_data)

    # 4. Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 5. Start backend
    backend_proc = start_backend(device)

    # 6. Validate camera calibration
    log("Startup", "Validating camera calibration...")
    cal_ok = validate_camera_configs()
    if not cal_ok:
        log("Startup", "Camera calibration has errors. Cannot continue.", "FAIL")
        cleanup()
        sys.exit(1)

    # 7. Start shared cross-camera fusion worker
    fusion_proc = start_fusion_worker()

    # 8. Start camera workers
    workers = start_camera_workers(video_files, device, args.speed_factor)
    for camera_id, proc in workers:
        log("Worker", f"{camera_id} running (pid={proc.pid})", "OK")

    # 9. Print example commands
    print_example_commands()

    # 8. Wait for any process to exit
    try:
        while True:
            # Check if backend is still alive
            if backend_proc.poll() is not None:
                log("Backend", "Backend process exited unexpectedly", "FAIL")
                break
            # Check workers
            for camera_id, proc in workers:
                if proc.poll() is not None:
                    log("Worker", f"{camera_id} exited (code={proc.returncode})", "WARN")
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
