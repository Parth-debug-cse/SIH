#!/usr/bin/env bash
# ANPR Pipeline Demo Launcher
# Usage: ./start_demo.sh [--keep-data] [--gpu] [--cpu] [--speed-factor N]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[X]${NC} $1"; }

cleanup() {
    log "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    log "All processes stopped."
}
trap cleanup EXIT INT TERM

PIDS=()

# Parse args
KEEP_DATA=""
DEVICE=""
SPEED=0.5
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-data) KEEP_DATA="--keep-data"; shift ;;
        --gpu) DEVICE="--gpu"; shift ;;
        --cpu) DEVICE="--cpu"; shift ;;
        --speed-factor) SPEED="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check prerequisites
echo "============================================"
echo "  ANPR City-Wide Trajectory Tracking"
echo "============================================"
echo

if [ ! -f "models/detection/yolov8n.pt" ]; then
    if [ -f "yolov8n.pt" ]; then
        mv yolov8n.pt models/detection/yolov8n.pt
        log "Moved yolov8n.pt to models/detection/"
    else
        fail "Vehicle model not found: models/detection/yolov8n.pt"
        exit 1
    fi
fi
log "Model: models/detection/yolov8n.pt"

VIDEO_COUNT=$(ls data/raw_videos/camera_*.mp4 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    fail "No camera videos found in data/raw_videos/"
    exit 1
fi
log "Videos: $VIDEO_COUNT camera feeds"

if [ ! -f "data/calibration/cameras.json" ]; then
    fail "Camera config not found: data/calibration/cameras.json"
    exit 1
fi
log "Config: data/calibration/cameras.json"

# Detect device
if [ "$DEVICE" = "--gpu" ]; then
    COMPUTE_DEVICE="cuda:0"
elif [ "$DEVICE" = "--cpu" ]; then
    COMPUTE_DEVICE="cpu"
else
    COMPUTE_DEVICE="cpu"
    if command -v python3 &>/dev/null; then
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            COMPUTE_DEVICE="cuda:0"
            log "GPU detected, using CUDA"
        fi
    fi
fi
log "Device: $COMPUTE_DEVICE"

# Init database
python3 -c "
import sys; sys.path.insert(0, '.')
from src.db.schema import init_db
init_db()
print('[OK] Database initialized')
" 2>/dev/null || python3 -c "
import sys; sys.path.insert(0, '.')
from src.db.schema import init_db
from pathlib import Path
db = Path('data/anpr.db')
if db.exists() and '$KEEP_DATA' == '--keep-data':
    print('[OK] Keeping existing database')
else:
    if db.exists(): db.unlink()
    init_db()
    print('[OK] Database initialized')
"

# Start backend
log "Starting backend on :8000..."
PYTHONPATH="$SCRIPT_DIR" python3 -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 &
PIDS+=($!)
BACKEND_PID=$!

# Wait for health
for i in $(seq 1 30); do
    sleep 1
    if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
        log "Backend ready on :8000"
        break
    fi
    if [ "$i" -eq 30 ]; then
        fail "Backend failed to start in 30s"
        exit 1
    fi
done

# Start shared cross-camera fusion worker FIRST so that camera workers only
# ever write sightings and a single fusion service turns them into
# trajectories (no per-worker fusion state).
log "Starting shared fusion worker..."
PYTHONPATH="$SCRIPT_DIR" python3 -m src.fusion.worker --interval 3 --log-level INFO &
PIDS+=($!)

# Start camera workers
for video in data/raw_videos/camera_*.mp4; do
    BASENAME=$(basename "$video")
    # Extract camera ID from cameras.json
    CAM_ID=$(python3 -c "
import json
with open('data/calibration/cameras.json') as f:
    cfg = json.load(f)
for c in cfg['cameras']:
    if c['video_file'] == '$BASENAME':
        print(c['camera_id'])
        break
else:
    print('cam_unknown')
" 2>/dev/null)

    GPS_LAT=$(python3 -c "
import json
with open('data/calibration/cameras.json') as f:
    cfg = json.load(f)
for c in cfg['cameras']:
    if c['video_file'] == '$BASENAME':
        print(c.get('gps_lat', 0))
        break
else:
    print(0)
" 2>/dev/null)

    GPS_LON=$(python3 -c "
import json
with open('data/calibration/cameras.json') as f:
    cfg = json.load(f)
for c in cfg['cameras']:
    if c['video_file'] == '$BASENAME':
        print(c.get('gps_lon', 0))
        break
else:
    print(0)
" 2>/dev/null)

    log "Starting worker: $CAM_ID -> $BASENAME"
    PYTHONPATH="$SCRIPT_DIR" python3 -m src.pipeline_runner \
        --camera-id "$CAM_ID" \
        --video "$video" \
        --gps-lat "$GPS_LAT" \
        --gps-lon "$GPS_LON" \
        --device "$COMPUTE_DEVICE" \
        --speed-factor "$SPEED" &
    PIDS+=($!)
done

echo
echo "============================================"
echo "  ANPR Pipeline is running!"
echo "============================================"
echo
echo "  Health:       curl http://127.0.0.1:8000/health"
echo "  Trajectory:   curl http://127.0.0.1:8000/trajectory/KA01AB1234"
echo "  Density:      curl http://127.0.0.1:8000/analytics/density"
echo "  Congestion:   curl http://127.0.0.1:8000/analytics/congestion"
echo "  Alerts:       curl http://127.0.0.1:8000/alerts"
echo "  Sightings:    curl http://127.0.0.1:8000/sightings"
echo
echo "  Press Ctrl+C to stop."
echo "============================================"

wait
