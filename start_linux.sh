#!/usr/bin/env bash
# ANPR Pipeline Go-Live starter (Linux).
# Runs backend + shared fusion worker + one pipeline runner per video in
# data/raw_videos/, then prints: vehicles detected, plates read, plate numbers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[X]${NC} $1"; }

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- python ---------------------------------------------------------------
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    fail "No python found. Create your venv: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# --- model ----------------------------------------------------------------
if [ ! -f "models/detection/yolov8n.pt" ]; then
    if [ -f "yolov8n.pt" ]; then mv yolov8n.pt models/detection/yolov8n.pt; else
        fail "model missing: models/detection/yolov8n.pt"; exit 1
    fi
fi

# --- videos ---------------------------------------------------------------
VIDEOS=()
while IFS= read -r -d '' f; do VIDEOS+=("$f"); done < <(find data/raw_videos -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.MOV' \) -print0 | sort -z)
if [ "${#VIDEOS[@]}" -eq 0 ]; then
    fail "No videos in data/raw_videos/"; exit 1
fi
log "Found ${#VIDEOS[@]} video(s)"

# --- DB -------------------------------------------------------------------
rm -f data/anpr.db data/anpr.db-wal data/anpr.db-shm
PYTHONPATH="$SCRIPT_DIR" "$PY" -c "from src.db.schema import init_db; init_db()"
log "Database initialized: data/anpr.db"

# --- device ---------------------------------------------------------------
COMPUTE="${1:-cpu}"

# --- backend + fusion (silent, needed for trajectories) ---------------------
PYTHONPATH="$SCRIPT_DIR" "$PY" -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --log-level warning >/dev/null 2>&1 &
PIDS+=($!)
for i in $(seq 1 30); do
    sleep 1
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
done

PYTHONPATH="$SCRIPT_DIR" "$PY" -m src.fusion.worker --interval 3 --log-level WARNING >/dev/null 2>&1 &
PIDS+=($!)

# --- camera workers --------------------------------------------------------
CAM_PIDS=()
n=0
for video in "${VIDEOS[@]}"; do
    n=$((n+1))
    cam="cam_$n"
    log "Processing: $cam <- $(basename "$video")"
    PYTHONPATH="$SCRIPT_DIR" "$PY" -m src.pipeline_runner \
        --camera-id "$cam" --video "$video" --gps-lat 12.9758 --gps-lon 77.6082 \
        --device "$COMPUTE" --speed-factor 0.5 --log-level WARNING >/dev/null 2>&1 &
    PIDS+=($!)
    CAM_PIDS+=($!)
done

# Wait for all camera workers to finish, then show results
for pid in "${CAM_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
sleep 1  # let fusion persist any final sightings

echo
PYTHONPATH="$SCRIPT_DIR" "$PY" "$SCRIPT_DIR/scripts/plates_report.py"

echo
log "Done. Run 'python scripts/plates_report.py' again anytime to re-print results."
echo "Press Ctrl+C to stop the backend/fusion services."