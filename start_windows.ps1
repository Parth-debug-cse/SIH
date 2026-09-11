# ANPR Pipeline Go-Live starter (Windows PowerShell).
# Starts backend + shared fusion worker + one pipeline runner per video found
# in data/raw_videos/ (any .mp4).  Nothing else is touched or renamed.
param([string]$Device = "cpu")

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
Set-Location -LiteralPath $SCRIPT_DIR

function Log($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[X] $msg" -ForegroundColor Red }

$PIDS = @()
$Killed = $false
$Cleanup = {
    if ($script:Killed) { return }
    $script:Killed = $true
    Log "Stopping all processes..."
    foreach ($p in $script:PIDS) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
    Log "Stopped."
}
Register-EngineEvent -SourceIdentifier ProcessExited -Action $Cleanup | Out-Null

function Stop-Cleanup {
    & $Cleanup
    exit 0
}

# --- python ---------------------------------------------------------------
$VENV = Resolve-Path -LiteralPath ".venv\Scripts\python.exe" -ErrorAction SilentlyContinue
if ($VENV) {
    $PY = $VENV.Path
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $PY = "python" } else {
        Fail "No python found. Create your venv: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt"
        exit 1
    }
}
Log "Python: $PY"

# --- model ----------------------------------------------------------------
if (-not (Test-Path -LiteralPath "models\detection\yolov8n.pt")) {
    if (Test-Path -LiteralPath "yolov8n.pt") {
        Move-Item -LiteralPath "yolov8n.pt" -Destination "models\detection\yolov8n.pt"
    } else {
        Fail "model missing: models\detection\yolov8n.pt"
        exit 1
    }
}

# --- videos: auto-discover, assign cam_1, cam_2, ... in sorted order --------
$VIDEOS = @(Get-ChildItem -LiteralPath "data\raw_videos" -File | Where-Object { $_.Extension -in ".mp4", ".MOV" } | Sort-Object Name | ForEach-Object { $_.FullName })
if ($VIDEOS.Count -eq 0) {
    Fail "No videos in data/raw_videos/"
    exit 1
}
Log "Found $($VIDEOS.Count) video(s)"

# --- DB: reset data/anpr.db, then init -------------------------------------
Remove-Item -LiteralPath "data\anpr.db", "data\anpr.db-wal", "data\anpr.db-shm" -Force -ErrorAction SilentlyContinue
& $PY -c "from src.db.schema import init_db; init_db()"
if ($LASTEXITCODE -ne 0) { Fail "DB init failed"; exit 1 }
Log "Database initialized: data/anpr.db"

# --- device ----------------------------------------------------------------
$COMPUTE = $Device
Log "Device: $COMPUTE"

# --- backend ---------------------------------------------------------------
Log "Starting backend on :8000..."
$env:PYTHONPATH = $SCRIPT_DIR
$backend = Start-Process -FilePath $PY -ArgumentList @("-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info") -PassThru -WindowStyle Hidden
$PIDS += $backend.Id

$ready = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        $ready = $true
        break
    } catch {}
}
if (-not $ready) { Fail "Backend failed to start in 30s"; & $Cleanup; exit 1 }
Log "Backend ready"

# --- fusion worker ---------------------------------------------------------
Log "Starting shared fusion worker..."
$fusion = Start-Process -FilePath $PY -ArgumentList @("-m", "src.fusion.worker", "--interval", "3", "--log-level", "INFO") -PassThru -WindowStyle Hidden
$PIDS += $fusion.Id

# --- camera workers --------------------------------------------------------
$n = 0
foreach ($video in $VIDEOS) {
    $n++
    $cam = "cam_$n"
    Log "Starting worker: $cam -> $(Split-Path $video -Leaf)"
    $w = Start-Process -FilePath $PY -ArgumentList @("-m", "src.pipeline_runner", "--camera-id", $cam, "--video", $video, "--gps-lat", "12.9758", "--gps-lon", "77.6082", "--device", $COMPUTE, "--speed-factor", "0.5", "--log-level", "INFO") -PassThru -WindowStyle Hidden
    $PIDS += $w.Id
}

Write-Host ""
Write-Host "============================================"
Write-Host "  ANPR PIPELINE IS LIVE"
Write-Host "============================================"
Write-Host "  Health:     curl http://localhost:8000/health"
Write-Host "  Trajectory: curl http://localhost:8000/trajectory/KA01AB1234"
Write-Host "  Sightings:  curl http://localhost:8000/sightings"
Write-Host "  Alerts:     curl http://localhost:8000/alerts"
Write-Host "  Ctrl+C to stop."
Write-Host "============================================"

try {
    Wait-Process -Id $PIDS -ErrorAction SilentlyContinue
} catch {}
& $Cleanup