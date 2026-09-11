# ANPR Pipeline Go-Live starter (Windows PowerShell).
# Runs backend + shared fusion worker + one pipeline runner per video in
# data/raw_videos/, then prints: vehicles detected, plates read, plate numbers.
param([string]$Device = "cpu")

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
Set-Location -LiteralPath $SCRIPT_DIR

function Log($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "[X] $msg" -ForegroundColor Red; exit 1 }

# --- python ---------------------------------------------------------------
$VENV = Resolve-Path -LiteralPath ".venv\Scripts\python.exe" -ErrorAction SilentlyContinue
if ($VENV) { $PY = $VENV.Path }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $PY = "python" }
else { Fail "No python found. Create your venv: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt" }

# --- model ----------------------------------------------------------------
if (-not (Test-Path -LiteralPath "models\detection\yolov8n.pt")) {
    if (Test-Path -LiteralPath "yolov8n.pt") {
        Move-Item -LiteralPath "yolov8n.pt" -Destination "models\detection\yolov8n.pt"
    } else { Fail "model missing: models\detection\yolov8n.pt" }
}

# --- videos ---------------------------------------------------------------
$VIDEOS = @(Get-ChildItem -LiteralPath "data\raw_videos" -File | Where-Object { $_.Extension -in ".mp4", ".MOV" } | Sort-Object Name | ForEach-Object { $_.FullName })
if ($VIDEOS.Count -eq 0) { Fail "No videos in data/raw_videos/" }
Log "Found $($VIDEOS.Count) video(s)"

# --- DB -------------------------------------------------------------------
Remove-Item -LiteralPath "data\anpr.db", "data\anpr.db-wal", "data\anpr.db-shm" -Force -ErrorAction SilentlyContinue
& $PY -c "from src.db.schema import init_db; init_db()"
if ($LASTEXITCODE -ne 0) { Fail "DB init failed" }
Log "Database initialized: data/anpr.db"

# --- backend + fusion (silent, needed for trajectories) ---------------------
$env:PYTHONPATH = $SCRIPT_DIR
$backend = Start-Process -FilePath $PY -ArgumentList @("-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning") -PassThru -WindowStyle Hidden
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try { $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2; break } catch {}
}
$fusion = Start-Process -FilePath $PY -ArgumentList @("-m", "src.fusion.worker", "--interval", "3", "--log-level", "WARNING") -PassThru -WindowStyle Hidden

# --- camera workers --------------------------------------------------------
$PROCS = @()
$n = 0
foreach ($video in $VIDEOS) {
    $n++
    $cam = "cam_$n"
    Log "Processing: $cam <- $(Split-Path $video -Leaf)"
    $w = Start-Process -FilePath $PY -ArgumentList @("-m", "src.pipeline_runner", "--camera-id", $cam, "--video", $video, "--gps-lat", "12.9758", "--gps-lon", "77.6082", "--device", $Device, "--speed-factor", "0.5", "--log-level", "WARNING") -PassThru -WindowStyle Hidden
    $PROCS += $w
}

# Wait for all camera workers to finish, then show results
foreach ($w in $PROCS) { $w.WaitForExit() }
Start-Sleep -Seconds 1  # let fusion persist any final sightings

Write-Host ""
& $PY "$SCRIPT_DIR\scripts\plates_report.py"

Write-Host ""
Log "Done. Run 'python scripts\plates_report.py' again anytime to re-print results."
Write-Host "Press Ctrl+C to stop the backend/fusion services."

# Keep backend + fusion alive until user stops them
try { Wait-Process -Id @($backend.Id, $fusion.Id) -ErrorAction SilentlyContinue } catch {}
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue