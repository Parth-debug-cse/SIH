# ANPR Pipeline Go-Live starter (Windows PowerShell).
# Runs backend + fusion worker silently, then processes each video in
# data/raw_videos/ IN THE FOREGROUND so results print live to the console,
# then shows vehicles / plates / plate numbers from the database.
param([string]$Device = "cpu")

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
Set-Location -LiteralPath $SCRIPT_DIR

function Log($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
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

# --- camera workers (FOREGROUND: results print live on screen) -------------
$n = 0
foreach ($video in $VIDEOS) {
    $n++
    $cam = "cam_$n"
    Write-Host ""
    Write-Host "========== PROCESSING: $cam <- $(Split-Path $video -Leaf) =========="
    & $PY -m src.pipeline_runner --camera-id $cam --video $video --gps-lat 12.9758 --gps-lon 77.6082 --device $Device --speed-factor 0 --log-level INFO
    if ($LASTEXITCODE -ne 0) { Warn "Pipeline exit code $LASTEXITCODE" }
}

# --- show live results -----------------------------------------------------
Start-Sleep -Seconds 1
Write-Host ""
& $PY "$SCRIPT_DIR\scripts\plates_report.py"

Write-Host ""
Log "Rerun report anytime: python scripts\plates_report.py"
Write-Host "Backend + fusion remain up. Press Ctrl+C to stop them."

# Keep backend + fusion alive until user stops them
try { Wait-Process -Id @($backend.Id, $fusion.Id) -ErrorAction SilentlyContinue } catch {}
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue