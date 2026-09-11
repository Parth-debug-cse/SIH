from __future__ import annotations

import sqlite3
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.db.schema import get_connection

app = FastAPI(title="ANPR Pipeline API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _query(sql: str, params: tuple = ()) -> list[dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


# ---------------------------------------------------------------------------
# Fusion (shared cross-camera engine)
# ---------------------------------------------------------------------------
@app.post("/fusion/run")
def fusion_run():
    """Trigger the shared fusion engine over the current sightings."""
    try:
        from src.fusion.engine import DEFAULT_CAMERAS_PATH, run_fusion_once
        stats = run_fusion_once(cameras_config_path=str(DEFAULT_CAMERAS_PATH))
        return {"status": "ok", **stats}
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Fusion failed: {exc}")


# ---------------------------------------------------------------------------
# Trajectory (fused, produced by the fusion engine)
# ---------------------------------------------------------------------------
def _trajectory_from_children(traj_row: dict) -> list[dict]:
    """Build ordered sightings for a trajectory from trajectory_sightings."""
    rows = _query(
        "SELECT plate, camera_id, gps_lat, gps_lon, timestamp, direction, "
        "       confidence, vehicle_class, association_score "
        "FROM trajectory_sightings WHERE trajectory_id = ? ORDER BY seq",
        (traj_row["id"],),
    )
    return [
        {
            "camera_id": r["camera_id"],
            "gps_lat": r["gps_lat"],
            "gps_lon": r["gps_lon"],
            "timestamp": r["timestamp"],
            "confidence": r["confidence"],
            "direction": r["direction"],
            "vehicle_class": r["vehicle_class"],
            "association_score": r["association_score"],
        }
        for r in rows
    ]


def _trajectory_from_raw_sightings(plate: str) -> list[dict]:
    """Fallback: derive ordered sightings from raw DB rows.

    Used only when a fused trajectory record exists but its child rows were
    not materialised (e.g. hand-seeded test data).  The response
    ``route_summary`` marks this case via ``sighting_source``.
    """
    rows = _query(
        "SELECT camera_id, gps_lat, gps_lon, timestamp, plate_confidence, "
        "       vehicle_class, direction "
        "FROM sightings WHERE plate = ? ORDER BY timestamp",
        (plate,),
    )
    return [
        {
            "camera_id": r["camera_id"],
            "gps_lat": r["gps_lat"],
            "gps_lon": r["gps_lon"],
            "timestamp": r["timestamp"],
            "confidence": r["plate_confidence"],
            "direction": r["direction"],
            "vehicle_class": r["vehicle_class"],
            "association_score": None,
        }
        for r in rows
    ]


@app.get("/trajectory/{plate}")
def trajectory(plate: str):
    """Return the fused trajectory produced by the fusion engine.

    The response is built from ``trajectories`` + ``trajectory_sightings``
    (the actual output of the fusion engine), not from ad-hoc raw-sighting
    queries.  A raw-sighting fallback is used only for trajectory records
    without materialised children and is flagged in the response.
    """
    plate = plate.strip().upper()
    rows = _query(
        "SELECT id, trajectory_code, plate, first_camera, last_camera, "
        "       first_seen, last_seen, num_sightings, route, total_distance_km, "
        "       total_duration_seconds, trajectory_confidence, source "
        "FROM trajectories WHERE plate = ? ORDER BY id DESC",
        (plate,),
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No fused trajectory for plate '{plate}' yet. "
                   "Raw sightings (if any) are available at /sightings. "
                   "Run the fusion engine first.",
        )

    traj = rows[0]
    sightings = _trajectory_from_children(traj)
    sighting_source = "trajectory_sightings"
    if not sightings:
        sightings = _trajectory_from_raw_sightings(plate)
        sighting_source = "raw_sightings_fallback"

    duration = traj["total_duration_seconds"]
    if duration is None and len(sightings) >= 2:
        duration = sightings[-1]["timestamp"] - sightings[0]["timestamp"]

    return {
        "trajectory_id": traj["id"],
        "trajectory_code": traj["trajectory_code"],
        "plate": plate,
        "source": traj["source"],
        "trajectory_confidence": traj["trajectory_confidence"],
        "sighting_source": sighting_source,
        "sightings": sightings,
        "route_summary": {
            "first_camera": traj["first_camera"],
            "last_camera": traj["last_camera"],
            "num_sightings": traj["num_sightings"],
            "duration_seconds": duration,
            "route": traj["route"],
            "total_distance_km": traj["total_distance_km"],
        },
        "total_route_duration": duration,
        "total_distance_km": traj["total_distance_km"],
    }


# ---------------------------------------------------------------------------
# Analytics – Density
# ---------------------------------------------------------------------------
@app.get("/analytics/density")
def analytics_density():
    rows = _query(
        "SELECT camera_id, "
        "  AVG(vehicle_count) AS avg_density, "
        "  MAX(vehicle_count) AS latest_count "
        "FROM analytics GROUP BY camera_id"
    )
    cameras = [
        {
            "camera_id": r["camera_id"],
            "avg_density": r["avg_density"],
            "latest_count": r["latest_count"],
        }
        for r in rows
    ]
    return {"cameras": cameras}


# ---------------------------------------------------------------------------
# Analytics – Congestion
# ---------------------------------------------------------------------------
@app.get("/analytics/congestion")
def analytics_congestion():
    rows = _query(
        "SELECT camera_id, congestion_flag, vehicle_count, avg_speed, density_level "
        "FROM analytics ORDER BY timestamp DESC"
    )
    seen: dict[str, dict] = {}
    for r in rows:
        cid = r["camera_id"]
        if cid not in seen:
            seen[cid] = r

    cameras = [
        {
            "camera_id": r["camera_id"],
            "is_congested": bool(r["congestion_flag"]),
            "vehicle_count": r["vehicle_count"],
            "avg_speed": r["avg_speed"],
            "density_level": r["density_level"],
        }
        for r in seen.values()
    ]
    return {"cameras": cameras}


# ---------------------------------------------------------------------------
# Analytics – OD Patterns (from real fused trajectories)
# ---------------------------------------------------------------------------
@app.get("/analytics/od-patterns")
def analytics_od_patterns():
    rows = _query(
        "SELECT first_camera AS origin_camera, last_camera AS dest_camera, "
        "  COUNT(*) AS count, plate "
        "FROM trajectories GROUP BY first_camera, last_camera"
    )

    patterns: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["origin_camera"], r["dest_camera"])
        if key not in patterns:
            patterns[key] = {
                "origin_camera": r["origin_camera"],
                "dest_camera": r["dest_camera"],
                "count": r["count"],
                "sample_plates": [],
            }
        if len(patterns[key]["sample_plates"]) < 5:
            patterns[key]["sample_plates"].append(r["plate"])

    return {"patterns": list(patterns.values())}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
@app.get("/alerts")
def get_alerts():
    rows = _query(
        "SELECT id, plate, alert_type, camera_id, gps_lat, gps_lon, "
        "  timestamp, details, acknowledged "
        "FROM alerts ORDER BY timestamp DESC"
    )
    return {"alerts": rows}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
class IngestBody(BaseModel):
    plate: str
    camera_id: str
    gps_lat: float
    gps_lon: float
    timestamp: float
    plate_confidence: Optional[float] = None
    vehicle_class: Optional[str] = None
    vehicle_bbox: Optional[str] = None
    plate_bbox: Optional[str] = None
    frame_number: Optional[int] = None
    track_id: Optional[int] = None


@app.post("/ingest")
def ingest(body: IngestBody):
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO sightings "
            "(plate, camera_id, gps_lat, gps_lon, timestamp, "
            " plate_confidence, vehicle_class, vehicle_bbox, plate_bbox, frame_number, track_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                body.plate,
                body.camera_id,
                body.gps_lat,
                body.gps_lon,
                body.timestamp,
                body.plate_confidence,
                body.vehicle_class,
                body.vehicle_bbox,
                body.plate_bbox,
                body.frame_number,
                body.track_id,
            ),
        )
        conn.commit()
        new_id = cur.lastrowid
    finally:
        conn.close()

    return {"status": "ok", "id": new_id}


# ---------------------------------------------------------------------------
# Sightings
# ---------------------------------------------------------------------------
@app.get("/sightings")
def list_sightings(plate: Optional[str] = Query(None, description="Filter by plate")):
    if plate:
        rows = _query(
            "SELECT * FROM sightings WHERE plate = ? ORDER BY timestamp DESC",
            (plate,),
        )
    else:
        rows = _query("SELECT * FROM sightings ORDER BY timestamp DESC")
    return {"sightings": rows}


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)