from __future__ import annotations

import sqlite3
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.db.schema import get_connection

app = FastAPI(title="ANPR Pipeline API", version="1.0.0")

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
# Trajectory
# ---------------------------------------------------------------------------
@app.get("/trajectory/{plate}")
def trajectory(plate: str):
    rows = _query(
        "SELECT camera_id, gps_lat, gps_lon, timestamp, plate_confidence "
        "FROM sightings WHERE plate = ? ORDER BY timestamp",
        (plate,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No sightings for plate '{plate}'")

    sightings = [
        {
            "camera_id": r["camera_id"],
            "gps_lat": r["gps_lat"],
            "gps_lon": r["gps_lon"],
            "timestamp": r["timestamp"],
            "confidence": r["plate_confidence"],
        }
        for r in rows
    ]

    first_ts = rows[0]["timestamp"]
    last_ts = rows[-1]["timestamp"]

    return {
        "plate": plate,
        "sightings": sightings,
        "route_summary": {
            "first_camera": rows[0]["camera_id"],
            "last_camera": rows[-1]["camera_id"],
            "num_sightings": len(sightings),
            "duration_seconds": last_ts - first_ts,
        },
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
        "SELECT camera_id, congestion_flag, vehicle_count, density_level "
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
            "density_level": r["density_level"],
        }
        for r in seen.values()
    ]
    return {"cameras": cameras}


# ---------------------------------------------------------------------------
# Analytics – OD Patterns
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


@app.post("/ingest")
def ingest(body: IngestBody):
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO sightings "
            "(plate, camera_id, gps_lat, gps_lon, timestamp, "
            " plate_confidence, vehicle_class, vehicle_bbox, plate_bbox, frame_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
