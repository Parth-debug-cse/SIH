"""Shared cross-camera fusion engine.

This is the single logical fusion service for the demo.  Camera workers
write normalized sightings to the shared SQLite database; this engine
consumes those sightings and persists unified trajectories back to the
database in a ``trajectories`` + ``trajectory_sightings`` model.

The engine is safe to run concurrently (as a dedicated fusion worker
process, from each camera worker at end-of-run, or via the ``/fusion/run``
API endpoint): trajectory tables are rebuilt inside a single immediate
transaction, so concurrent runs serialize without corruption.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from src.db.schema import get_connection
from src.fusion.reid import CrossCameraFusion

logger = logging.getLogger(__name__)

DEFAULT_CAMERAS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "calibration" / "cameras.json"

_SIGHTING_COLUMNS = (
    "id", "plate", "plate_confidence AS confidence", "camera_id",
    "gps_lat", "gps_lon", "timestamp", "vehicle_class", "track_id", "direction",
)


def load_sightings(db_path: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """Load all sighting rows from the shared database."""
    conn = get_connection(db_path)
    try:
        cols = ", ".join(_SIGHTING_COLUMNS)
        rows = conn.execute(
            f"SELECT {cols} FROM sightings ORDER BY timestamp"
        ).fetchall()
        sightings: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["plate"] = d["plate"].strip().upper()
            d["confidence"] = d.get("confidence")
            sightings.append(d)
        return sightings
    finally:
        conn.close()


def _trajectory_stats(
    sightings: list[dict[str, Any]],
    fusion: CrossCameraFusion,
) -> dict[str, Any]:
    """Compute summary stats for an ordered trajectory list."""
    first_ts = fusion._epoch(sightings[0]["timestamp"])
    last_ts = fusion._epoch(sightings[-1]["timestamp"])
    distance = 0.0
    for a, b in zip(sightings, sightings[1:]):
        distance += fusion.compute_distance_km(
            a["gps_lat"], a["gps_lon"], b["gps_lat"], b["gps_lon"]
        )

    if len(sightings) > 1:
        link_scores = [
            fusion.associate_sightings(a, b).association_score
            for a, b in zip(sightings, sightings[1:])
        ]
        confidence = round(sum(link_scores) / len(link_scores), 4)
    else:
        confidence = round(float(sightings[0].get("confidence") or 0.5), 4)

    return {
        "first_camera": sightings[0]["camera_id"],
        "last_camera": sightings[-1]["camera_id"],
        "first_seen": first_ts,
        "last_seen": last_ts,
        "num_sightings": len(sightings),
        "route": "->".join(s["camera_id"] for s in sightings),
        "total_duration_seconds": last_ts - first_ts,
        "total_distance_km": round(distance, 4),
        "trajectory_confidence": confidence,
    }


def store_trajectories(
    trajectories: dict[str, list[dict[str, Any]]],
    db_path: Optional[str | Path] = None,
    source: str = "fusion_engine",
    fusion: Optional[CrossCameraFusion] = None,
) -> int:
    """Replace trajectory tables with the output of the fusion engine.

    Runs inside a single transaction (``BEGIN IMMEDIATE``) so that
    concurrent fusion runs are serialized safely.  Returns the number of
    trajectories written.
    """
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM trajectory_sightings")
        conn.execute("DELETE FROM trajectories")

        fusion = fusion or CrossCameraFusion()
        count = 0
        for plate, sightings in trajectories.items():
            if not sightings:
                continue
            stats = _trajectory_stats(sightings, fusion)
            cursor = conn.execute(
                """INSERT INTO trajectories
                   (trajectory_code, plate, first_camera, last_camera, first_seen,
                    last_seen, num_sightings, route, total_distance_km,
                    total_duration_seconds, trajectory_confidence, source)
                   VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    plate,
                    stats["first_camera"],
                    stats["last_camera"],
                    stats["first_seen"],
                    stats["last_seen"],
                    stats["num_sightings"],
                    stats["route"],
                    stats["total_distance_km"],
                    stats["total_duration_seconds"],
                    stats["trajectory_confidence"],
                    source,
                ),
            )
            trajectory_id = cursor.lastrowid
            conn.execute(
                "UPDATE trajectories SET trajectory_code = ? WHERE id = ?",
                (f"TRJ-{trajectory_id:06d}", trajectory_id),
            )

            prev_assoc = None
            for seq, s in enumerate(sightings):
                assoc_score = prev_assoc
                if seq > 0:
                    assoc = fusion.associate_sightings(sightings[seq - 1], s)
                    assoc_score = assoc.association_score if assoc.accepted else 0.0
                conn.execute(
                    """INSERT INTO trajectory_sightings
                       (trajectory_id, plate, camera_id, gps_lat, gps_lon, timestamp,
                        direction, confidence, vehicle_class, association_score, seq)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        trajectory_id,
                        s["plate"],
                        s["camera_id"],
                        s.get("gps_lat"),
                        s.get("gps_lon"),
                        s["timestamp"],
                        s.get("direction"),
                        s.get("confidence"),
                        s.get("vehicle_class"),
                        assoc_score,
                        seq,
                    ),
                )
            count += 1

        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_fusion_once(
    db_path: Optional[str | Path] = None,
    cameras_config_path: Optional[str | Path] = None,
    max_edit_distance: int = 2,
    max_speed_kmh: float = 150.0,
    max_gap_seconds: float = 300.0,
    min_association_score: float = 0.55,
) -> dict[str, Any]:
    """Consume sightings from the shared DB and persist fused trajectories.

    Returns a small stats dict describing what was written.
    """
    config_path = Path(cameras_config_path or DEFAULT_CAMERAS_PATH)
    fusion = CrossCameraFusion(
        cameras_config_path=str(config_path) if config_path.exists() else None,
        max_edit_distance=max_edit_distance,
        max_speed_kmh=max_speed_kmh,
        max_gap_seconds=max_gap_seconds,
        min_association_score=min_association_score,
    )

    sightings = load_sightings(db_path)
    if not sightings:
        return {"num_sightings": 0, "num_trajectories": 0, "trajectories": []}

    trajectories = fusion.fuse_trajectories(sightings)
    count = store_trajectories(trajectories, db_path=db_path)

    _log_trajectories(trajectories, fusion)

    return {
        "num_sightings": len(sightings),
        "num_trajectories": count,
        "trajectories": sorted(trajectories.keys()),
    }


def _log_trajectories(
    trajectories: dict[str, list[dict[str, Any]]],
    fusion: CrossCameraFusion,
) -> None:
    """Emit observability events for accepted trajectories."""
    for plate, sightings in trajectories.items():
        route = " -> ".join(s["camera_id"] for s in sightings)
        conf = _trajectory_stats(sightings, fusion)["trajectory_confidence"]
        logger.info("[FUSION] accepted trajectory plate=%s route=%s sightings=%d conf=%.2f",
                    plate, route, len(sightings), conf)


def run_fusion_loop(
    interval_seconds: float = 3.0,
    db_path: Optional[str | Path] = None,
    cameras_config_path: Optional[str | Path] = None,
) -> None:
    """Poll the shared database until interrupted, fusing new sightings."""
    logger.info("[FUSION] worker started (interval=%.1fs, db=%s)",
                interval_seconds, str(Path(db_path or "default")))
    while True:
        try:
            stats = run_fusion_once(db_path=db_path, cameras_config_path=cameras_config_path)
            if stats["num_trajectories"]:
                logger.info(
                    "[FUSION] batch complete: %d sightings -> %d trajectories",
                    stats["num_sightings"], stats["num_trajectories"],
                )
        except Exception:
            logger.exception("[FUSION] fusion batch failed")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shared cross-camera fusion worker")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CAMERAS_PATH))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )
    run_fusion_loop(
        interval_seconds=args.interval,
        db_path=args.db_path,
        cameras_config_path=args.config,
    )