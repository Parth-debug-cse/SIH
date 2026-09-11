"""SQLite schema and connection helpers for the ANPR pipeline.

The database is the single shared persistence layer between camera workers,
the fusion engine, the API and the alert/analytics modules.  Multi-process
access is supported via:

* ``WAL`` journal mode (concurrent readers + single writer).
* a ``busy_timeout`` so short-lived writers wait (rather than fail) when
  another process is mid-transaction.
* ``insert_row()`` which retries transient ``database is locked`` errors
  with backoff.
"""

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "anpr.db"
DEFAULT_CAMERAS_PATH = Path(__file__).parent.parent.parent / "data" / "calibration" / "cameras.json"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    plate_confidence REAL,
    camera_id TEXT NOT NULL,
    gps_lat REAL NOT NULL,
    gps_lon REAL NOT NULL,
    timestamp REAL NOT NULL,
    vehicle_class TEXT,
    vehicle_bbox TEXT,
    plate_bbox TEXT,
    frame_number INTEGER,
    track_id INTEGER,
    direction TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_code TEXT UNIQUE,
    plate TEXT NOT NULL,
    first_camera TEXT NOT NULL,
    last_camera TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    num_sightings INTEGER NOT NULL,
    route TEXT,
    total_distance_km REAL,
    total_duration_seconds REAL,
    trajectory_confidence REAL,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trajectory_sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id),
    sighting_id INTEGER,
    plate TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    gps_lat REAL,
    gps_lon REAL,
    timestamp REAL NOT NULL,
    direction TEXT,
    confidence REAL,
    vehicle_class TEXT,
    association_score REAL,
    seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    camera_id TEXT,
    gps_lat REAL,
    gps_lon REAL,
    timestamp REAL NOT NULL,
    details TEXT,
    acknowledged INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    vehicle_count INTEGER,
    avg_speed REAL,
    density_level TEXT,
    congestion_flag INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sightings_plate ON sightings(plate);
CREATE INDEX IF NOT EXISTS idx_sightings_camera ON sightings(camera_id);
CREATE INDEX IF NOT EXISTS idx_sightings_timestamp ON sightings(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_plate ON alerts(plate);
CREATE INDEX IF NOT EXISTS idx_traj_plate ON trajectories(plate);
CREATE INDEX IF NOT EXISTS idx_traj_sightings_tid ON trajectory_sightings(trajectory_id);
"""

# Columns that may be added to databases created before a schema revision.
_MIGRATIONS: dict[str, list[str]] = {
    "sightings": [
        "track_id INTEGER",
        "direction TEXT",
    ],
    "trajectories": [
        "trajectory_code TEXT",
        "total_distance_km REAL",
        "total_duration_seconds REAL",
        "trajectory_confidence REAL",
        "source TEXT",
    ],
}


def _connect(path: Path) -> sqlite3.Connection:
    """Open a connection configured for multi-process use."""
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        # Some filesystems (e.g. network shares) do not support WAL;
        # trading safety for compatibility is acceptable for the demo.
        pass
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    for table, columns in _MIGRATIONS.items():
        try:
            existing = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
        except sqlite3.OperationalError:
            continue
        for column_def in columns:
            col_name = column_def.split()[0]
            if col_name not in existing:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column_def}"
                )


def init_db(db_path: Optional[str | Path] = None):
    """Create (or upgrade) the database schema at *db_path*."""
    path = Path(db_path or DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = _connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
        _ensure_columns(conn)
        conn.commit()
    finally:
        conn.close()
    return path


def get_connection(db_path: Optional[str | Path] = None):
    """Return a configured connection to the database."""
    path = Path(db_path or DB_PATH)
    return _connect(path)


def insert_row(
    sql: str,
    params: Iterable[Any],
    db_path: Optional[str | Path] = None,
    attempts: int = 5,
) -> int:
    """Execute an ``INSERT``/``UPDATE`` with retry on transient lock errors.

    Returns the last inserted row id (or 0 for non-insert statements).
    Short single-statement transactions keep write contention low across
    the camera workers and fusion engine.
    """
    path = Path(db_path or DB_PATH)
    params = tuple(params)
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        conn = _connect(path)
        try:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.lastrowid or 0
        except sqlite3.OperationalError as exc:
            last_error = exc
            if "locked" not in str(exc).lower() or attempt >= attempts - 1:
                raise
            conn.close()
            time.sleep(0.05 * (2 ** attempt))
        finally:
            conn.close()
    raise RuntimeError(f"DB write failed after {attempts} attempts") from last_error


if __name__ == "__main__":
    p = init_db()
    print(f"Database initialized at {p}")