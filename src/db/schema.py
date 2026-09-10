import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "anpr.db"

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    first_camera TEXT NOT NULL,
    last_camera TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    num_sightings INTEGER NOT NULL,
    route TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
"""

def init_db(db_path=None):
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    return path

def get_connection(db_path=None):
    path = db_path or DB_PATH
    return sqlite3.connect(str(path))

if __name__ == "__main__":
    p = init_db()
    print(f"Database initialized at {p}")
