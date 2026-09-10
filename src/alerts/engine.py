"""
Alert engine for blacklist matching and anomaly detection.

Loads a blacklist of plate numbers from JSON, checks every new OCR reading
against it using exact and fuzzy (Levenshtein) matching, stores alerts in
the database, and provides an alert feed for the API.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from rapidfuzz.distance import Levenshtein

from src.db import schema

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_BLACKLIST_PATH = _DATA_DIR / "blacklist.json"
_DEFAULT_PLATES = ["MH12AB1234", "DL01CA5678", "KA01XX9999"]
_FUZZY_THRESHOLD = 2


class AlertEngine:
    """Blacklist matching and anomaly detection engine.

    Args:
        blacklist_path: Path to a JSON file containing a list of plate
            strings.  When *None* the built-in default location is used.
    """

    def __init__(self, blacklist_path: Optional[str | Path] = None) -> None:
        self._blacklist_path = Path(blacklist_path) if blacklist_path else _DEFAULT_BLACKLIST_PATH
        self._plates: list[str] = self.load_blacklist(self._blacklist_path)

    # ------------------------------------------------------------------
    # Blacklist management
    # ------------------------------------------------------------------

    def load_blacklist(self, blacklist_path: Optional[str | Path] = None) -> list[str]:
        """Load the blacklist from a JSON file.

        If the file does not exist a default list is written and returned.

        Args:
            blacklist_path: Path to the JSON file.  Defaults to the path
                provided at construction time.

        Returns:
            Normalised (uppercased) list of plate strings.
        """
        path = Path(blacklist_path) if blacklist_path else self._blacklist_path

        if not path.exists():
            logger.warning("Blacklist file not found at %s – creating default.", path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_DEFAULT_PLATES, indent=2), encoding="utf-8")

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Blacklist JSON must be a list of plate strings")
            plates = [str(p).upper().strip() for p in raw]
            logger.info("Loaded %d plate(s) from %s", len(plates), path)
            return plates
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse blacklist at %s: %s", path, exc)
            return []

    # ------------------------------------------------------------------
    # Plate checking
    # ------------------------------------------------------------------

    def check_plate(
        self,
        plate: str,
        camera_id: Optional[str] = None,
        gps_lat: Optional[float] = None,
        gps_lon: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> list[dict]:
        """Check a plate against the blacklist and store any matching alerts.

        Matching strategy:
        1. Exact match (case-insensitive).
        2. Fuzzy match via Levenshtein distance <= 2.

        Args:
            plate: The OCR-read plate string.
            camera_id: Identifier of the camera that captured the reading.
            gps_lat: Latitude of the camera.
            gps_lon: Longitude of the camera.
            timestamp: Epoch timestamp of the reading.  Defaults to now.

        Returns:
            A list of alert dicts that were created (may be empty).
        """
        if not plate or not plate.strip():
            return []

        normalised = plate.upper().strip()
        ts = timestamp if timestamp is not None else time.time()
        created: list[dict] = []

        for bl_plate in self._plates:
            alert_type: Optional[str] = None
            matched_plate = bl_plate

            if normalised == bl_plate:
                alert_type = "blacklist_exact"
            elif Levenshtein.distance(normalised, bl_plate) <= _FUZZY_THRESHOLD:
                alert_type = "blacklist_fuzzy"

            if alert_type is not None:
                alert = self._store_alert(
                    plate=normalised,
                    alert_type=alert_type,
                    camera_id=camera_id,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    timestamp=ts,
                    matched_plate=matched_plate,
                )
                created.append(alert)

        return created

    def _store_alert(
        self,
        plate: str,
        alert_type: str,
        camera_id: Optional[str],
        gps_lat: Optional[float],
        gps_lon: Optional[float],
        timestamp: float,
        matched_plate: str,
    ) -> dict:
        """Insert an alert row and return the resulting dict."""
        details = json.dumps({"matched_plate": matched_plate})
        conn = schema.get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO alerts (plate, alert_type, camera_id, gps_lat, gps_lon, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (plate, alert_type, camera_id, gps_lat, gps_lon, timestamp, details),
            )
            conn.commit()
            alert_id = cursor.lastrowid
            logger.info(
                "Alert created: id=%s plate=%s type=%s", alert_id, plate, alert_type
            )
            return {
                "id": alert_id,
                "plate": plate,
                "alert_type": alert_type,
                "camera_id": camera_id,
                "gps_lat": gps_lat,
                "gps_lon": gps_lon,
                "timestamp": timestamp,
                "details": details,
                "acknowledged": 0,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Alert feed
    # ------------------------------------------------------------------

    def get_alerts(self, limit: int = 50) -> list[dict]:
        """Return the most recent alerts, newest first.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of alert dicts.
        """
        conn = schema.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT id, plate, alert_type, camera_id, gps_lat, gps_lon,
                       timestamp, details, acknowledged
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "id": row[0],
                    "plate": row[1],
                    "alert_type": row[2],
                    "camera_id": row[3],
                    "gps_lat": row[4],
                    "gps_lon": row[5],
                    "timestamp": row[6],
                    "details": row[7],
                    "acknowledged": row[8],
                }
                for row in rows
            ]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Acknowledgement
    # ------------------------------------------------------------------

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged.

        Args:
            alert_id: Primary key of the alert row.

        Returns:
            *True* if a row was updated, *False* otherwise.
        """
        conn = schema.get_connection()
        try:
            cursor = conn.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,),
            )
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.info("Alert %s acknowledged", alert_id)
            else:
                logger.warning("Alert %s not found for acknowledgement", alert_id)
            return updated
        finally:
            conn.close()
