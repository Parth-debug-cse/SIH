"""Macro traffic analytics over the shared database.

Density and congestion are computed from **unique active track IDs** per
camera over a time window wherever tracking is available - not from a sum of
per-frame detections.  When track data is unavailable the engine falls back
to a documented proxy (mean per-frame count / latest count).

Speed estimates use the camera calibration (``pixel_to_meter_ratio``) from
``cameras.json``.  When calibration is missing, ``compute_speed`` returns
``None`` instead of fabricating ``0 km/h``.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from src.db.schema import get_connection

DEFAULT_CAMERAS_PATH = Path(__file__).parent.parent.parent / "data" / "calibration" / "cameras.json"

# Congestion thresholds over unique active vehicles in a window (documented
# in DECISIONS.md / VERIFICATION.md).  A proxy metric - see module docstring.
CONGESTION_LOW_MAX = 2         # <= 2 active vehicles -> low
CONGESTION_MEDIUM_MAX = 6      # <= 6 active vehicles -> medium, else high

DEFAULT_DENSITY_WINDOW_SECONDS = 60
DEFAULT_CONGESTION_WINDOW_SECONDS = 300


class TrafficAnalytics:
    def __init__(self, cameras_config_path=None):
        self.cameras = {}
        self.camera_counts = defaultdict(list)   # camera -> [{timestamp,count,track_ids,frame_shape}]
        self.camera_speeds = defaultdict(list)
        self._track_states = defaultdict(dict)   # camera -> {track_id: (bbox, timestamp)}
        self.analytics_store = []
        if cameras_config_path:
            self.load_cameras(cameras_config_path)

    def load_cameras(self, config_path) -> None:
        with open(config_path, "r") as f:
            data = json.load(f)
        for cam in data.get("cameras", []):
            self.cameras[cam["camera_id"]] = {
                "pixel_to_meter_ratio": cam.get("pixel_to_meter_ratio"),
                "gps_lat": cam.get("gps_lat", 0.0),
                "gps_lon": cam.get("gps_lon", 0.0),
                "description": cam.get("description", ""),
            }

    # ------------------------------------------------------------------
    # Data intake
    # ------------------------------------------------------------------

    def update_camera_count(self, camera_id, timestamp, vehicle_count, frame_shape=None, track_ids=None) -> None:
        """Record the state of one frame for a camera.

        Args:
            vehicle_count: number of detections (used only as a proxy when
                track data is unavailable).
            track_ids: unique active track IDs for this frame (preferred).
        """
        self.camera_counts[camera_id].append({
            "timestamp": timestamp,
            "count": vehicle_count,
            "track_ids": list(track_ids) if track_ids is not None else None,
            "frame_shape": frame_shape,
        })

    def update_tracked_speeds(self, camera_id, tracks, timestamp) -> None:
        """Compute per-track speeds from consecutive track observations.

        Uses bbox center displacement * ``pixel_to_meter_ratio`` / dt.
        Only produces a value when calibration is present; otherwise nothing
        is appended (and ``avg_speed`` will be reported as unavailable).
        """
        states = self._track_states[camera_id]
        for t in tracks:
            tid = t["track_id"]
            bbox = t["bbox"]
            prev = states.get(tid)
            if prev is not None:
                prev_bbox, prev_ts = prev
                dt = timestamp - prev_ts
                if dt > 0:
                    speed = self.compute_speed(camera_id, prev_bbox, bbox, dt, record=True)
            states[tid] = (bbox, timestamp)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _get_pixel_to_meter(self, camera_id):
        cam = self.cameras.get(camera_id)
        if not cam:
            return None
        ratio = cam.get("pixel_to_meter_ratio")
        if ratio is None:
            return None
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            return None
        return ratio if ratio > 0 else None

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------

    def compute_speed(self, camera_id, bbox1, bbox2, time_delta, record=True):
        """Estimate travel speed (km/h) from bbox displacement.

        Returns ``None`` (never an invented ``0 km/h``) when the elapsed time
        is non-positive or the pixel-to-meter calibration is missing.  The
        same camera-fixed ``pixel_to_meter_ratio`` documented for the demo is
        applied to center displacement.
        """
        if time_delta is None or time_delta <= 0:
            return None
        ratio = self._get_pixel_to_meter(camera_id)
        if ratio is None:
            return None
        cx1 = (bbox1[0] + bbox1[2]) / 2.0
        cy1 = (bbox1[1] + bbox1[3]) / 2.0
        cx2 = (bbox2[0] + bbox2[2]) / 2.0
        cy2 = (bbox2[1] + bbox2[3]) / 2.0
        pixel_dist = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
        real_dist_m = pixel_dist * ratio
        speed_ms = real_dist_m / time_delta
        speed_kmh = speed_ms * 3.6
        if record:
            self.camera_speeds[camera_id].append(speed_kmh)
        return speed_kmh

    # ------------------------------------------------------------------
    # Active-vehicle metrics (track-based)
    # ------------------------------------------------------------------

    def _window_entries(self, camera_id, window_seconds, now=None):
        now = time.time() if now is None else now
        return [
            e for e in self.camera_counts.get(camera_id, [])
            if now - e["timestamp"] <= window_seconds
        ]

    def _has_track_data(self, entries) -> bool:
        return any(e.get("track_ids") is not None and len(e["track_ids"]) > 0 for e in entries)

    def compute_active_vehicles(self, camera_id, time_window_seconds=DEFAULT_DENSITY_WINDOW_SECONDS) -> int:
        """Number of unique active track IDs over the window.

        Falls back to the most recent per-frame count (documented proxy) when
        track data is unavailable.
        """
        entries = self._window_entries(camera_id, time_window_seconds)
        if not entries:
            return 0
        if self._has_track_data(entries):
            return len({tid for e in entries for tid in (e.get("track_ids") or [])})
        return entries[-1]["count"]

    def compute_density(self, camera_id, time_window_seconds=DEFAULT_DENSITY_WINDOW_SECONDS) -> float:
        """Mean per-frame vehicle count OR unique active tracks over a window.

        With track data available this is the number of unique active track
        IDs (a real occupancy metric).  Without it, the mean per-frame count
        is reported as a documented proxy.
        """
        entries = self._window_entries(camera_id, time_window_seconds)
        if not entries:
            return 0.0
        if self._has_track_data(entries):
            return float(self.compute_active_vehicles(camera_id, time_window_seconds))
        counts = [e["count"] for e in entries]
        return sum(counts) / len(counts)

    def detect_congestion(self, camera_id, time_window_seconds=DEFAULT_CONGESTION_WINDOW_SECONDS) -> dict:
        """Congestion level from unique active vehicles in the window.

        Thresholds (documented): <= 2 low, <= 6 medium, else high.
        Falls back to the latest per-frame count as a proxy when tracking is
        unavailable.
        """
        entries = self._window_entries(camera_id, time_window_seconds)
        if not entries:
            active = 0
        elif self._has_track_data(entries):
            active = len({tid for e in entries for tid in (e.get("track_ids") or [])})
        else:
            active = entries[-1]["count"]

        if active <= CONGESTION_LOW_MAX:
            level = "low"
        elif active <= CONGESTION_MEDIUM_MAX:
            level = "medium"
        else:
            level = "high"
        return {
            "camera_id": camera_id,
            "vehicle_count": active,
            "congestion_level": level,
            "is_congested": level == "high",
        }

    # ------------------------------------------------------------------
    # O-D patterns
    # ------------------------------------------------------------------

    def compute_od_patterns(self, trajectories) -> list:
        """Origin -> destination counts extracted from fused trajectories."""
        pairs = defaultdict(lambda: {"count": 0, "plates": set()})
        for plate, sightings in trajectories.items():
            if len(sightings) < 2:
                continue
            origin = sightings[0]["camera_id"]
            dest = sightings[-1]["camera_id"]
            key = (origin, dest)
            pairs[key]["count"] += 1
            pairs[key]["plates"].add(plate)
        results = []
        for (origin, dest), info in pairs.items():
            results.append({
                "origin_camera": origin,
                "dest_camera": dest,
                "count": info["count"],
                "plates": list(info["plates"]),
            })
        return results

    # ------------------------------------------------------------------
    # Summary / persistence
    # ------------------------------------------------------------------

    def get_analytics_summary(self) -> dict:
        summary = {}
        for cam_id in self.cameras:
            counts = self.camera_counts.get(cam_id, [])
            speeds = self.camera_speeds.get(cam_id, [])
            active = self.compute_active_vehicles(cam_id)
            cong = self.detect_congestion(cam_id)
            # avg_speed is None (reported unavailable) unless measured.
            avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else None
            summary[cam_id] = {
                "total_detections": sum(e["count"] for e in counts),
                "total_vehicles": sum(e["count"] for e in counts),
                "active_vehicles": active,
                "avg_speed_kmh": avg_speed,
                "density": round(self.compute_density(cam_id), 2),
                "congestion": cong,
            }
        return summary

    def store_analytics(self, camera_id, timestamp, vehicle_count, avg_speed, density_level, congestion_flag) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO analytics
                   (camera_id, timestamp, vehicle_count, avg_speed, density_level, congestion_flag)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (camera_id, timestamp, vehicle_count, avg_speed, density_level, int(congestion_flag)),
            )
            conn.commit()
        finally:
            conn.close()