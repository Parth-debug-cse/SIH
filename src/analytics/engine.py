import json
import time
from collections import defaultdict
from pathlib import Path

from src.db.schema import get_connection

DEFAULT_CAMERAS_PATH = Path(__file__).parent.parent.parent / "data" / "calibration" / "cameras.json"


class TrafficAnalytics:
    def __init__(self, cameras_config_path=None):
        self.cameras = {}
        self.camera_counts = defaultdict(list)
        self.camera_speeds = defaultdict(list)
        self.analytics_store = []
        if cameras_config_path:
            self.load_cameras(cameras_config_path)

    def load_cameras(self, config_path) -> None:
        with open(config_path, "r") as f:
            data = json.load(f)
        for cam in data.get("cameras", []):
            self.cameras[cam["camera_id"]] = {
                "pixel_to_meter_ratio": cam.get("pixel_to_meter_ratio", 1.0),
                "gps_lat": cam.get("gps_lat", 0.0),
                "gps_lon": cam.get("gps_lon", 0.0),
                "description": cam.get("description", ""),
            }

    def _get_pixel_to_meter(self, camera_id):
        cam = self.cameras.get(camera_id)
        if cam:
            return cam["pixel_to_meter_ratio"]
        return 1.0

    def update_camera_count(self, camera_id, timestamp, vehicle_count, frame_shape=None) -> None:
        self.camera_counts[camera_id].append({
            "timestamp": timestamp,
            "count": vehicle_count,
            "frame_shape": frame_shape,
        })

    def compute_speed(self, camera_id, bbox1, bbox2, time_delta) -> float:
        if time_delta <= 0:
            return 0.0
        cx1 = (bbox1[0] + bbox1[2]) / 2.0
        cy1 = (bbox1[1] + bbox1[3]) / 2.0
        cx2 = (bbox2[0] + bbox2[2]) / 2.0
        cy2 = (bbox2[1] + bbox2[3]) / 2.0
        pixel_dist = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
        ratio = self._get_pixel_to_meter(camera_id)
        real_dist_m = pixel_dist * ratio
        speed_ms = real_dist_m / time_delta
        speed_kmh = speed_ms * 3.6
        self.camera_speeds[camera_id].append(speed_kmh)
        return speed_kmh

    def compute_density(self, camera_id, time_window_seconds=60) -> float:
        now = time.time()
        counts = [
            e["count"]
            for e in self.camera_counts.get(camera_id, [])
            if now - e["timestamp"] <= time_window_seconds
        ]
        if not counts:
            return 0.0
        return sum(counts) / len(counts)

    def detect_congestion(self, camera_id, time_window_seconds=300) -> dict:
        now = time.time()
        counts = [
            e["count"]
            for e in self.camera_counts.get(camera_id, [])
            if now - e["timestamp"] <= time_window_seconds
        ]
        total = sum(counts) if counts else 0
        if total < 5:
            level = "low"
        elif total <= 10:
            level = "medium"
        else:
            level = "high"
        return {
            "camera_id": camera_id,
            "vehicle_count": total,
            "congestion_level": level,
            "is_congested": level == "high",
        }

    def compute_od_patterns(self, trajectories) -> list:
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

    def get_analytics_summary(self) -> dict:
        summary = {}
        for cam_id in self.cameras:
            counts = self.camera_counts.get(cam_id, [])
            speeds = self.camera_speeds.get(cam_id, [])
            total_vehicles = sum(e["count"] for e in counts)
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
            congestion = self.detect_congestion(cam_id)
            density = self.compute_density(cam_id)
            summary[cam_id] = {
                "total_vehicles": total_vehicles,
                "avg_speed_kmh": round(avg_speed, 2),
                "density": round(density, 2),
                "congestion": congestion,
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
