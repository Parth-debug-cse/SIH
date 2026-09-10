"""Cross-camera fusion / re-identification module.

Fuses ANPR plate sightings across different camera feeds into unified
vehicle trajectories using fuzzy string matching and spatiotemporal
plausibility checks.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz


EARTH_RADIUS_KM = 6371.0


@dataclass
class CameraConfig:
    camera_id: str
    gps_lat: float
    gps_lon: float
    name: str = ""
    location: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossCameraFusion:
    """Fuses plate sightings across camera feeds into unified trajectories."""

    def __init__(
        self,
        cameras_config_path: str | Path | None = None,
        max_edit_distance: int = 2,
        max_speed_kmh: float = 150.0,
    ) -> None:
        self.max_edit_distance = max_edit_distance
        self.max_speed_kmh = max_speed_kmh
        self.cameras: dict[str, CameraConfig] = {}
        self.sightings: list[dict[str, Any]] = []

        if cameras_config_path is not None:
            self.load_cameras(cameras_config_path)

    # ------------------------------------------------------------------
    # Camera config
    # ------------------------------------------------------------------

    def load_cameras(self, config_path: str | Path) -> dict[str, CameraConfig]:
        """Load camera definitions from a JSON or CSV file.

        JSON format: {"cameras": [{"camera_id": ..., "gps_lat": ..., "gps_lon": ...}]}
        CSV format: camera_id,gps_lat,gps_lon,name,location
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Camera config not found: {config_path}")

        cameras: dict[str, CameraConfig] = {}

        if config_path.suffix.lower() == ".json":
            with open(config_path, encoding="utf-8") as fh:
                data = json.load(fh)
            for cam in data.get("cameras", []):
                cam_id = cam["camera_id"]
                cameras[cam_id] = CameraConfig(
                    camera_id=cam_id,
                    gps_lat=float(cam["gps_lat"]),
                    gps_lon=float(cam["gps_lon"]),
                    name=cam.get("description", cam.get("name", "")),
                    location=cam.get("description", ""),
                    metadata={
                        k: v for k, v in cam.items()
                        if k not in {"camera_id", "gps_lat", "gps_lon", "name", "location", "description"}
                    },
                )
        else:
            with open(config_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    cam_id = row["camera_id"]
                    cameras[cam_id] = CameraConfig(
                        camera_id=cam_id,
                        gps_lat=float(row["gps_lat"]),
                        gps_lon=float(row["gps_lon"]),
                        name=row.get("name", ""),
                        location=row.get("location", ""),
                        metadata={
                            k: v for k, v in row.items()
                            if k not in {"camera_id", "gps_lat", "gps_lon", "name", "location"}
                        },
                    )

        self.cameras.update(cameras)
        return cameras

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two GPS points (haversine)."""
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return EARTH_RADIUS_KM * c

    # ------------------------------------------------------------------
    # Spatiotemporal plausibility
    # ------------------------------------------------------------------

    def is_spatiotemporally_plausible(
        self,
        sighting1: dict[str, Any],
        sighting2: dict[str, Any],
    ) -> bool:
        """Return True if travelling between the two sightings at a
        physically possible speed (<= max_speed_kmh)."""
        t1 = sighting1["timestamp"]
        t2 = sighting2["timestamp"]

        # Ensure timestamps are comparable (assume datetime objects or posix)
        if hasattr(t1, "timestamp"):
            ts1 = t1.timestamp()
        else:
            ts1 = float(t1)
        if hasattr(t2, "timestamp"):
            ts2 = t2.timestamp()
        else:
            ts2 = float(t2)

        dt_seconds = abs(ts2 - ts1)
        if dt_seconds == 0:
            return True  # same instant, no distance constraint violated

        dist_km = self.compute_distance_km(
            sighting1["gps_lat"],
            sighting1["gps_lon"],
            sighting2["gps_lat"],
            sighting2["gps_lon"],
        )

        speed_kmh = (dist_km / dt_seconds) * 3600.0
        return speed_kmh <= self.max_speed_kmh

    # ------------------------------------------------------------------
    # Plate matching
    # ------------------------------------------------------------------

    def match_plates(
        self,
        plate_readings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compare every pair of sightings from different cameras.

        Returns a list of dicts with keys:
            plate_a, plate_b, camera_a, camera_b, similarity, plausible
        """
        results: list[dict[str, Any]] = []

        for i, sa in enumerate(plate_readings):
            for j, sb in enumerate(plate_readings):
                if j <= i:
                    continue
                if sa["camera_id"] == sb["camera_id"]:
                    continue

                plate_a = sa["plate"]
                plate_b = sb["plate"]

                # Edit-distance pre-filter
                edit_dist = Levenshtein.distance(plate_a, plate_b)
                if edit_dist > self.max_edit_distance:
                    continue

                similarity = fuzz.ratio(plate_a, plate_b)

                plausible = self.is_spatiotemporally_plausible(sa, sb)

                results.append(
                    {
                        "plate_a": plate_a,
                        "plate_b": plate_b,
                        "camera_a": sa["camera_id"],
                        "camera_b": sb["camera_id"],
                        "similarity": similarity,
                        "plausible": plausible,
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Sighting management
    # ------------------------------------------------------------------

    def add_sighting(
        self,
        plate: str,
        camera_id: str,
        gps_lat: float,
        gps_lon: float,
        timestamp: Any,
        confidence: float,
    ) -> None:
        """Append a single sighting to the internal store."""
        self.sightings.append(
            {
                "plate": plate,
                "camera_id": camera_id,
                "gps_lat": gps_lat,
                "gps_lon": gps_lon,
                "timestamp": timestamp,
                "confidence": confidence,
            }
        )

    # ------------------------------------------------------------------
    # Trajectory fusion
    # ------------------------------------------------------------------

    def fuse_trajectories(
        self,
        sightings: list[dict[str, Any]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Group sightings into trajectories keyed by canonical plate text.

        Matching is performed via :meth:`match_plates`; matched plates
        are unified under the plate text that appeared first
        (lowest index).  Each trajectory list is sorted by timestamp.
        """
        if sightings is None:
            sightings = self.sightings

        # Union-find for grouping plates that match across cameras
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for s in sightings:
            p = s["plate"]
            if p not in parent:
                parent[p] = p

        # Run pairwise matching to discover equivalences
        matches = self.match_plates(sightings)
        for m in matches:
            if m["plausible"]:
                union(m["plate_a"], m["plate_b"])

        # Build canonical name: the plate that appeared first in the list
        first_seen: dict[str, str] = {}
        for s in sightings:
            p = s["plate"]
            root = find(p)
            if root not in first_seen:
                first_seen[root] = p

        canonical_map = {p: first_seen[find(p)] for p in parent}

        # Group and sort
        trajectories: dict[str, list[dict[str, Any]]] = {}
        for s in sightings:
            canon = canonical_map[s["plate"]]
            trajectories.setdefault(canon, []).append(dict(s))

        for plate in trajectories:
            trajectories[plate].sort(key=lambda s: s["timestamp"])

        return trajectories
