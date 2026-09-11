"""Cross-camera fusion / re-identification module.

Fuses ANPR plate sightings across different camera feeds into unified
vehicle trajectories.

The fusion logic intentionally combines several signals rather than raw
plate-string equality:

* plate string similarity (RapidFuzz, tolerant of OCR confusions)
* OCR confidence of the readings
* spatial / temporal plausibility (elapsed time, inter-camera distance,
  implied speed)
* vehicle class agreement
* camera compass-bearing alignment (soft proxy, never a hard reject)
* per-camera track identity (same camera, different track -> reject)

Chronological ordering is enforced: a transition ``A -> B`` requires
``timestamp_B > timestamp_A``.  Zero or negative elapsed time is rejected,
which also removes the unsafe "zero elapsed time is always valid" case.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

EARTH_RADIUS_KM = 6371.0

# ---------------------------------------------------------------------------
# Scoring configuration (documented in DECISIONS.md / VERIFICATION.md)
# ---------------------------------------------------------------------------

# Weight of each signal in the final association score.  Sum == 1.0.
PLATE_SIMILARITY_WEIGHT = 0.40
OCR_CONFIDENCE_WEIGHT = 0.20
SPATIOTEMPORAL_WEIGHT = 0.30
VEHICLE_ATTRIBUTE_WEIGHT = 0.10

# Hard constraints
DEFAULT_MAX_EDIT_DISTANCE = 2      # OCR confusions (0/O etc.) tolerated to here
DEFAULT_MAX_SPEED_KMH = 150.0      # a legal-ish travel speed upper bound
DEFAULT_MAX_GAP_SECONDS = 300.0    # ignore pairings more than 5 min apart
DEFAULT_MIN_ASSOCIATION_SCORE = 0.55

# Cameras closer than this (km) are treated as co-located: a small positive
# elapsed time is acceptable between them.
CO_LOCATED_KM = 0.05
# Within this many seconds, co-located cameras may observe the same vehicle.
CO_LOCATED_MAX_DT_SECONDS = 2.0


@dataclass
class CameraConfig:
    camera_id: str
    gps_lat: float
    gps_lon: float
    name: str = ""
    location: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Association:
    """Result of pairing two sightings."""

    sighting_a: dict[str, Any]
    sighting_b: dict[str, Any]
    accepted: bool
    hard_reject_reason: Optional[str] = None
    plate_similarity: float = 0.0
    ocr_confidence: float = 0.0
    elapsed_time: float = 0.0
    distance_km: float = 0.0
    speed_kmh: float = 0.0
    vehicle_class_match: Optional[bool] = None
    direction_compatibility: float = 1.0
    association_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plate_a": self.sighting_a.get("plate"),
            "plate_b": self.sighting_b.get("plate"),
            "camera_a": self.sighting_a.get("camera_id"),
            "camera_b": self.sighting_b.get("camera_id"),
            "similarity": self.plate_similarity * 100.0,
            "plausible": self.accepted and self.speed_kmh <= DEFAULT_MAX_SPEED_KMH,
            "accepted": self.accepted,
            "hard_reject_reason": self.hard_reject_reason,
            "ocr_confidence": self.ocr_confidence,
            "elapsed_time": self.elapsed_time,
            "distance_km": self.distance_km,
            "speed_kmh": self.speed_kmh,
            "vehicle_class_match": self.vehicle_class_match,
            "direction_compatibility": self.direction_compatibility,
            "association_score": self.association_score,
        }


class CrossCameraFusion:
    """Fuses plate sightings across camera feeds into unified trajectories."""

    def __init__(
        self,
        cameras_config_path: str | Path | None = None,
        max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
        max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
        min_association_score: float = DEFAULT_MIN_ASSOCIATION_SCORE,
    ) -> None:
        self.max_edit_distance = max_edit_distance
        self.max_speed_kmh = max_speed_kmh
        self.max_gap_seconds = max_gap_seconds
        self.min_association_score = min_association_score
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

        Extra keys (``pixel_to_meter_ratio``, ``compass_bearing``, ...) are
        preserved in ``CameraConfig.metadata`` for the analytics/fusion code.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Camera config not found: {config_path}")

        cameras: dict[str, CameraConfig] = {}
        excluded = {"camera_id", "gps_lat", "gps_lon", "name", "location", "description"}

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
                    metadata={k: v for k, v in cam.items() if k not in excluded},
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
                        metadata={k: v for k, v in row.items() if k not in excluded},
                    )

        self.cameras.update(cameras)
        return cameras

    def _bearing(self, camera_id: str) -> Optional[float]:
        cam = self.cameras.get(camera_id)
        if cam is None:
            return None
        raw = cam.metadata.get("compass_bearing")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

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
        c = 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
        return EARTH_RADIUS_KM * c

    @staticmethod
    def _initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Initial great-circle bearing from point 1 to point 2 (degrees)."""
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        dlon = lon2_r - lon1_r
        y = math.sin(dlon) * math.cos(lat2_r)
        x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    @staticmethod
    def _angular_diff(a: float, b: float) -> float:
        """Smallest angular difference in degrees."""
        diff = abs(a - b) % 360.0
        return diff if diff <= 180.0 else 360.0 - diff

    # ------------------------------------------------------------------
    # Temporal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _epoch(value: Any) -> float:
        """Normalise ``datetime`` or epoch-float timestamps to epoch seconds."""
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        return float(value)

    # ------------------------------------------------------------------
    # Spatiotemporal plausibility
    # ------------------------------------------------------------------

    def _speed_between(self, s1: dict[str, Any], s2: dict[str, Any]) -> tuple:
        """Return ``(dt_seconds, distance_km, speed_kmh)`` for an ordered pair.

        Requires ``s1`` to precede ``s2`` in time; otherwise ``dt_seconds``
        is negative and the caller must reject the transition.
        """
        ts1 = self._epoch(s1["timestamp"])
        ts2 = self._epoch(s2["timestamp"])
        dt = ts2 - ts1
        dist = self.compute_distance_km(
            s1["gps_lat"], s1["gps_lon"], s2["gps_lat"], s2["gps_lon"]
        )
        speed = (dist / dt) * 3600.0 if dt > 0 else float("inf")
        return dt, dist, speed

    def is_spatiotemporally_plausible(
        self,
        sighting1: dict[str, Any],
        sighting2: dict[str, Any],
    ) -> bool:
        """True if travelling *sighting1 -> sighting2* is physically possible.

        Rules:
        * ``elapsed = timestamp2 - timestamp1`` must be strictly positive
          (reverse ordering and zero-gap are rejected).
        * co-located cameras (within ``CO_LOCATED_KM``) are accepted for
          small positive gaps.
        * otherwise the implied speed must be <= ``max_speed_kmh``.
        """
        dt, dist, speed = self._speed_between(sighting1, sighting2)
        if dt <= 0:
            return False
        if dist <= CO_LOCATED_KM and dt <= CO_LOCATED_MAX_DT_SECONDS:
            return True
        return speed <= self.max_speed_kmh

    # ------------------------------------------------------------------
    # Association scoring
    # ------------------------------------------------------------------

    def _direction_compatibility(self, s1: dict[str, Any], s2: dict[str, Any]) -> float:
        """Soft proxy (0.5..1.0) based on camera compass-bearing alignment.

        Never a hard reject.  Returns 1.0 (neutral) when bearings or GPS are
        unavailable.
        """
        b1 = self._bearing(s1.get("camera_id", ""))
        b2 = self._bearing(s2.get("camera_id", ""))
        if b1 is None or b2 is None:
            return 1.0
        try:
            lat1, lon1 = float(s1["gps_lat"]), float(s1["gps_lon"])
            lat2, lon2 = float(s2["gps_lat"]), float(s2["gps_lon"])
            heading = self._initial_bearing_deg(lat1, lon1, lat2, lon2)
        except (KeyError, TypeError, ValueError):
            return 1.0
        d1 = self._angular_diff(b1, heading)
        d2 = self._angular_diff(b2, heading)
        diff = max(d1, d2)
        return 1.0 if diff <= 120.0 else 0.5

    def associate_sightings(
        self,
        sighting_a: dict[str, Any],
        sighting_b: dict[str, Any],
    ) -> Association:
        """Score the association between two sightings (order matters).

        The pair is interpreted as a transition ``A -> B``; chronological
        ordering is enforced (``timestamp_B > timestamp_A``).

        Hard rejection rules (score is unusable -> association rejected):
        * same camera but explicitly different track ids (two different
          vehicles cannot be the same vehicle),
        * plate edit distance beyond ``max_edit_distance``,
        * non-positive elapsed time (reverse / zero gap),
        * implied speed above ``max_speed_kmh``,
        * conflicting vehicle classes when both are present.
        """
        ts_a = self._epoch(sighting_a["timestamp"])
        ts_b = self._epoch(sighting_b["timestamp"])
        elapsed = ts_b - ts_a

        plate_a = str(sighting_a.get("plate", "")).strip().upper()
        plate_b = str(sighting_b.get("plate", "")).strip().upper()
        edit_dist = Levenshtein.distance(plate_a, plate_b)
        plate_similarity = fuzz.ratio(plate_a, plate_b) / 100.0

        # ---- hard rejections -------------------------------------------
        a_cam = sighting_a.get("camera_id")
        b_cam = sighting_b.get("camera_id")
        a_track = sighting_a.get("track_id")
        b_track = sighting_b.get("track_id")
        if a_cam == b_cam and a_track is not None and b_track is not None and a_track != b_track:
            return Association(sighting_a, sighting_b, False,
                               hard_reject_reason="same camera, different track ids")

        if edit_dist > self.max_edit_distance:
            return Association(sighting_a, sighting_b, False,
                               hard_reject_reason=f"plate edit distance {edit_dist} > {self.max_edit_distance}")

        if elapsed <= 0:
            return Association(
                sighting_a, sighting_b, False,
                hard_reject_reason=f"non-positive elapsed time ({elapsed:.1f}s); reverse/zero ordering rejected",
                plate_similarity=plate_similarity, elapsed_time=elapsed,
            )

        distance_km = self.compute_distance_km(
            sighting_a["gps_lat"], sighting_a["gps_lon"],
            sighting_b["gps_lat"], sighting_b["gps_lon"],
        )
        speed_kmh = (distance_km / elapsed) * 3600.0

        if distance_km > CO_LOCATED_KM and speed_kmh > self.max_speed_kmh:
            return Association(
                sighting_a, sighting_b, False,
                hard_reject_reason=f"implied speed {speed_kmh:.0f} km/h > {self.max_speed_kmh:.0f} km/h",
                plate_similarity=plate_similarity,
                elapsed_time=elapsed, distance_km=distance_km, speed_kmh=speed_kmh,
            )

        class_a = sighting_a.get("vehicle_class")
        class_b = sighting_b.get("vehicle_class")
        if class_a and class_b and str(class_a).lower() != str(class_b).lower():
            return Association(
                sighting_a, sighting_b, False,
                hard_reject_reason=f"vehicle class conflict {class_a} vs {class_b}",
                plate_similarity=plate_similarity,
                elapsed_time=elapsed, distance_km=distance_km, speed_kmh=speed_kmh,
                vehicle_class_match=False, ocr_confidence=self._pair_ocr_confidence(sighting_a, sighting_b),
            )

        # ---- signal terms ----------------------------------------------
        conf = self._pair_ocr_confidence(sighting_a, sighting_b)

        if distance_km <= CO_LOCATED_KM and elapsed <= CO_LOCATED_MAX_DT_SECONDS:
            spatiotemporal = 1.0
        else:
            speed_ratio = min(speed_kmh / self.max_speed_kmh, 1.0)
            spatiotemporal = 1.0 - 0.8 * speed_ratio
        spatiotemporal *= self._direction_compatibility(sighting_a, sighting_b)

        attr = 1.0
        class_match: Optional[bool] = None
        if class_a and class_b:
            class_match = str(class_a).lower() == str(class_b).lower()
            attr = 1.0 if class_match else 0.4

        score = (
            PLATE_SIMILARITY_WEIGHT * plate_similarity
            + OCR_CONFIDENCE_WEIGHT * conf
            + SPATIOTEMPORAL_WEIGHT * spatiotemporal
            + VEHICLE_ATTRIBUTE_WEIGHT * attr
        )

        return Association(
            sighting_a, sighting_b, True,
            plate_similarity=plate_similarity,
            ocr_confidence=conf,
            elapsed_time=elapsed,
            distance_km=distance_km,
            speed_kmh=speed_kmh,
            vehicle_class_match=class_match,
            direction_compatibility=self._direction_compatibility(sighting_a, sighting_b),
            association_score=round(score, 4),
        )

    @staticmethod
    def _pair_ocr_confidence(sighting_a: dict[str, Any], sighting_b: dict[str, Any]) -> float:
        confs = []
        for s in (sighting_a, sighting_b):
            c = s.get("confidence")
            if c is None:
                c = s.get("plate_confidence")
            if c is not None:
                confs.append(float(c))
        if not confs:
            return 1.0  # neutral
        return min(confs)

    # ------------------------------------------------------------------
    # Plate matching
    # ------------------------------------------------------------------

    def match_plates(
        self,
        plate_readings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compare every pair of sightings from different cameras.

        Returns a list of dicts with keys:
            plate_a, plate_b, camera_a, camera_b, similarity, plausible,
            accepted, association_score, hard_reject_reason, elapsed_time,
            distance_km, speed_kmh, ocr_confidence.
        """
        results: list[dict[str, Any]] = []
        for i, sa in enumerate(plate_readings):
            for j, sb in enumerate(plate_readings):
                if j <= i:
                    continue
                if sa["camera_id"] == sb["camera_id"]:
                    continue
                # Present the pair in chronological order for scoring.
                if self._epoch(sa["timestamp"]) <= self._epoch(sb["timestamp"]):
                    first, second = sa, sb
                else:
                    first, second = sb, sa
                assoc = self.associate_sightings(first, second)
                d = assoc.to_dict()
                # Preserve the original reported orientation of plate_a/plate_b.
                d["plate_a"] = sa["plate"]
                d["plate_b"] = sb["plate"]
                d["camera_a"] = sa["camera_id"]
                d["camera_b"] = sb["camera_id"]
                results.append(d)
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
        vehicle_class: Optional[str] = None,
        track_id: Optional[int] = None,
        direction: Optional[str] = None,
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
                "vehicle_class": vehicle_class,
                "track_id": track_id,
                "direction": direction,
            }
        )

    # ------------------------------------------------------------------
    # Trajectory fusion
    # ------------------------------------------------------------------

    def fuse_trajectories(
        self,
        sightings: list[dict[str, Any]] | None = None,
        max_gap_seconds: Optional[float] = None,
        min_association_score: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Group sightings into chronologically-fused trajectory objects.

        Strategy (trajectory-aware sequential association):
        1. Sightings are sorted by timestamp and consumed in order: each
           trajectory is grown forward from an unassigned seed.
        2. Among the accepted, in-window candidates the *earliest* sighting is
           chosen, so a route that visits intermediate cameras continuously is
           preferred over skipping them for a higher-scoring distant match.
        3. A candidate is added only when the association from the current
           tail is accepted *and* its plate is still within
           ``max_edit_distance`` of the trajectory's **canonical** (first)
           plate.  The canonical check prevents one weak intermediate match
           from chaining together two unrelated vehicles (union-find
           transitive-merge hazard).
        4. Each finished chain is validated against the same association
           rules and split at any weak/invalid internal link.

        Each returned trajectory is a dict with a **unique identity**, never
        the canonical plate string::

            {
                "trajectory_id": "trajectory_0",
                "canonical_plate": "KA01AB1234",
                "sightings": [ {...}, ... ],   # ordered chronologically
            }

        Two chains that share a canonical plate but represent different
        vehicles are kept as separate trajectories.  As a hard invariant, no
        trajectory ever contains two sightings from the same camera with
        different track IDs - even when their OCR plate strings are identical.
        """
        if sightings is None:
            sightings = self.sightings

        ordered = sorted(sightings, key=lambda s: self._epoch(s["timestamp"]))
        if not ordered:
            return []

        window = max_gap_seconds if max_gap_seconds is not None else self.max_gap_seconds
        min_score = (
            min_association_score
            if min_association_score is not None
            else self.min_association_score
        )

        n = len(ordered)
        available = [True] * n
        chains: list[list[dict[str, Any]]] = []

        while True:
            candidates_seed = [i for i in range(n) if available[i]]
            if not candidates_seed:
                break
            seed_i = min(candidates_seed)
            available[seed_i] = False
            chain = [ordered[seed_i]]
            canonical = str(ordered[seed_i]["plate"]).strip().upper()

            while True:
                tail = chain[-1]
                tail_ts = self._epoch(tail["timestamp"])
                best: Optional[tuple[Association, float, int]] = None
                for j in range(n):
                    if not available[j]:
                        continue
                    s = ordered[j]
                    s_ts = self._epoch(s["timestamp"])
                    gap = s_ts - tail_ts
                    if gap <= 0:
                        continue
                    if gap > window:
                        break  # ordered ascending: every later sighting exceeds the window too
                    if Levenshtein.distance(canonical, str(s["plate"]).strip().upper()) > self.max_edit_distance:
                        continue  # canonical plate coherence safeguard
                    assoc = self.associate_sightings(tail, s)
                    if not assoc.accepted or assoc.association_score < min_score:
                        continue
                    # Prefer the chronologically-earliest accepted candidate so
                    # intermediate cameras are visited in order.
                    if best is None or (s_ts, -assoc.association_score) < (best[1], -best[0].association_score):
                        best = (assoc, s_ts, j)
                if best is None:
                    break
                _, _, j = best
                chain.append(ordered[j])
                available[j] = False

            chains.append(chain)

        trajectories: list[dict[str, Any]] = []
        for chain in chains:
            for piece_raw in self._split_invalid_chain(chain):
                piece = self._dedup_sightings(
                    sorted(piece_raw, key=lambda s: self._epoch(s["timestamp"]))
                )
                if not piece:
                    continue
                # Hard invariant: one trajectory must never mix two different
                # track IDs on the same camera - regardless of plate text.
                for sub in self._split_integrity_violations(piece):
                    if not sub:
                        continue
                    canon = str(sub[0]["plate"]).strip().upper()
                    trajectories.append(
                        {
                            "trajectory_id": f"trajectory_{len(trajectories)}",
                            "canonical_plate": canon,
                            "sightings": sub,
                        }
                    )

        return trajectories

    def _dedup_sightings(self, sightings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop exact-duplicate rows (same plate/camera/timestamp/track)."""
        seen: set[tuple] = set()
        dedup: list[dict[str, Any]] = []
        for s in sightings:
            key = (
                str(s.get("plate", "")),
                str(s.get("camera_id", "")),
                round(self._epoch(s.get("timestamp", 0.0)), 6),
                s.get("track_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            dedup.append(s)
        return dedup

    def _split_integrity_violations(
        self,
        sightings: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Split a chain at any same-camera / different-track boundary.

        Guarantees the invariant that a single trajectory never contains two
        sightings from the same camera with different track IDs.  The rest of
        the chain (e.g. a continuous cross-camera route) is preserved intact.
        """
        pieces: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        seen_track: dict = {}
        for s in sightings:
            cam = s.get("camera_id")
            tid = s.get("track_id")
            prev = seen_track.get(cam) if cam is not None else None
            if cam is not None and tid is not None and prev is not None and prev != tid:
                pieces.append(current)
                current = [s]
                seen_track = {cam: tid}
                continue
            if cam is not None and tid is not None:
                seen_track[cam] = tid
            current.append(s)
        if current:
            pieces.append(current)
        return [p for p in pieces if p]

    @staticmethod
    def trajectory_violates_track_integrity(sightings: list[dict[str, Any]]) -> bool:
        """True if a trajectory mixes two track IDs on the same camera."""
        seen_track: dict = {}
        for s in sightings:
            cam = s.get("camera_id")
            tid = s.get("track_id")
            if cam is None or tid is None:
                continue
            if cam in seen_track and seen_track[cam] != tid:
                return True
            seen_track[cam] = tid
        return False

    def _split_invalid_chain(
        self,
        chain: list[dict[str, Any]],
        split_min_score: float = 0.40,
    ) -> list[list[dict[str, Any]]]:
        """Validate an assembled chain and split it at weak/invalid links.

        This is the post-construction validation safeguard that prevents a
        single weak intermediate from permanently merging two unrelated
        vehicles.
        """
        if len(chain) <= 2:
            pieces: list[list[dict[str, Any]]] = []
            current = list(chain)
            if len(current) >= 2:
                only = self.associate_sightings(current[0], current[1])
                if only.accepted and only.association_score >= split_min_score:
                    pieces.append(current)
                else:
                    pieces.append([current[0]])
                    pieces.append([current[1]])
            else:
                pieces.append(current)
            return pieces

        pieces = []
        current = [chain[0]]

        # Simple scan over adjacent pairs.
        for a, b in zip(chain, chain[1:]):
            assoc = self.associate_sightings(a, b)
            if assoc.accepted and assoc.association_score >= split_min_score:
                current.append(b)
            else:
                pieces.append(current)
                current = [b]
        pieces.append(current)

        # Strong negative evidence between a chain's endpoints -> split at the
        # weakest link (one weak intermediate must not merge two vehicles).
        final: list[list[dict[str, Any]]] = []
        for piece in pieces:
            if len(piece) < 3:
                final.append(piece)
                continue
            first = str(piece[0]["plate"]).strip().upper()
            last = str(piece[-1]["plate"]).strip().upper()
            endpoint_ok = Levenshtein.distance(first, last) <= self.max_edit_distance
            endpoint_assoc = self.associate_sightings(piece[0], piece[-1])
            if endpoint_ok and endpoint_assoc.accepted and endpoint_assoc.association_score >= split_min_score:
                final.append(piece)
                continue

            # Find and remove the weakest acceptable link.
            weakest_pos = -1
            weakest_score = float("inf")
            for k in range(len(piece) - 1):
                score = self.associate_sightings(piece[k], piece[k + 1]).association_score
                if score < weakest_score:
                    weakest_score = score
                    weakest_pos = k
            if weakest_pos <= 0:
                final.append(piece)
                continue
            final.append(piece[: weakest_pos + 1])
            final.append(piece[weakest_pos + 1:])

        return final