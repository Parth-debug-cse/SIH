"""Unit-level tests for the cross-camera fusion engine (no DB, no video).

Verifies the temporal / spatial hard constraints, multi-signal scoring and
the guard against transitive (union-find style) false merges.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")

import time

from src.fusion.reid import (
    CrossCameraFusion,
    CO_LOCATED_KM,
    DEFAULT_MAX_SPEED_KMH,
    DEFAULT_MAX_EDIT_DISTANCE,
)

CONFIG = "data/calibration/cameras.json"

CAM1 = {"camera_id": "cam_1", "gps_lat": 12.9758, "gps_lon": 77.6082}
CAM2 = {"camera_id": "cam_2", "gps_lat": 12.9768, "gps_lon": 77.6050}
CAM3 = {"camera_id": "cam_3", "gps_lat": 12.9745, "gps_lon": 77.6095}


def make_sighting(plate, cam, ts, conf=0.95, cls="car", track=None, direction=None):
    return {
        "plate": plate,
        "camera_id": cam["camera_id"],
        "gps_lat": cam["gps_lat"],
        "gps_lon": cam["gps_lon"],
        "timestamp": ts,
        "confidence": conf,
        "vehicle_class": cls,
        "track_id": track,
        "direction": direction,
    }


fusion = CrossCameraFusion(cameras_config_path=CONFIG, max_speed_kmh=DEFAULT_MAX_SPEED_KMH)
now = time.time()


def find_by_plate(trajs, plate):
    """Return the single trajectory whose canonical plate equals *plate*."""
    matches = [t for t in trajs if t["canonical_plate"] == plate]
    return matches[0] if len(matches) == 1 else matches


print("[OK] Fusion initialized with %d cameras" % len(fusion.cameras))

# ---------------------------------------------------------------------------
# 1. Chronological ordering is enforced (reversed order rejected)
# ---------------------------------------------------------------------------
s_late = make_sighting("KA01AB1234", CAM1, now)
s_early = make_sighting("KA01AB1234", CAM2, now - 30)
assoc = fusion.associate_sightings(s_late, s_early)
assert not assoc.accepted, "Reversed temporal order must be rejected"
assert "elapsed" in assoc.hard_reject_reason
assert fusion.is_spatiotemporally_plausible(s_late, s_early) is False
print("[OK] Reversed temporal order rejected")

# ---------------------------------------------------------------------------
# 2. Zero elapsed time is NOT automatically valid
# ---------------------------------------------------------------------------
s1 = make_sighting("KA01AB1234", CAM1, now + 100)
s2 = make_sighting("KA01AB1234", CAM3, now + 100)  # same instant, ~0.45km apart
assoc = fusion.associate_sightings(s1, s2)
assert not assoc.accepted, "Zero elapsed time between distant cameras must be rejected"
assert fusion.is_spatiotemporally_plausible(s1, s2) is False
print("[OK] Zero-elapsed-time edge case rejected (no longer always valid)")

# ---------------------------------------------------------------------------
# 3. Physically impossible travel rejected (10 km / 2 s -> > 150 km/h)
# ---------------------------------------------------------------------------
far1 = make_sighting("KA01AB1234", {"camera_id": "cam_a", "gps_lat": 12.0, "gps_lon": 77.0}, now + 200)
far2 = make_sighting("KA01AB1234", {"camera_id": "cam_b", "gps_lat": 12.1, "gps_lon": 77.0}, now + 202)
d = fusion.compute_distance_km(12.0, 77.0, 12.1, 77.0)
assert d > 5, "test cameras must be > 5 km apart"
assoc = fusion.associate_sightings(far1, far2)
assert not assoc.accepted
assert "speed" in assoc.hard_reject_reason
print("[OK] Impossible travel rejected (dist=%.1f km in 2s)" % d)


# ---------------------------------------------------------------------------
# 4. Valid cross-camera trip fuses into one ordered trajectory
# ---------------------------------------------------------------------------
t0 = now + 300
fusion.sightings.clear()
fusion.add_sighting("KA01AB1234", "cam_1", CAM1["gps_lat"], CAM1["gps_lon"], t0, 0.95)
fusion.add_sighting("KA01AB1234", "cam_2", CAM2["gps_lat"], CAM2["gps_lon"], t0 + 30, 0.88)
fusion.add_sighting("KA01AB1234", "cam_3", CAM3["gps_lat"], CAM3["gps_lon"], t0 + 60, 0.91)

trajs = fusion.fuse_trajectories()
traj = find_by_plate(trajs, "KA01AB1234")
route = [s["camera_id"] for s in traj["sightings"]]
assert route == ["cam_1", "cam_2", "cam_3"], route
assert traj["trajectory_id"] != traj["canonical_plate"]
print("[OK] Valid trip -> one ordered trajectory: %s" % " -> ".join(route))

# Speed of the first leg (cam_1 -> cam_2, ~0.36 km in 30 s -> ~44 km/h)
leg = fusion.associate_sightings(
    make_sighting("KA01AB1234", CAM1, t0),
    make_sighting("KA01AB1234", CAM2, t0 + 30),
)
assert leg.accepted
assert 35.0 < leg.speed_kmh < 55.0, leg.speed_kmh
assert leg.association_score >= fusion.min_association_score
print("[OK] Speed estimate valid: %.1f km/h (score=%.2f)" % (leg.speed_kmh, leg.association_score))

# ---------------------------------------------------------------------------
# 5. Letzten: transitive false merge is prevented (weak intermediate)
# ---------------------------------------------------------------------------
# A lon/greedy chain of plates that drift apart by 1 edit per step, so the
# endpoints are > max_edit_distance apart even though each adjacent pair is
# acceptable.  A naive union-find would merge all four; the canonical-plate
# constraint must keep the last vehicle separate.
t1 = now + 500
drift = [
    make_sighting("AB12CD34", CAM1, t1, track=1),
    make_sighting("AB13CD34", CAM2, t1 + 60, track=2),
    make_sighting("AB13CE34", CAM3, t1 + 120, track=3),
    make_sighting("XX13CE34", {"camera_id": "cam_4", "gps_lat": 12.977, "gps_lon": 77.602}, t1 + 180, track=4),
]
endpoint_assoc = fusion.associate_sightings(drift[0], drift[3])
assert not endpoint_assoc.accepted, "endpoints must be beyond plate edit distance"

fusion.sightings = list(drift)
merged = fusion.fuse_trajectories()
assert len(merged) >= 2, "Weak intermediate must not merge unrelated vehicles: %s" % merged
print("[OK] Transitive false merge prevented -> %d trajectory(ies)" % len(merged))
for t in merged:
    print("      %s (%s): %s" % (t["trajectory_id"], t["canonical_plate"],
                                 " -> ".join(s["camera_id"] for s in t["sightings"])))

# ---------------------------------------------------------------------------
# 6. Same-camera, different track (two different vehicles) rejected
# ---------------------------------------------------------------------------
a1 = make_sighting("KA01AB1234", CAM1, now + 700, track=10)
a2 = make_sighting("KA01AB1234", CAM1, now + 701, track=11)
assoc = fusion.associate_sightings(a1, a2)
assert not assoc.accepted
assert "track" in assoc.hard_reject_reason
print("[OK] Same camera + different track rejected")
# ---------------------------------------------------------------------------
# 7. Reverse-ordered same-plate sightings reorder chronologically,
#    never producing a backward transition
# ---------------------------------------------------------------------------
fusion2 = CrossCameraFusion(cameras_config_path=CONFIG)
fusion2.sightings = [
    make_sighting("MH12CD5678", CAM1, t0 + 400),
    make_sighting("MH12CD5678", CAM2, t0 + 380),  # arrives second, timestamp earlier
]
rev_trajs = fusion2.fuse_trajectories()
rev_traj = find_by_plate(rev_trajs, "MH12CD5678")
route = [s["camera_id"] for s in rev_traj["sightings"]]
assert route == ["cam_2", "cam_1"], route
# The pair must be traversed chronologically (cam_2 first, then cam_1).
assert len(rev_traj["sightings"]) == 2
print("[OK] Reverse-ordered input -> chronological traversal: cam_2 -> cam_1")


# ---------------------------------------------------------------------------
# 8. match_plates API preserved
# ---------------------------------------------------------------------------
fusion3 = CrossCameraFusion(cameras_config_path=CONFIG)
fusion3.sightings = [
    make_sighting("KA01AB1234", CAM1, t0 + 800),
    make_sighting("KAO1AB1234", CAM2, t0 + 830),   # OCR 0/O confusion
]
matches = fusion3.match_plates(fusion3.sightings)
assert len(matches) == 1
assert matches[0]["plausible"] is True
assert matches[0]["similarity"] >= 90.0
print("[OK] match_plates preserved (0/O confusion tolerated, similarity=%.1f)" % matches[0]["similarity"])

# ---------------------------------------------------------------------------
# 9. REGRESSION: same camera + different track + identical OCR plate must
#    stay two trajectories (association rejected, no plate-keyed merge)
# ---------------------------------------------------------------------------
t9 = now + 1000
reg1 = [
    make_sighting("KA01AB1234", CAM1, t9, track=1),
    make_sighting("KA01AB1234", CAM1, t9 + 1, track=2),
]
assoc = fusion.associate_sightings(reg1[0], reg1[1])
assert not assoc.accepted, "same camera + different track must be rejected"
assert "track" in assoc.hard_reject_reason
fusion.sightings = list(reg1)
reg_trajs = fusion.fuse_trajectories()
assert len(reg_trajs) == 2, "must produce exactly 2 trajectories, got: %s" % reg_trajs
for t in reg_trajs:
    assert len(t["sightings"]) == 1
    assert t["sightings"][0]["track_id"] in (1, 2)
    assert len({s["track_id"] for s in t["sightings"]}) == 1, "trajectory leaked the other track"
    assert not CrossCameraFusion.trajectory_violates_track_integrity(t["sightings"])
assert reg_trajs[0]["trajectory_id"] != reg_trajs[1]["trajectory_id"]
assert reg_trajs[0]["canonical_plate"] == reg_trajs[1]["canonical_plate"] == "KA01AB1234"
print("[OK] Same camera, different track, identical plate -> 2 trajectories, "
      "ids=%s/%s (identity != plate)" % (reg_trajs[0]["trajectory_id"], reg_trajs[1]["trajectory_id"]))
print("      integrity_violates helper: %r/%r" % (
    CrossCameraFusion.trajectory_violates_track_integrity(reg_trajs[0]["sightings"]),
    CrossCameraFusion.trajectory_violates_track_integrity(reg_trajs[1]["sightings"])))

# ---------------------------------------------------------------------------
# 10. REGRESSION: two realistic, well-separated chains that share the SAME
#     canonical plate must stay two trajectories
# ---------------------------------------------------------------------------
t10a = now + 1300
t10b = t10a + 1000  # beyond max_gap_seconds window
chain_a = [
    make_sighting("KA01AB1234", CAM1, t10a, track=1),
    make_sighting("KA01AB1234", CAM2, t10a + 30, track=2),
    make_sighting("KA01AB1234", CAM3, t10a + 60, track=3),
]
chain_b = [
    make_sighting("KA01AB1234", CAM1, t10b, track=4),
    make_sighting("KA01AB1234", CAM2, t10b + 30, track=5),
    make_sighting("KA01AB1234", CAM3, t10b + 60, track=6),
]
fusion.sightings = chain_a + chain_b
same_plate = fusion.fuse_trajectories()
assert len(same_plate) == 2, "separate chains sharing a canonical plate must NOT merge: %s" % same_plate
for t in same_plate:
    assert t["canonical_plate"] == "KA01AB1234"
    route = [s["camera_id"] for s in t["sightings"]]
    assert route == ["cam_1", "cam_2", "cam_3"], route
    assert len(t["sightings"]) == 3
    assert t["trajectory_id"] != "KA01AB1234"
print("[OK] Same canonical plate, separate chains -> %d separate trajectories (ids %s, %s)" % (
    len(same_plate), same_plate[0]["trajectory_id"], same_plate[1]["trajectory_id"]))

# ---------------------------------------------------------------------------
# 11. Same camera + same track is fine: stays a single trajectory
# ---------------------------------------------------------------------------
fusion.sightings = [
    make_sighting("KA01AB1234", CAM1, now + 1500, track=7),
    make_sighting("KA01AB1234", CAM1, now + 1510, track=7),
]
one_traj = fusion.fuse_trajectories()
assert len(one_traj) == 1, one_traj
assert len(one_traj[0]["sightings"]) == 2
assert len({s["track_id"] for s in one_traj[0]["sightings"]}) == 1
assert not CrossCameraFusion.trajectory_violates_track_integrity(one_traj[0]["sightings"])
print("[OK] Same camera + same track -> single trajectory")

# ---------------------------------------------------------------------------
# 12. Different plates within edit distance still fuse valid transitions
# ---------------------------------------------------------------------------
fusion.sightings = [
    make_sighting("KA01AB1234", CAM1, now + 1600, track=8),
    make_sighting("KA01AB123B", CAM2, now + 1630, track=9),  # 1 edit (4 -> B)
]
edit_trajs = fusion.fuse_trajectories()
assert len(edit_trajs) == 1, edit_trajs
assert edit_trajs[0]["canonical_plate"] == "KA01AB1234"
route = [s["camera_id"] for s in edit_trajs[0]["sightings"]]
assert route == ["cam_1", "cam_2"], route
print("[OK] 1-edit plate drift fuses into one trajectory (%s)" % " -> ".join(route))

# ---------------------------------------------------------------------------
# 13. Edge-case guards preserved after the identity refactor
# ---------------------------------------------------------------------------
# dt <= 0 must never fuse (already checked in section 1, re-check via the API)
fusion.sightings = [
    make_sighting("KA01AB1234", CAM1, now + 1700, track=10),
    make_sighting("KA01AB1234", CAM2, now + 1700, track=11),  # identical instant
]
edge_trajs = fusion.fuse_trajectories()
assert len(edge_trajs) == 2, "zero-elapsed same-instant pair must yield 2 trajectories: %s" % edge_trajs
# Overspeed must never fuse (distant cameras, tiny gap)
overspeed = [
    make_sighting("KA01AB1234", {"camera_id": "cam_x", "gps_lat": 12.0, "gps_lon": 77.0}, now + 1800, track=12),
    make_sighting("KA01AB1234", {"camera_id": "cam_y", "gps_lat": 12.1, "gps_lon": 77.0}, now + 1802, track=13),
]
assoc = fusion.associate_sightings(overspeed[0], overspeed[1])
assert not assoc.accepted and "speed" in assoc.hard_reject_reason
fusion.sightings = overspeed
speed_trajs = fusion.fuse_trajectories()
assert len(speed_trajs) == 2, "overspeed pair must remain 2 trajectories: %s" % speed_trajs
print("[OK] Edge guards preserved: zero-elapsed and overspeed stay unfused")

print("\nFUSION ENGINE UNIT TESTS PASSED")