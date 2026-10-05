"""Phase-2 tests — MOT + Re-ID, no weights / GPU / scipy needed.

Proves the money requirement: when plate OCR fails in frame 2 (no text,
displaced box), the Track ID is maintained by cosine similarity of the
visual Re-ID embedding.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infrastructure.reid.vehicle_reid import (  # noqa: E402
    VehicleReID,
    cosine_similarity_matrix,
    select_reid_device,
)
from src.infrastructure.tracking.mot_tracker import (  # noqa: E402
    ReIDTracker,
    ReIDTrackerConfig,
)


def _emb(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / float(np.linalg.norm(v))


def test_device_probe_never_raises() -> None:
    assert select_reid_device() in ("mps", "cpu")


def test_cosine_similarity_identity() -> None:
    a = _emb(0)
    b = a.copy()
    c = _emb(99)
    sims = cosine_similarity_matrix(np.stack([a]), np.stack([b, c]))
    assert sims.shape == (1, 2)
    assert abs(float(sims[0, 0]) - 1.0) < 1e-5
    assert float(sims[0, 0]) > float(sims[0, 1])


def test_track_birth_and_confirm() -> None:
    tr = ReIDTracker(ReIDTrackerConfig(min_hits=2, max_age=5))
    e = _emb(1)
    d = {"bbox": [10.0, 10.0, 60.0, 60.0], "confidence": 0.9,
         "class_name": "car", "embedding": e}
    assert tr.update([dict(d)]) == []  # tentative first sighting
    out = tr.update([dict(d, bbox=[12.0, 11.0, 62.0, 61.0])])
    assert len(out) == 1 and out[0]["track_id"] == 1
    assert out[0]["embedding"] is not None  # embedding bound to the ID


def test_ocr_failure_holds_track_id_via_embedding() -> None:
    """Frame 1: plate read OK. Frame 2: OCR fails AND box jumps far away
    (occlusion — IoU ~ 0 with prediction). Same visual embedding must
    revive the SAME track_id through Stage-2 appearance matching."""
    tr = ReIDTracker(ReIDTrackerConfig(
        min_hits=1, max_age=10, iou_thresh=0.3, reid_thresh=0.4))
    car = _emb(7)

    t1 = tr.update([{"bbox": [10.0, 10.0, 60.0, 60.0], "confidence": 0.9,
                      "class_name": "car", "embedding": car,
                      "ocr_text": "MH12AB1234", "ocr_ok": True}])
    assert t1[0]["track_id"] == 1

    # Frame 2: OCR failed, box displaced (no motion overlap at all).
    t2 = tr.update([{"bbox": [300.0, 300.0, 350.0, 350.0], "confidence": 0.8,
                      "class_name": "car", "embedding": car.copy(),
                      "ocr_text": "", "ocr_ok": False}])
    assert len(t2) == 1, f"track lost on OCR failure: {t2}"
    assert t2[0]["track_id"] == 1, f"ID switch on OCR failure: {t2}"
    # Embedding stays bound (EMA-updated, still normalized).
    e = t2[0]["embedding"]
    assert e is not None and abs(float(np.linalg.norm(e)) - 1.0) < 1e-4


def test_different_vehicle_gets_new_id() -> None:
    tr = ReIDTracker(ReIDTrackerConfig(
        min_hits=1, max_age=10, iou_thresh=0.3, reid_thresh=0.4))
    tr.update([{"bbox": [10.0, 10.0, 60.0, 60.0], "confidence": 0.9,
                 "class_name": "car", "embedding": _emb(7)}])
    out = tr.update([{"bbox": [300.0, 300.0, 350.0, 350.0], "confidence": 0.85,
                       "class_name": "car", "embedding": _emb(1234)}])
    assert len(out) == 1 and out[0]["track_id"] == 2


def test_lost_track_revival_within_max_age() -> None:
    tr = ReIDTracker(ReIDTrackerConfig(
        min_hits=1, max_age=5, iou_thresh=0.3, reid_thresh=0.4))
    car = _emb(21)
    tr.update([{"bbox": [10.0, 10.0, 60.0, 60.0], "confidence": 0.9,
                 "class_name": "car", "embedding": car}])
    tr.update([])  # gap frame: track goes un-updated but is kept
    tr.update([])
    out = tr.update([{"bbox": [200.0, 200.0, 250.0, 250.0], "confidence": 0.8,
                       "class_name": "car", "embedding": car.copy()}])
    assert out and out[0]["track_id"] == 1


def test_frame_crop_embedding_path() -> None:
    """End-to-end without precomputed vectors: crops -> ReID -> bind."""
    reid = VehicleReID()  # histogram/numpy fallback, no weights needed
    tr = ReIDTracker(ReIDTrackerConfig(min_hits=1, max_age=5), reid=reid)
    red = np.zeros((480, 640, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)  # BGR red car
    out = tr.update([{"bbox": [10.0, 10.0, 60.0, 60.0],
                      "confidence": 0.9, "class_name": "car"}], frame=red)
    assert out and out[0]["embedding"] is not None
    # Same red car, displaced box, no OCR -> same ID via appearance.
    out2 = tr.update([{"bbox": [400.0, 400.0, 450.0, 450.0],
                       "confidence": 0.8, "class_name": "car"}], frame=red)
    assert out2 and out2[0]["track_id"] == out[0]["track_id"]


def test_async_update_off_loop() -> None:
    async def _run() -> None:
        tr = ReIDTracker(ReIDTrackerConfig(min_hits=1))
        out = await tr.aupdate(
            [{"bbox": [5.0, 5.0, 50.0, 50.0], "confidence": 0.9,
              "class_name": "bus", "embedding": _emb(3)}])
        assert out and out[0]["track_id"] == 1

    asyncio.run(_run())


if __name__ == "__main__":
    test_device_probe_never_raises()
    test_cosine_similarity_identity()
    test_track_birth_and_confirm()
    test_ocr_failure_holds_track_id_via_embedding()
    test_different_vehicle_gets_new_id()
    test_lost_track_revival_within_max_age()
    test_frame_crop_embedding_path()
    test_async_update_off_loop()
    print("PHASE2_OK")
