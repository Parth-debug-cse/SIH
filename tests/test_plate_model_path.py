"""Regression test: no HF cache blob path may ever reach Ultralytics.

Covers the live Colab failure where `resolve_plate_model_from_hub()`
returned a stable `.pt` path but plate inference still received
`.../blobs/<hash>` and YOLO raised
`TypeError: model='.../blobs/...' is not a supported model format`.

Uses a fake Hugging Face hub layout in a tmp dir — no network, no GPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import tempfile

from src.detection import detector as det_mod
from src.detection.detector import (
    PlateDetector,
    _stabilize_pt_path,
)

WEIGHTS = b"fake-plate-weights-payload" * 100


def make_fake_hub(with_snapshot=True):
    tmp = Path(tempfile.mkdtemp(prefix="fake_hub_"))
    blobs = tmp / "hub" / "models--Koushim--yolov8-license-plate-detection" / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / "2d958618abc123"
    blob.write_bytes(WEIGHTS)
    snap = None
    if with_snapshot:
        snap = tmp / "hub" / "models--Koushim--yolov8-license-plate-detection" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "best.pt").write_bytes(WEIGHTS)
    return blob


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


# 1. Blob stabilizes to a real .pt (via snapshot sibling), never a blobs path.
blob = make_fake_hub(with_snapshot=True)
stable = _stabilize_pt_path(blob, "best.pt")
check("stabilized suffix is .pt", stable.suffix == ".pt")
check("no /blobs/ segment in stabilized path", "blobs" not in stable.parts)
check("stabilized file non-empty", stable.stat().st_size > 0)

# 2. The init guard accepts the blob and yields a .pt path.
guarded = PlateDetector._require_pt_weights(blob)
check("guard output is .pt", guarded.suffix == ".pt")
check("guard output has no /blobs/", "blobs" not in guarded.parts)

# 3. _load_yolo refuses a blob loudly instead of Ultralytics' opaque TypeError.
try:
    PlateDetector._load_yolo(blob)
    check("_load_yolo refuses blob", False)
except RuntimeError as exc:
    check("_load_yolo refuses blob", "non-.pt" in str(exc))

# 4. Full constructor path: capture what would be handed to YOLO.
captured = {}
orig_load = PlateDetector._load_yolo

def spy(path):
    captured.setdefault("paths", []).append(Path(path))
    return object()

PlateDetector._load_yolo = staticmethod(spy)
try:
    PlateDetector(
        vehicle_model_path="yolov8n.pt",
        plate_model_path=str(blob),
        device="cpu",
    )
finally:
    PlateDetector._load_yolo = staticmethod(orig_load)

plate_paths = [p for p in captured["paths"] if "yolov8" not in p.name.lower()]
check("constructor passed plate path to loader", len(plate_paths) == 1)
check("constructor plate path ends .pt", plate_paths[0].suffix == ".pt")
check("constructor plate path has no /blobs/", "blobs" not in plate_paths[0].parts)

print("\nPLATE MODEL PATH REGRESSION TESTS PASSED")
