"""Unit test: per-position confidence-weighted fusion + position-aware correction.

Synthetic noisy observations only — proves the voting logic converges to the
correct plate without running any model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr.reader import fuse_track_observations, position_correct, looks_like_plate


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


# 1. Noisy multi-frame track converges to the true plate.
obs = [
    ("KA01AB1234", 0.91),
    ("KA01AB1234", 0.84),
    ("KA01AB1284", 0.51),   # single-char OCR error, low confidence
    ("KA01AB1234", 0.76),
    ("KA01AB123", 0.60),    # truncated read: different length, must not vote
]
res = fuse_track_observations(obs)
print("fusion result:", res)
check("canonical plate correct", res["canonical_plate"] == "KA01AB1234")
check("method is per-position", res["method"] == "per_position_weighted")
check("observation count uses compatible group", res["observation_count"] == 4)
check("aggregate confidence sane", 0.0 < res["aggregate_confidence"] <= 1.0)

# 2. Empty input never hallucinates.
res0 = fuse_track_observations([])
check("empty -> empty", res0["canonical_plate"] == "" and res0["observation_count"] == 0)

# 3. Position-aware correction: letter O inside the digit block -> 0.
check("O-in-digits fixed", position_correct("KA01AB123O") == "KA01AB1230")
# Digit 8 inside the state-code letters -> B.
check("8-in-state fixed", position_correct("K801AB1234") == "KB01AB1234")
# Already-valid plates are untouched.
check("valid untouched", position_correct("MH12CD5678") == "MH12CD5678")
# Garbage is returned unchanged, never hallucinated into a plate.
check("garbage unchanged", position_correct("HABILNA") == "HABILNA"
      and not looks_like_plate(position_correct("HABILNA")))

print("\nPER-POSITION VOTING UNIT TESTS PASSED")
