"""Regression test: fastplate track-level fusion via the EXISTING fusion.

Proves:
1. Noisy fastplate observations on one track fuse deterministically with
   `fuse_track_observations` (position-aware, confidence-weighted).
2. Logging observations NEVER changes `record_vote`/`_aggregate` semantics
   (identical results with and without interleaved logging).
3. Source filtering isolates backends (easyocr view unaffected by fastplate).
4. Verdicts never claim correctness (`verified` is always False).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr.reader import PlateOCR


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


TRACK = ("cam_wa", 3)
NOISY = [
    ("40GD3030", 0.71, 10),
    ("40GD7070", 0.68, 12),
    ("40GD7030", 0.74, 14),
    ("40GD3030", 0.66, 16),
    ("30GD7030", 0.60, 18),
]

# 1. Noisy fastplate fusion on one track.
ocr = PlateOCR()
for text, conf, frame in NOISY:
    ocr.log_observation(text, conf, TRACK, source="fastplate", frame=frame)
rep = ocr.track_fusion_report(TRACK, source="fastplate")
print("report:", {k: rep[k] for k in ("n_observations", "fused", "fused_confidence",
                                      "method", "regex_pass", "verdict", "verified")})
check("5 observations", rep["n_observations"] == 5)
check("deterministic fused string", rep["fused"] == "40GD7030")
check("method per-position", rep["method"] == "per_position_weighted")
check("non-plate verdict", rep["verdict"] == "format_rejected")
check("never verified", rep["verified"] is False)

# 1b. Fake-state-code singleton is rejected by the whitelist (never "close").
ocr.log_observation("GD75000", 0.587, ("cam_wa", 9), source="fastplate", frame=20)
rep2 = ocr.track_fusion_report(("cam_wa", 9), source="fastplate")
check("singleton method", rep2["method"] == "single")
check("fake state code rejected", rep2["regex_pass"] is False)
check("verdict names the cause", rep2["verdict"] == "invalid_state_code"
      and rep2["verified"] is False)

# 2. Voting semantics byte-identical with/without logging.
a, b = PlateOCR(), PlateOCR()
for text, conf in [("KA01AB1234", 0.8), ("KA01AB1234", 0.7), ("KA01AB1284", 0.4)]:
    ra = a.record_vote(text, conf, track_key=TRACK)
    b.log_observation(text, conf, TRACK, source="fastplate", frame=1)
    rb = b.record_vote(text, conf, track_key=TRACK)
check("votes identical despite logging", ra == rb)

# 3. Source isolation.
ocr.log_observation("KA01AB1234", 0.9, TRACK, source="easyocr", frame=11)
re_ = ocr.track_fusion_report(TRACK, source="easyocr")
rf = ocr.track_fusion_report(TRACK, source="fastplate")
check("easyocr view isolated", re_["n_observations"] == 1
      and re_["fused"] == "KA01AB1234")
check("fastplate view unchanged", rf["n_observations"] == 5)

# 4. Empty track.
re0 = ocr.track_fusion_report(("cam_wa", 999), source="fastplate")
check("empty verdict", re0["verdict"] == "no_observations"
      and re0["fused"] == "")

print("\nFASTPLATE TRACK-FUSION TESTS PASSED")
