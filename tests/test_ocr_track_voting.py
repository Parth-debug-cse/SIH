"""Unit tests for track-aware OCR voting.

Exercises ``PlateOCR.record_vote`` / ``_aggregate`` directly - no EasyOCR
inference is required, so the test is fast and deterministic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")

from src.ocr.reader import PlateOCR

# ---------------------------------------------------------------------------
# 1. Voting is isolated PER TRACK (two vehicles never share a vote)
# ---------------------------------------------------------------------------
ocr = PlateOCR(voting_window=5, history_ttl_seconds=30.0)
now = 1000.0

r1a = ocr.record_vote("KA01AB1234", 0.90, track_key=("cam_1", 1), now=now)
r1b = ocr.record_vote("KA01AB1234", 0.80, track_key=("cam_1", 1), now=now + 1)
r2a = ocr.record_vote("MH12CD5678", 0.70, track_key=("cam_1", 2), now=now + 2)

assert r1b["text"] == "KA01AB1234"
assert abs(r1b["confidence"] - 0.85) < 1e-9, r1b   # mean of the two contributing reads
assert r1b["voted"] is True
assert r1b["voted_count"] == 2
assert r2a["text"] == "MH12CD5678"
assert r2a["confidence"] == 0.70               # single read: not a vote yet
assert r2a["voted"] is False
assert r2a["voted_count"] == 1

# The car-2 history must not have borrowed car-1's readings.
print("[OK] Track isolation: technologies do not share votes")

# ---------------------------------------------------------------------------
# 2. Confidence always matches the returned (voted) text
# ---------------------------------------------------------------------------
ocr2 = PlateOCR(voting_window=5)
r = ocr2.record_vote("KA01AB1234", 0.90, track_key=("t", 1), now=2000.0)
r = ocr2.record_vote("KA01AB1234", 0.80, track_key=("t", 1), now=2001.0)
r = ocr2.record_vote("KAO1AB1234", 0.99, track_key=("t", 1), now=2002.0)  # minority misread

assert r["text"] == "KA01AB1234", r
assert abs(r["confidence"] - 0.85) < 1e-9, r    # mean(0.90, 0.80), NOT the 0.99 minority
assert r["raw_text"] == "KAO1AB1234"
assert r["raw_confidence"] == 0.99
assert r["voted"] is True
print("[OK] Confidence semantics: confidence matches voted text (raw kept separate)")

# ---------------------------------------------------------------------------
# 3. History expires per-track after the TTL
# ---------------------------------------------------------------------------
ocr3 = PlateOCR(history_ttl_seconds=30.0)
ocr3.record_vote("KA01AB1234", 0.9, track_key=("c", 1), now=3000.0)
ocr3.record_vote("KA01AB1234", 0.8, track_key=("c", 1), now=3005.0)
assert ocr3.record_vote("KA01AB1234", 0.7, track_key=("c", 1), now=3006.0)["voted_count"] == 3

# A long gap (> TTL) expires the old history: the old votes must not linger.
r = ocr3.record_vote("MH12CD5678", 0.7, track_key=("c", 2), now=3040.0)
r = ocr3.record_vote("KA01AB1234", 0.95, track_key=("c", 1), now=3041.0)
assert r["voted_count"] == 1, r                  # fresh history, not 3
assert r["voted"] is False
print("[OK] History TTL expiry enforced per track")

# ---------------------------------------------------------------------------
# 4. The number of live track histories is bounded (LRU eviction)
# ---------------------------------------------------------------------------
ocr4 = PlateOCR(max_track_histories=2)
base = 4000.0
for i in range(4):
    ocr4.record_vote("PLATE%d" % i, 0.8, track_key=("cam", i), now=base + i)
assert len(ocr4._histories) <= 2, len(ocr4._histories)
print("[OK] Live history bound respected (<= %d histories)" % ocr4.max_track_histories)

# ---------------------------------------------------------------------------
# 5. clear_history removes only the requested track
# ---------------------------------------------------------------------------
ocr5 = PlateOCR()
ocr5.record_vote("A0000", 0.8, track_key=("c", 1), now=5000.0)
ocr5.record_vote("B0000", 0.8, track_key=("c", 2), now=5001.0)
ocr5.clear_history(track_key=("c", 1))
assert ("c", 1) not in ocr5._histories
assert ("c", 2) in ocr5._histories
ocr5.clear_history()
assert len(ocr5._histories) == 0
print("[OK] clear_history() scope respected")

print("\nOCR TRACK-VOTING UNIT TESTS PASSED")