"""Regression test: candidate-ledger bookkeeping (diagnostic-only).

Proves every evaluated plate candidate produces a complete ledger row
(frame, track, bbox, det conf, crop size, aspect, bumper verdict, raw/pre
OCR outputs, corrected text, regex result, final reason), that the CSV
persists, and that the summary prints distributions — without running any
model (uses PipelineRunner.__new__ with stubbed attributes).
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline_runner import PipelineRunner


def make_runner():
    r = PipelineRunner.__new__(PipelineRunner)
    r.camera_id = "cam_test"
    r.min_ocr_confidence = 0.50
    r.detector = SimpleNamespace(plate_conf_threshold=0.25)
    r._diag = {
        "candidate_boxes": 0, "aspect_rejected": 0, "position_rejected": 0,
        "tiny_rejected": 0, "detector_conf_rejected": 0, "ocr_attempted": 0,
        "ocr_empty": 0, "ocr_low_conf": 0, "regex_rejected": 0, "accepted": 0,
    }
    r._ledger = []
    r._debug_saved = {}
    r.debug_crops_dir = Path(tempfile.mkdtemp(prefix="ledger_test_"))
    return r


def sample_row(reason="ocr_low_conf"):
    return {
        "frame": 4, "track_id": 7, "det_source": "in_crop",
        "plate_bbox": [100, 200, 220, 244], "det_conf": 0.312,
        "det_threshold": 0.25, "crop_w": 138, "crop_h": 50, "aspect": 2.76,
        "bumper_passed": True, "ocr_raw_text": "KAO1AB1234", "ocr_raw_conf": 0.41,
        "ocr_pre_text": "KAO1AB1234", "ocr_pre_conf": 0.44,
        "accept_text": "KAO1AB1234", "accept_conf": 0.44,
        "corrected": "KA01AB1234", "regex_pass": False,
        "ocr_threshold": 0.50, "variant_error": "", "final_reason": reason,
    }


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


r = make_runner()

# 1. Row recorded with every required field (spec item 1).
r._record_candidate(sample_row())
check("ledger has 1 row", len(r._ledger) == 1)
required = ["frame", "track_id", "plate_bbox", "det_conf", "crop_w", "crop_h",
            "aspect", "bumper_passed", "ocr_raw_text", "ocr_raw_conf",
            "ocr_pre_text", "ocr_pre_conf", "corrected", "regex_pass",
            "final_reason"]
check("row has all required fields",
      all(k in r._ledger[0] for k in required))

# 2. CSV persists with header + row.
r._record_candidate(sample_row(reason="accepted"))
csv_path = r._save_ledger_csv()
check("csv written", csv_path is not None and os.path.exists(csv_path))
header = open(csv_path, encoding="utf-8").readline()
check("csv header has det_conf + final_reason",
      "det_conf" in header and "final_reason" in header)

# 3. Summary prints distributions, thresholds, gate survival, reasons.
r._diag.update({"candidate_boxes": 2, "ocr_low_conf": 1, "accepted": 1,
                "ocr_attempted": 2, "ocr_empty": 0, "regex_rejected": 0,
                "aspect_rejected": 0, "position_rejected": 0,
                "tiny_rejected": 0, "detector_conf_rejected": 0})
buf = io.StringIO()
with redirect_stdout(buf):
    r._print_ledger_summary()
out = buf.getvalue()
for needle in ["0.25", "0.5", "Plate-detector conf", "OCR accept-conf",
               "ocr_low_conf", "accepted",
               "candidate_boxes=2", "below_gate", "gate_cleared"]:
    check(f"summary mentions '{needle}'", needle in out)

print("\nCANDIDATE LEDGER TESTS PASSED")
