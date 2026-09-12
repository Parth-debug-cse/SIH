"""Fetch license-plate detection weights from Hugging Face."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.detector import resolve_plate_model_from_hub

try:
    p = resolve_plate_model_from_hub("Koushim/yolov8-license-plate-detection", "best.pt")
    print("OK:", p, os.path.getsize(p))
except Exception as e:
    print("FAILED:", type(e).__name__, e)