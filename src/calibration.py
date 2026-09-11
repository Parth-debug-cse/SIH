"""Camera configuration loading and validation.

Reads ``data/calibration/cameras.json`` and makes the calibration values
(``pixel_to_meter_ratio``, ``compass_bearing``, GPS) available to every
runtime component that needs them: pipeline workers, analytics, fusion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CAMERAS_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration" / "cameras.json"

REQUIRED_FIELDS = ("camera_id", "gps_lat", "gps_lon")
RECOMMENDED_FIELDS = ("video_file", "pixel_to_meter_ratio", "compass_bearing")


def load_cameras_config(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load camera definitions into a ``{camera_id: config}`` dict.

    Raises:
        FileNotFoundError if the config file is missing.
        ValueError if the file is not valid or uses an unexpected layout.
    """
    path = Path(path or DEFAULT_CAMERAS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Camera config not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    cameras_raw = data.get("cameras", data if isinstance(data, list) else [])
    if not isinstance(cameras_raw, list):
        raise ValueError(f"Camera config must contain a 'cameras' list: {path}")

    cameras: dict[str, dict[str, Any]] = {}
    for cam in cameras_raw:
        if not isinstance(cam, dict) or "camera_id" not in cam:
            continue
        cam_id = str(cam["camera_id"])
        cameras[cam_id] = dict(cam)
    return cameras


def merge_camera_config(
    camera_id: str,
    cameras: dict[str, dict[str, Any]],
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge a camera's stored calibration onto a *base* config dict.

    The explicit *base* values (normally gathered from CLI/launcher)
    take precedence over the calibration file.
    """
    merged: dict[str, Any] = dict(cameras.get(camera_id, {}))
    if base:
        merged.update({k: v for k, v in base.items() if v is not None})
    return merged


def validate_cameras_config(
    cameras: dict[str, dict[str, Any]],
) -> list[str]:
    """Return a list of human-readable warnings/errors about a config set.

    Warning strings use prefix ``"[ERROR]"`` / ``"[WARN]"`` so callers can
    distinguish hard failures from advisories.
    """
    issues: list[str] = []

    if not cameras:
        issues.append("[ERROR] No cameras defined in configuration.")
        return issues

    seen_ids: set[str] = set()
    for cam_id, cfg in cameras.items():
        if cam_id in seen_ids:
            issues.append(f"[ERROR] Duplicate camera_id '{cam_id}'.")
        seen_ids.add(cam_id)

        for field in REQUIRED_FIELDS:
            if field not in cfg or cfg[field] in (None, ""):
                issues.append(f"[ERROR] Camera '{cam_id}' missing required field '{field}'.")
            elif field.startswith("gps") and cfg[field] in ("", None):
                issues.append(f"[ERROR] Camera '{cam_id}' has empty '{field}'.")

        for field in ("pixel_to_meter_ratio", "compass_bearing"):
            val = cfg.get(field)
            if val is None:
                issues.append(
                    f"[WARN] Camera '{cam_id}' missing '{field}' – speed estimates "
                    "and direction metadata will be unavailable."
                )
            elif field == "pixel_to_meter_ratio" and float(val) <= 0:
                issues.append(f"[WARN] Camera '{cam_id}' has non-positive 'pixel_to_meter_ratio'.")

    return issues


def log_validation(issues: list[str]) -> None:
    """Log validation issues: errors always, warnings at WARNING level."""
    for msg in issues:
        if msg.startswith("[ERROR]"):
            logger.error("%s", msg)
        else:
            logger.warning("%s", msg)