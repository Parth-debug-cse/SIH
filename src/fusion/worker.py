"""Dedicated shared fusion worker process.

Run with::

    python -m src.fusion.worker --interval 3

This process polls the shared SQLite database for new sightings and persists
fused trajectories, so that multi-camera demo runs converge on a single
logical cross-camera fusion state without any per-worker fusion instances.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from src.fusion.engine import DEFAULT_CAMERAS_PATH, run_fusion_loop

_FORMAT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shared cross-camera fusion worker")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Poll interval in seconds (default: 3.0).")
    parser.add_argument("--db-path", default=None,
                        help="Path to the SQLite DB (default: data/anpr.db).")
    parser.add_argument("--config", default=str(DEFAULT_CAMERAS_PATH),
                        help="Path to camera calibration JSON.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format=_FORMAT)
    logger = logging.getLogger("src.fusion.worker")

    def _stop(*_):  # pragma: no cover - signal handler
        logger.info("Fusion worker shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    run_fusion_loop(
        interval_seconds=args.interval,
        db_path=args.db_path,
        cameras_config_path=args.config,
    )


if __name__ == "__main__":
    main()