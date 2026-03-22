"""
mrv.utils.log — Logging setup driven by config.yaml.

Call ``setup(cfg)`` once at application start to configure console + optional
file logging.  All mrv modules use ``logging.getLogger(__name__)`` as usual.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure Python logging from the ``logging`` section of mrv config.

    When ``logging.dir`` is set, a timestamped log file is created
    automatically, e.g. ``logs/mrv_20260319_143012.log``.

    Parameters
    ----------
    cfg : dict, optional
        Full mrv config dict (as returned by ``mrv.utils.config.load()``).
        If *None*, sensible defaults are applied.
    """
    log_cfg = (cfg or {}).get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s %(levelname)s [%(name)s] %(message)s")
    log_dir = log_cfg.get("log_dir")
    quiet = log_cfg.get("quiet", [])

    root = logging.getLogger()

    # Avoid duplicate handlers on repeated calls
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
        root.setLevel(level)

    # File handler: auto-generate timestamped filename under log_dir
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _add_file_handler(log_path / f"mrv_{ts}.log", level, fmt)

    # Silence noisy loggers
    for name in quiet:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.captureWarnings(True)


def _add_file_handler(path: Path, level: int, fmt: str) -> None:
    """Add a file handler if one for this path does not already exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    existing = set()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                existing.add(Path(h.baseFilename).resolve())
            except Exception:
                continue
    if path.resolve() not in existing:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
    if root.level > level:
        root.setLevel(level)
