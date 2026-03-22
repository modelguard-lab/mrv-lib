"""
mrv.utils.config — Load YAML configuration.

Single source of truth is config.yaml. No built-in defaults to maintain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _find_default_config() -> Path:
    """Locate config.yaml by walking upward from this file."""
    anchor = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = anchor / "config.yaml"
        if candidate.exists():
            return candidate
        anchor = anchor.parent
    return Path.cwd() / "config.yaml"


_DEFAULT_CONFIG_PATH = _find_default_config()


def load(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML.

    Parameters
    ----------
    path : str or Path, optional
        If None, searches for config.yaml automatically.

    Raises
    ------
    FileNotFoundError
        If no config file found.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config (expected dict): {cfg_path}")
    return cfg


def get_data_dir(cfg: Dict[str, Any], base: Optional[Path] = None) -> Path:
    """Resolve data directory from config."""
    data_dir = Path(cfg.get("download", {}).get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = (base or Path.cwd()) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _normalize_freq(raw) -> List[str]:
    """Ensure freq is always a list of strings."""
    if isinstance(raw, list):
        return [str(f) for f in raw]
    if raw is None:
        return ["1d"]
    return [str(raw)]


def get_assets(cfg: Dict[str, Any], freq: Optional[str] = None) -> List[Dict[str, Any]]:
    """Expand download.symbols into asset dicts."""
    dl = cfg.get("download", {})
    symbols = dl.get("symbols", [])
    freqs = _normalize_freq(dl.get("freq"))
    start = dl.get("start")
    end = dl.get("end")

    assets = [
        {"symbol": str(s), "freq": freqs, "start": start, "end": end}
        for s in symbols
    ]
    if freq is not None:
        assets = [a for a in assets if freq in a["freq"]]
    return assets
