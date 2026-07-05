"""
mrv.data.normalize -- Feature normalization / standardization.

Adapted from Paper 1 (Representation-Invariant) ``features.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "rolling_zscore",
    "minmax",
    "normalize",
]


def _warn_normalization_inputs(df: pd.DataFrame, window: int, fn_name: str) -> None:
    """Emit warnings for common normalization pitfalls."""
    if window > len(df):
        logger.warning(
            "%s: window=%d exceeds DataFrame length=%d; all output rows will be NaN.",
            fn_name, window, len(df),
        )
    for col in df.columns:
        series = df[col].dropna()
        if len(series) >= window and series.std(ddof=1) == 0.0:
            logger.warning(
                "%s: column '%s' has zero standard deviation; output will be all-NaN "
                "for that column.",
                fn_name, col,
            )


def rolling_zscore(
    df: pd.DataFrame,
    window: int = 120,
) -> pd.DataFrame:
    """
    Rolling z-score normalization: ``(x - rolling_mean) / rolling_std``.

    Parameters
    ----------
    df : DataFrame
        Raw feature matrix.
    window : int
        Lookback window for mean and std.
    """
    _warn_normalization_inputs(df, window, "rolling_zscore")
    mean = df.rolling(window=window, min_periods=window).mean()
    std = df.rolling(window=window, min_periods=window).std()
    std = std.replace(0, np.nan)  # avoid division by zero
    return (df - mean) / std


def minmax(
    df: pd.DataFrame,
    window: int = 120,
) -> pd.DataFrame:
    """
    Rolling min-max normalization to [0, 1].

    Parameters
    ----------
    df : DataFrame
        Raw feature matrix.
    window : int
        Lookback window for min and max.
    """
    _warn_normalization_inputs(df, window, "minmax")
    lo = df.rolling(window=window, min_periods=window).min()
    hi = df.rolling(window=window, min_periods=window).max()
    denom = hi - lo
    denom = denom.replace(0, np.nan)
    return (df - lo) / denom


def normalize(
    df: pd.DataFrame,
    mode: Optional[str] = None,
    window: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Normalize a feature DataFrame using the specified method.

    Parameters
    ----------
    df : DataFrame
        Raw feature matrix (output of ``factors.build_factors``).
    mode : str, optional
        ``"rolling_zscore"`` | ``"minmax"`` | ``"none"``.
        If *None*, read from ``cfg["normalize"]["mode"]``.
    window : int, optional
        Lookback window.  If *None*, read from ``cfg["normalize"]["window"]``.
    cfg : dict, optional
        Full mrv config dict.

    Returns
    -------
    DataFrame
        Normalized features (same shape as input).
    """
    norm_cfg = (cfg or {}).get("normalize", {})
    mode = mode or norm_cfg.get("mode", "rolling_zscore")
    window = window or int(norm_cfg.get("window", 120))

    if mode == "none" or mode is None:
        return df.copy()
    if mode == "rolling_zscore":
        return rolling_zscore(df, window)
    if mode == "minmax":
        return minmax(df, window)

    raise ValueError(f"Unknown normalization mode: {mode!r}")
