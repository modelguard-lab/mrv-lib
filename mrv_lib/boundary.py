"""
mrv_lib.boundary — Identifiability Index.

Computes the Identifiability Index (I) and detects Phase Boundaries
where model inference collapses.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# Default threshold below which the market is considered in "Zone of Collapse"
DEFAULT_COLLAPSE_THRESHOLD = 0.3


@dataclass
class BoundaryResult:
    """
    Result of identifiability boundary detection.

    Attributes
    ----------
    index : float
        Identifiability Index (I); higher = more identifiable regime.
    is_collapsed : bool
        True if index below threshold (Inference Collapse Zone).
    threshold : float
        Threshold used for is_collapsed.
    """

    index: float
    is_collapsed: bool
    threshold: float = DEFAULT_COLLAPSE_THRESHOLD


def detect_boundary(
    data: pd.DataFrame,
    threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    window: Optional[int] = None,
) -> BoundaryResult:
    """
    Detect identifiability boundary: compute Identifiability Index (I) and
    whether the market is in a "Zone of Collapse".

    Parameters
    ----------
    data : DataFrame
        OHLCV market data.
    threshold : float
        Index below this value is considered collapsed (default 0.3).
    window : int, optional
        Rolling window for drift/separation; default uses 20% of length.

    Returns
    -------
    BoundaryResult
        .index (Identifiability Index), .is_collapsed, .threshold.
    """
    df = data.copy()
    if "close" not in df.columns and len(df.columns) >= 4:
        df = df.rename(columns={df.columns[3]: "close"})
    close = df["close"] if "close" in df.columns else df.iloc[:, 3]
    close = pd.Series(close).dropna()
    n = len(close)
    if n < 2:
        return BoundaryResult(index=0.0, is_collapsed=True, threshold=threshold)

    w = window or max(int(0.2 * n), 2)
    returns = close.pct_change().dropna()
    if len(returns) < w:
        return BoundaryResult(index=0.0, is_collapsed=True, threshold=threshold)

    # Structural drift: normalized change in rolling volatility
    vol = returns.rolling(w).std().dropna()
    if vol.empty or vol.iloc[-1] == 0:
        drift = 0.0
    else:
        vol_early = vol.iloc[: len(vol) // 2].mean()
        vol_late = vol.iloc[len(vol) // 2 :].mean()
        drift = abs(vol_late - vol_early) / (vol.mean() + 1e-10)
        drift = min(drift, 1.0)

    # Regime separation proxy: inverse of drift (when drift is high, identifiability is low)
    identifiability_index = float(np.clip(1.0 - drift, 0.0, 1.0))
    is_collapsed = identifiability_index < threshold

    return BoundaryResult(
        index=identifiability_index,
        is_collapsed=is_collapsed,
        threshold=threshold,
    )
