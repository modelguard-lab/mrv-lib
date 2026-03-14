"""
mrv_lib.metrics — Ordinal Robustness.

When absolute labels (ARI) collapse, measures Ordinal Consistency
(Spearman's Rho) for fail-safe risk ranking.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def ordinal_consistency(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """
    Ordinal Consistency: Spearman's rank correlation between true and predicted
    regime order. Valid when ARI is low but ranking for hedging remains meaningful.

    Parameters
    ----------
    y_true : array-like
        True regime labels or risk ordering (numeric).
    y_pred : array-like
        Predicted regime labels or risk ordering (numeric).

    Returns
    -------
    float
        Spearman's Rho in [-1, 1]; higher = more ordinal consistency.
    """
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(np.clip(rho, -1.0, 1.0) if not np.isnan(rho) else 0.0)


def ari_score(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """
    Adjusted Rand Index for absolute label agreement.
    Use ordinal_consistency when ARI collapses but ranking still matters.

    Parameters
    ----------
    y_true : array-like
        True cluster/regime labels (integer).
    y_pred : array-like
        Predicted cluster/regime labels (integer).

    Returns
    -------
    float
        ARI in [-1, 1]; 1 = perfect match.
    """
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return 0.0
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return 0.0
    return float(adjusted_rand_score(y_true, y_pred))
