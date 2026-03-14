"""
mrv_lib.core — Sensitivity diagnostics, identifiability boundary, ordinal metrics, and CLI.

- Scanner / RepresentationTestResult: representation and resolution stability (RSS).
- detect_boundary / BoundaryResult: Identifiability Index and collapse zone.
- ordinal_consistency / ari_score: ordinal robustness when ARI collapses.
- main: command-line entry point (mrv-lib).
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Representation stability (RSS)
# ---------------------------------------------------------------------------

@dataclass
class RepresentationTestResult:
    """Result of run_representation_test; holds RSS and per-resolution scores."""

    rss_score: float
    resolution_scores: dict
    representation_scores: Optional[dict] = None


class Scanner:
    """
    Diagnostic scanner for regime label stability across resolutions and representations.
    """

    def __init__(
        self,
        resolution: Optional[List[str]] = None,
        feature_sets: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        resolution : list of str, optional
            Temporal resolutions to test, e.g. ['5m', '1h', '1d'].
        feature_sets : list of str, optional
            Feature set names for representation sensitivity; defaults to standard OHLCV-derived sets.
        """
        self.resolution = resolution or ["1h", "1d"]
        self.feature_sets = feature_sets or ["returns", "volatility", "volume"]

    def run_representation_test(
        self,
        data: pd.DataFrame,
        model: str = "HMM",
        **kwargs,
    ) -> RepresentationTestResult:
        """
        Run representation stability test across resolutions and feature sets.

        Parameters
        ----------
        data : DataFrame
            OHLCV market data (columns expected: open, high, low, close, volume or similar).
        model : str
            Regime model type, e.g. "HMM". Used for extensibility.
        **kwargs
            Passed to internal model fit (e.g. n_states for HMM).

        Returns
        -------
        RepresentationTestResult
            Contains rss_score and per-resolution / per-representation scores.
        """
        data = _ensure_ohlcv(data)
        resolution_scores = {}
        for res in self.resolution:
            resolution_scores[res] = _score_at_resolution(data, res, model, **kwargs)
        representation_scores = {}
        for fs in self.feature_sets:
            representation_scores[fs] = _score_representation(data, fs, model, **kwargs)

        rss = _compute_rss(resolution_scores, representation_scores)
        return RepresentationTestResult(
            rss_score=rss,
            resolution_scores=resolution_scores,
            representation_scores=representation_scores,
        )


def _ensure_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower-case open/high/low/close/volume if present."""
    df = data.copy()
    cols = {c.lower(): c for c in df.columns}
    for name in ["open", "high", "low", "close", "volume"]:
        if name in cols and df.columns[df.columns.get_loc(cols[name])] != name:
            df = df.rename(columns={cols[name]: name})
    if "close" not in df.columns and len(df.columns) >= 4:
        df.columns = ["open", "high", "low", "close"] + list(df.columns[4:])
    return df


def _score_at_resolution(
    data: pd.DataFrame,
    resolution: str,
    model: str,
    **kwargs,
) -> float:
    """
    Placeholder: score stability at a given temporal resolution.
    In production, resample data to resolution and compare regime consistency.
    """
    close = data["close"] if "close" in data.columns else data.iloc[:, 3]
    returns = close.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    vol = returns.std()
    vol_norm = min(vol * 10, 1.0)
    return float(1.0 - vol_norm)


def _score_representation(
    data: pd.DataFrame,
    feature_set: str,
    model: str,
    **kwargs,
) -> float:
    """
    Placeholder: score stability across feature set (representation).
    In production, fit regime model on different feature sets and compare labels.
    """
    return 0.75


def _compute_rss(
    resolution_scores: dict,
    representation_scores: dict,
) -> float:
    """
    Representation Stability Score: aggregate of resolution and representation stability.
    RSS in [0, 1]; higher = more robust.
    """
    r_scores = list(resolution_scores.values()) if resolution_scores else [0.0]
    rep_scores = list(representation_scores.values()) if representation_scores else [0.0]
    all_scores = r_scores + rep_scores
    return float(np.clip(np.mean(all_scores), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Identifiability boundary
# ---------------------------------------------------------------------------

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

    vol = returns.rolling(w).std().dropna()
    if vol.empty or vol.iloc[-1] == 0:
        drift = 0.0
    else:
        vol_early = vol.iloc[: len(vol) // 2].mean()
        vol_late = vol.iloc[len(vol) // 2 :].mean()
        drift = abs(vol_late - vol_early) / (vol.mean() + 1e-10)
        drift = min(drift, 1.0)

    identifiability_index = float(np.clip(1.0 - drift, 0.0, 1.0))
    if np.isnan(identifiability_index):
        identifiability_index = 0.0
    is_collapsed = identifiability_index < threshold

    return BoundaryResult(
        index=identifiability_index,
        is_collapsed=is_collapsed,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Ordinal robustness (metrics)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mrv-lib",
        description=(
            "Market Regime Validity diagnostics: "
            "Representation Stability (RSS) and Identifiability Index."
        ),
    )
    parser.add_argument(
        "csv",
        help="Path to OHLCV market data CSV file.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        default=["5m", "1h", "1d"],
        help="Temporal resolutions to test, e.g. 5m 1h 1d (default: 5m 1h 1d).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="HMM",
        help="Regime model type label (passed through to Scanner; default: HMM).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point. Example: mrv-lib market_data.csv --resolution 5m 1h 1d --model HMM."""
    args = _parse_args(argv)

    data = pd.read_csv(args.csv)

    scanner = Scanner(resolution=args.resolution)
    results = scanner.run_representation_test(data, model=args.model)

    boundary = detect_boundary(data)

    print(f"File: {args.csv}")
    print(f"Resolutions: {args.resolution}")
    print(f"Model: {args.model}")
    print(f"RSS (Representation Stability Score): {results.rss_score:.4f}")
    print(
        f"Identifiability Index: {boundary.index:.4f} "
        f"(threshold={boundary.threshold:.2f})"
    )
    if boundary.is_collapsed:
        print("Status: COLLAPSED (Inference Collapse Zone)")
    else:
        print("Status: STABLE (outside collapse zone)")
