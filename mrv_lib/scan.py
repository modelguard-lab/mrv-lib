"""
mrv_lib.scan — Sensitivity Diagnostic.

Stress-testing of regime labels across feature sets (Representation) and
temporal scales (Resolution). Computes RSS (Representation Stability Score).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd


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
        data = self._ensure_ohlcv(data)
        resolution_scores = {}
        for res in self.resolution:
            resolution_scores[res] = self._score_at_resolution(data, res, model, **kwargs)
        representation_scores = {}
        for fs in self.feature_sets:
            representation_scores[fs] = self._score_representation(data, fs, model, **kwargs)

        rss = self._compute_rss(resolution_scores, representation_scores)
        return RepresentationTestResult(
            rss_score=rss,
            resolution_scores=resolution_scores,
            representation_scores=representation_scores,
        )

    def _ensure_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
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
        self,
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
        # Simple proxy: inverse of normalized volatility (higher stability when vol is moderate)
        vol = returns.std()
        vol_norm = min(vol * 10, 1.0)
        return float(1.0 - vol_norm)

    def _score_representation(
        self,
        data: pd.DataFrame,
        feature_set: str,
        model: str,
        **kwargs,
    ) -> float:
        """
        Placeholder: score stability across feature set (representation).
        In production, fit regime model on different feature sets and compare labels.
        """
        return 0.75  # Stub; replace with cross-representation agreement

    def _compute_rss(
        self,
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
