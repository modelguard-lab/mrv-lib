"""
mrv.validator.res — Resolution Invariance test (Paper 2).

One asset, multiple frequencies (5m/15m/1h/1d) → fit regime model per freq →
align to 5m timestamps → compare via cross-frequency ARI matrix.

Features from Paper 2:
- Multi-frequency resampling and regime fitting (GMM/HMM on log-vol)
- Cross-frequency ARI/AMI/VI matrices
- Permutation p-values for mean off-diagonal ARI
- Event/calm window analysis
- Daily & 7-day rolling ARI summaries
- Time-of-day (TOD) seasonality analysis
- TOD-adjusted volatility robustness
- Calendar-window robustness (fixed physical-time window)
- Robustness sweeps over K and window scales
- Timeline and rolling ARI visualizations
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, mutual_info_score

from mrv.validator.base import BaseValidator
from mrv.validator.metrics import ARI_THRESHOLD
from mrv.data.reader import load_ohlcv, resample_ohlc

import matplotlib
if matplotlib.get_backend().lower() == "qtagg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FREQS = ("5m", "15m", "1h", "1d")
TZ = "America/New_York"
DEFAULT_GMM_K = 2
DEFAULT_WINDOW_SCALE = 1.0
DEFAULT_ROLLING_DAYS = 7
MIN_TRADING_DAYS = 5
BARS_PER_DAY = {"SPY": 78, "SPX": 78, "CL": 276, "USDJPY": 288}

# Episode registry: maps name → (event_window, calm_window)
EPISODES: Dict[str, Tuple[Tuple[str, str], Tuple[str, str]]] = {
    "2026_iran": (("2026-01-20", "2026-01-24"), ("2026-02-10", "2026-02-14")),
    "2022_ukraine": (("2022-02-22", "2022-02-28"), ("2022-01-10", "2022-01-14")),
}


# ── Feature Engineering ──────────────────────────────────────────────────────

def _window_spec(freq: str, window_scale: float = DEFAULT_WINDOW_SCALE) -> "str | int":
    """Return the rolling-volatility window for a given frequency."""
    scale = max(float(window_scale), 0.25)
    if freq in {"5m", "15m"}:
        minutes = max(30, int(round(120 * scale)))
        return f"{minutes}min"
    if freq == "1h":
        return max(2, int(round(24 * scale)))
    if freq == "1d":
        return max(2, int(round(5 * scale)))
    raise ValueError(f"Unsupported freq for window spec: {freq}")


def _robust_filter_returns(ret: pd.Series, stem: str, freq: str) -> pd.Series:
    """Mitigate extreme continuous-futures roll jumps (CL only)."""
    if stem != "CL" or ret.dropna().empty:
        return ret
    med = ret.median()
    mad = (ret - med).abs().median()
    if pd.isna(mad) or mad < 1e-10:
        return ret
    scale = 1.4826 * mad
    clip_k = 8.0 if freq == "5m" else 10.0
    return ret.clip(lower=med - clip_k * scale, upper=med + clip_k * scale)


def _features(
    df_ohlc: pd.DataFrame,
    freq: str,
    stem: str = "",
    window_scale: float = DEFAULT_WINDOW_SCALE,
    calendar_window: Optional[str] = None,
) -> pd.DataFrame:
    """Build log-return and rolling-volatility features.

    If *calendar_window* is provided (e.g. "6h"), it overrides the default
    frequency-specific window, enabling cross-frequency comparability.
    """
    out = pd.DataFrame(index=df_ohlc.index)
    out["ret"] = np.log(df_ohlc["Close"] / df_ohlc["Close"].shift(1))
    out["ret"] = out["ret"].clip(lower=-0.03, upper=0.03)
    out["ret"] = _robust_filter_returns(out["ret"], stem, freq)
    win = calendar_window if calendar_window else _window_spec(freq, window_scale=window_scale)
    out["vol"] = out["ret"].rolling(window=win, min_periods=2).std()
    out = out.replace([np.inf, -np.inf], np.nan)
    out["vol"] = out["vol"].ffill().bfill()
    return out


# ── Regime Fitting ───────────────────────────────────────────────────────────

def _fit_regime_gmm(
    feats: pd.DataFrame,
    n_components: int = DEFAULT_GMM_K,
    freq: str = "",
) -> Tuple[pd.Series, bool]:
    """Fit GMM on log(vol); highest-mean cluster = crisis.

    Returns (labels, fallback_triggered).
    """
    from sklearn.mixture import GaussianMixture

    vol_vals = feats["vol"].fillna(feats["vol"].median()).values
    eps = 1e-12
    log_vol = np.log(np.maximum(vol_vals, eps))

    X = log_vol[~np.isnan(log_vol) & np.isfinite(log_vol)].reshape(-1, 1)
    if len(X) < n_components * 2:
        return pd.Series(0, index=feats.index, dtype=int), False
    if np.std(X) < 1e-10 or np.ptp(X) < 1e-10:
        logger.warning("Regime skip (%s): log-vol constant, returning calm", freq or "?")
        return pd.Series(0, index=feats.index, dtype=int), False

    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
    gmm.fit(X)
    crisis_cluster = int(np.argmax(gmm.means_.ravel()))
    pred = (gmm.predict(log_vol.reshape(-1, 1)) == crisis_cluster).astype(int)

    crisis_pct = 100.0 * (pred == 1).mean()
    fallback = False
    if crisis_pct < 1.0 or crisis_pct > 99.0:
        thresh = np.nanpercentile(log_vol, 80)
        pred = (log_vol >= thresh).astype(int)
        fallback = True
        logger.info("Regime fallback (%s): GMM trivial (%.1f%%); using 80th pctl", freq or "?", crisis_pct)
    return pd.Series(pred, index=feats.index, dtype=int), fallback


def _fit_regime_hmm(
    feats: pd.DataFrame,
    n_components: int = DEFAULT_GMM_K,
    freq: str = "",
) -> Tuple[pd.Series, bool]:
    """Fit Gaussian HMM on log(vol); highest-mean state = crisis."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:
        logger.warning("HMM unavailable (%s); falling back to GMM for %s", e, freq or "?")
        return _fit_regime_gmm(feats, n_components=n_components, freq=freq)

    vol_vals = feats["vol"].fillna(feats["vol"].median()).values
    eps = 1e-12
    log_vol = np.log(np.maximum(vol_vals, eps))
    X = log_vol[~np.isnan(log_vol) & np.isfinite(log_vol)].reshape(-1, 1)
    if len(X) < max(4, n_components * 4):
        return pd.Series(0, index=feats.index, dtype=int), False
    if float(np.std(X)) < 1e-10 or float(np.ptp(X)) < 1e-10:
        logger.warning("Regime skip (%s): log-vol constant, returning calm", freq or "?")
        return pd.Series(0, index=feats.index, dtype=int), False

    hmm = GaussianHMM(n_components=int(n_components), covariance_type="diag",
                       n_iter=200, random_state=42, verbose=False)
    try:
        hmm.fit(X)
        crisis_state = int(np.argmax(hmm.means_.ravel()))
        states = hmm.predict(log_vol.reshape(-1, 1))
    except Exception as e:
        logger.warning("HMM fit/decode failed (%s): %s; returning calm", freq or "?", e)
        return pd.Series(0, index=feats.index, dtype=int), False

    pred = (states == crisis_state).astype(int)
    crisis_pct = 100.0 * (pred == 1).mean()
    fallback = False
    if crisis_pct < 1.0 or crisis_pct > 99.0:
        thresh = np.nanpercentile(log_vol, 80)
        pred = (log_vol >= thresh).astype(int)
        fallback = True
        logger.info("Regime fallback (%s,HMM): trivial (%.1f%%); using 80th pctl", freq or "?", crisis_pct)
    return pd.Series(pred, index=feats.index, dtype=int), fallback


def _fit_regime_model(
    feats: pd.DataFrame,
    model: str = "gmm",
    n_components: int = DEFAULT_GMM_K,
    freq: str = "",
) -> Tuple[pd.Series, bool]:
    """Dispatch to GMM or HMM regime fitter."""
    if model.strip().lower() == "hmm":
        return _fit_regime_hmm(feats, n_components=n_components, freq=freq)
    return _fit_regime_gmm(feats, n_components=n_components, freq=freq)


# ── Alignment & Metrics ──────────────────────────────────────────────────────

def _align_regimes_to_5m(
    regimes_by_freq: Dict[str, pd.Series],
    index_5m: pd.DatetimeIndex,
) -> Dict[str, pd.Series]:
    """Forward-fill each frequency's labels onto 5m timestamps."""
    return {
        freq: ser.reindex(index_5m, method="ffill").fillna(0).astype(int)
        for freq, ser in regimes_by_freq.items()
    }


def _compute_ari_matrix(
    aligned: Dict[str, pd.Series],
    index_subset: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Compute cross-frequency ARI matrix on chosen 5m timestamps."""
    if index_subset is None:
        base = aligned.get("5m")
        index_subset = base.index if base is not None else pd.DatetimeIndex([])

    n = len(FREQS)
    mat = np.eye(n, dtype=float)
    for i, fa in enumerate(FREQS):
        for j, fb in enumerate(FREQS):
            if j <= i:
                continue
            a = aligned[fa].reindex(index_subset)
            b = aligned[fb].reindex(index_subset)
            common = a.index.intersection(b.index).dropna()
            if len(common) < 10:
                mat[i, j] = mat[j, i] = np.nan
            else:
                score = adjusted_rand_score(a.loc[common].astype(int), b.loc[common].astype(int))
                mat[i, j] = mat[j, i] = float(np.clip(round(score, 6), -1.0, 1.0))
    return pd.DataFrame(mat, index=list(FREQS), columns=list(FREQS))


def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))


def _variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    mi = mutual_info_score(a, b)
    return _entropy(a) + _entropy(b) - 2.0 * mi


def _compute_extra_metrics(
    aligned: Dict[str, pd.Series],
    index_subset: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute AMI and VI matrices."""
    if index_subset is None:
        base = aligned.get("5m")
        index_subset = base.index if base is not None else pd.DatetimeIndex([])

    n = len(FREQS)
    ami_mat = np.eye(n, dtype=float)
    vi_mat = np.zeros((n, n), dtype=float)
    for i, fa in enumerate(FREQS):
        for j, fb in enumerate(FREQS):
            if j <= i:
                continue
            a = aligned[fa].reindex(index_subset)
            b = aligned[fb].reindex(index_subset)
            common = a.index.intersection(b.index).dropna()
            if len(common) < 10:
                ami_mat[i, j] = ami_mat[j, i] = np.nan
                vi_mat[i, j] = vi_mat[j, i] = np.nan
            else:
                av = a.loc[common].astype(int).values
                bv = b.loc[common].astype(int).values
                ami_mat[i, j] = ami_mat[j, i] = float(np.clip(
                    round(adjusted_mutual_info_score(av, bv), 6), -1.0, 1.0))
                vi_mat[i, j] = vi_mat[j, i] = float(round(_variation_of_information(av, bv), 6))
    return {
        "ami": pd.DataFrame(ami_mat, index=list(FREQS), columns=list(FREQS)),
        "vi": pd.DataFrame(vi_mat, index=list(FREQS), columns=list(FREQS)),
    }


def _mean_offdiag(mat: pd.DataFrame) -> Optional[float]:
    """Mean of off-diagonal entries."""
    if mat is None or mat.empty:
        return None
    vals = mat.values.astype(float)
    if vals.shape[0] != vals.shape[1]:
        return None
    mask = ~np.eye(vals.shape[0], dtype=bool)
    offdiag = vals[mask]
    return float(np.nanmean(offdiag)) if offdiag.size else None


# ── Permutation Test ─────────────────────────────────────────────────────────

def _permute_pvalue_mean_offdiag_ari(
    aligned: Dict[str, pd.Series],
    index_subset: Optional[pd.DatetimeIndex] = None,
    n_perm: int = 500,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Permutation test for mean off-diagonal ARI.

    Returns (p_value, (ci_low, ci_high)).
    """
    base = aligned.get("5m")
    if base is None or base.empty:
        return None, None
    if index_subset is None:
        index_subset = base.index
    if len(index_subset) < 50:
        return None, None

    obs_df = _compute_ari_matrix(aligned, index_subset=index_subset)
    obs = _mean_offdiag(obs_df)
    if obs is None or not np.isfinite(obs):
        return None, None

    rng = np.random.default_rng(seed)
    y = {freq: aligned[freq].reindex(index_subset).astype(int).to_numpy() for freq in FREQS}

    null_stats = np.empty(int(n_perm), dtype=float)
    for k in range(int(n_perm)):
        y_perm = {freq: rng.permutation(arr) for freq, arr in y.items()}
        vals = []
        for i, fa in enumerate(FREQS):
            for j, fb in enumerate(FREQS):
                if j <= i:
                    continue
                vals.append(adjusted_rand_score(y_perm[fa], y_perm[fb]))
        null_stats[k] = float(np.mean(vals)) if vals else np.nan

    ge = np.mean(null_stats >= obs)
    p = float((ge * n_perm + 1.0) / (n_perm + 1.0))
    ci = (float(np.nanpercentile(null_stats, 2.5)), float(np.nanpercentile(null_stats, 97.5)))
    return p, ci


# ── Block Permutation Test ──────────────────────────────────────────────────

def _block_permute_pvalue_mean_offdiag_ari(
    aligned: Dict[str, pd.Series],
    index_subset: Optional[pd.DatetimeIndex] = None,
    n_perm: int = 500,
    block_size: int = 50,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Block-permutation test preserving within-block autocorrelation.

    Instead of permuting individual timestamps, we permute contiguous blocks
    of *block_size* bars.  This preserves regime persistence within each block
    while destroying cross-resolution temporal alignment.

    Returns (p_value, (ci_low, ci_high)).
    """
    base = aligned.get("5m")
    if base is None or base.empty:
        return None, None
    if index_subset is None:
        index_subset = base.index
    n = len(index_subset)
    if n < block_size * 3:
        return None, None

    obs_df = _compute_ari_matrix(aligned, index_subset=index_subset)
    obs = _mean_offdiag(obs_df)
    if obs is None or not np.isfinite(obs):
        return None, None

    rng = np.random.default_rng(seed)
    y = {freq: aligned[freq].reindex(index_subset).astype(int).to_numpy() for freq in FREQS}

    n_blocks = n // block_size
    null_stats = np.empty(int(n_perm), dtype=float)

    for k in range(int(n_perm)):
        y_perm: Dict[str, np.ndarray] = {}
        for freq, arr in y.items():
            blocks = [arr[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
            remainder = arr[n_blocks * block_size:]
            perm_order = rng.permutation(len(blocks))
            shuffled = np.concatenate([blocks[i] for i in perm_order])
            if len(remainder):
                shuffled = np.concatenate([shuffled, remainder])
            y_perm[freq] = shuffled[:n]

        vals = []
        for i, fa in enumerate(FREQS):
            for j, fb in enumerate(FREQS):
                if j <= i:
                    continue
                vals.append(adjusted_rand_score(y_perm[fa][:n], y_perm[fb][:n]))
        null_stats[k] = float(np.mean(vals)) if vals else np.nan

    ge = np.mean(null_stats >= obs)
    p = float((ge * n_perm + 1.0) / (n_perm + 1.0))
    ci = (float(np.nanpercentile(null_stats, 2.5)), float(np.nanpercentile(null_stats, 97.5)))
    return p, ci


# ── GMM Fit Diagnostics ────────────────────────────────────────────────────

def _gmm_fit_diagnostics(
    df_ohlc: pd.DataFrame,
    freq: str,
    stem: str = "",
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = DEFAULT_WINDOW_SCALE,
) -> Dict[str, Any]:
    """Return GMM fit quality diagnostics: BIC, AIC, component means/stds, separation."""
    from sklearn.mixture import GaussianMixture

    feats = _features(df_ohlc, freq, stem=stem, window_scale=window_scale)
    vol_vals = feats["vol"].fillna(feats["vol"].median()).values
    eps = 1e-12
    log_vol = np.log(np.maximum(vol_vals, eps))
    X = log_vol[~np.isnan(log_vol) & np.isfinite(log_vol)].reshape(-1, 1)
    if len(X) < n_components * 2:
        return {"bic": np.nan, "aic": np.nan, "n_obs": len(X)}

    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
    gmm.fit(X)
    means = gmm.means_.ravel()
    stds = (np.sqrt(gmm.covariances_.ravel()) if gmm.covariances_.ndim <= 2
            else np.sqrt(gmm.covariances_[:, 0, 0]))
    weights = gmm.weights_.ravel()

    # Component separation: difference in means / pooled std
    if n_components == 2 and len(stds) == 2:
        pooled_std = np.sqrt(weights[0] * stds[0]**2 + weights[1] * stds[1]**2)
        separation = abs(means[1] - means[0]) / pooled_std if pooled_std > 1e-12 else np.nan
    else:
        separation = np.nan

    # Overlap: P(lower > boundary) + P(higher < boundary)
    if n_components == 2:
        from scipy.stats import norm
        boundary = (means[0] + means[1]) / 2.0
        lower_idx = int(np.argmin(means))
        higher_idx = 1 - lower_idx
        overlap = (
            norm.sf(boundary, loc=means[lower_idx], scale=max(stds[lower_idx], 1e-12))
            + norm.cdf(boundary, loc=means[higher_idx], scale=max(stds[higher_idx], 1e-12))
        )
    else:
        overlap = np.nan

    return {
        "bic": float(gmm.bic(X)),
        "aic": float(gmm.aic(X)),
        "n_obs": len(X),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "separation": float(separation),
        "overlap": float(overlap),
    }


# ── Expanding-Window Regime Fitting ────────────────────────────────────────

def _fit_regime_expanding(
    df_ohlc: pd.DataFrame,
    freq: str,
    stem: str = "",
    model: str = "gmm",
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    min_train_bars: int = 200,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Fit regime labels using an expanding window (no look-ahead).

    For each bar *t*, the GMM is fitted on bars [0, t] and the label for *t*
    is the prediction from that model.  To avoid degenerate fits at the start,
    the first *min_train_bars* bars use the model fitted on the initial block.

    Returns (labels, diagnostics) where diagnostics contains the number of
    refits and the fraction of bars where the expanding label differs from the
    full-sample label.
    """
    from sklearn.mixture import GaussianMixture

    feats = _features(df_ohlc, freq, stem=stem, window_scale=window_scale)
    vol_vals = feats["vol"].fillna(feats["vol"].median()).values
    eps = 1e-12
    log_vol = np.log(np.maximum(vol_vals, eps))
    n = len(log_vol)

    labels = np.full(n, 0, dtype=int)
    refit_count = 0

    step = max(1, n // 50)

    last_gmm = None
    for end in range(min_train_bars, n + 1, step):
        X_train = log_vol[:end]
        X_valid = X_train[~np.isnan(X_train) & np.isfinite(X_train)].reshape(-1, 1)
        if len(X_valid) < n_components * 2:
            continue
        if np.std(X_valid) < 1e-10:
            continue
        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=3)
        gmm.fit(X_valid)
        last_gmm = gmm
        refit_count += 1
        crisis_cluster = int(np.argmax(gmm.means_.ravel()))
        pred_end = min(end + step, n + 1)
        chunk = log_vol[end - step if end > min_train_bars else 0 : pred_end]
        chunk_pred = gmm.predict(chunk.reshape(-1, 1))
        chunk_labels = (chunk_pred == crisis_cluster).astype(int)
        start_idx = end - step if end > min_train_bars else 0
        labels[start_idx:start_idx + len(chunk_labels)] = chunk_labels

    # Label any remaining tail bars with the last model
    if last_gmm is not None:
        tail_start = min_train_bars + ((n - min_train_bars) // step) * step
        if tail_start < n:
            chunk = log_vol[tail_start:n]
            crisis_cluster = int(np.argmax(last_gmm.means_.ravel()))
            chunk_pred = last_gmm.predict(chunk.reshape(-1, 1))
            labels[tail_start:n] = (chunk_pred == crisis_cluster).astype(int)

    label_series = pd.Series(labels, index=feats.index, dtype=int)

    # Compare with full-sample labels for diagnostics
    full_labels, _ = _fit_regime_model(feats, model=model, n_components=n_components, freq=freq)
    agree_frac = float((label_series.values == full_labels.values).mean()) if len(full_labels) else np.nan

    diagnostics = {
        "refit_count": refit_count,
        "full_vs_expanding_agreement": agree_frac,
    }
    return label_series, diagnostics


# ── CL Roll-Week Analysis ──────────────────────────────────────────────────

def _cl_roll_week_analysis(
    aligned: Dict[str, pd.Series],
    index_5m: pd.DatetimeIndex,
    roll_dates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute ARI separately for roll-week and non-roll-week periods for CL.

    If *roll_dates* is not provided, we approximate roll weeks as the 3rd week
    of each month (typical WTI roll pattern: days 15-21).
    """
    if roll_dates is None:
        idx_ny = index_5m.tz_convert(TZ) if index_5m.tz is not None else index_5m
        days = idx_ny.day
        roll_mask = (days >= 15) & (days <= 21)
    else:
        roll_mask = np.zeros(len(index_5m), dtype=bool)
        idx_ny = index_5m.tz_convert(TZ) if index_5m.tz is not None else index_5m
        idx_dates = idx_ny.normalize()
        for rd in roll_dates:
            rd_ts = pd.Timestamp(rd)
            start = rd_ts - pd.Timedelta(days=2)
            end = rd_ts + pd.Timedelta(days=2)
            # Compare as tz-naive dates
            start_naive = start.tz_localize(None) if start.tz else start
            end_naive = end.tz_localize(None) if end.tz else end
            for i, ts_date in enumerate(idx_dates):
                d = ts_date.tz_localize(None) if ts_date.tz else ts_date
                if start_naive <= d <= end_naive:
                    roll_mask[i] = True

    roll_index = index_5m[roll_mask]
    nonroll_index = index_5m[~roll_mask]

    roll_ari = _compute_ari_matrix(aligned, roll_index) if len(roll_index) >= 50 else pd.DataFrame()
    nonroll_ari = _compute_ari_matrix(aligned, nonroll_index) if len(nonroll_index) >= 50 else pd.DataFrame()

    return {
        "roll_week_mean_ari": _mean_offdiag(roll_ari),
        "nonroll_week_mean_ari": _mean_offdiag(nonroll_ari),
        "roll_week_bars": len(roll_index),
        "nonroll_week_bars": len(nonroll_index),
        "roll_week_ari_matrix": roll_ari,
        "nonroll_week_ari_matrix": nonroll_ari,
    }


# ── Time-of-Day Analysis ────────────────────────────────────────────────────

def _compute_tod_crisis_distribution(aligned: Dict[str, pd.Series]) -> pd.DataFrame:
    """Crisis-label share by hour-of-day (NY time) for each frequency."""
    rows: List[Dict[str, Any]] = []
    for freq in FREQS:
        s = aligned.get(freq)
        if s is None or s.empty:
            continue
        idx_ny = s.index.tz_convert(TZ) if s.index.tz is not None else s.index
        hours = idx_ny.hour
        for h in sorted(hours.unique()):
            mask = hours == h
            vals = s.values[mask]
            rows.append({
                "freq": freq, "hour": int(h),
                "crisis_share": float(100.0 * (vals == 1).mean()),
                "n_bars": int(len(vals)),
            })
    return pd.DataFrame(rows)


def _compute_tod_adjusted_volatility(
    df_5m: pd.DataFrame,
    freq: str,
    stem: str,
    window_scale: float = DEFAULT_WINDOW_SCALE,
) -> pd.DataFrame:
    """Build features with TOD-adjusted vol (divide by median vol per hour)."""
    ohlc = resample_ohlc(df_5m, freq)
    feats = _features(ohlc, freq, stem=stem, window_scale=window_scale)
    if freq == "1d":
        return feats
    idx_ny = feats.index.tz_convert(TZ) if feats.index.tz is not None else feats.index
    hours = idx_ny.hour
    median_vol_by_hour = feats.groupby(hours)["vol"].transform("median")
    median_vol_by_hour = median_vol_by_hour.replace(0, np.nan).fillna(feats["vol"].median())
    feats_adj = feats.copy()
    feats_adj["vol"] = feats["vol"] / median_vol_by_hour
    feats_adj["vol"] = feats_adj["vol"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return feats_adj


# ── Window Subsetting ────────────────────────────────────────────────────────

def _subset_index_by_dates(
    index_5m: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex:
    """Select 5m timestamps whose NY calendar date is in [start, end]."""
    if index_5m.tz is None:
        idx = index_5m.tz_localize(TZ, ambiguous="infer")
    else:
        idx = index_5m.tz_convert(TZ)
    d = idx.normalize()
    start = pd.Timestamp(start_date).tz_localize(TZ)
    end = pd.Timestamp(end_date).tz_localize(TZ)
    mask = (d >= start.normalize()) & (d <= end.normalize())
    return index_5m[mask]


# ── Daily & Rolling Summaries ────────────────────────────────────────────────

def _compute_daily_outputs(
    aligned: Dict[str, pd.Series],
    rolling_days: int = DEFAULT_ROLLING_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build daily summaries and rolling ARI tables."""
    s_5m = aligned.get("5m")
    if s_5m is None or s_5m.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    index_5m = s_5m.index
    day_labels = index_5m.tz_convert(TZ).normalize()
    days = pd.DatetimeIndex(day_labels.unique()).sort_values()

    daily_rows: List[Dict[str, Any]] = []
    daily_pair_rows: List[Dict[str, Any]] = []
    rolling_rows: List[Dict[str, Any]] = []
    rolling_pair_rows: List[Dict[str, Any]] = []

    for day in days:
        mask = day_labels == day
        day_index = index_5m[mask]
        ari_df = _compute_ari_matrix(aligned, day_index)
        row: Dict[str, Any] = {
            "date": day.strftime("%Y-%m-%d"),
            "bars_5m": int(len(day_index)),
            "mean_offdiag_ari": _mean_offdiag(ari_df),
        }
        for freq in FREQS:
            sub = aligned[freq].reindex(day_index)
            row[f"crisis_share_{freq}"] = float(100.0 * (sub == 1).mean()) if len(day_index) else np.nan
        daily_rows.append(row)
        for i, fa in enumerate(FREQS):
            for j, fb in enumerate(FREQS):
                if j <= i:
                    continue
                daily_pair_rows.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "freq_a": fa, "freq_b": fb,
                    "ari": ari_df.loc[fa, fb],
                })

    for end_idx in range(rolling_days - 1, len(days)):
        window_days = days[end_idx - rolling_days + 1: end_idx + 1]
        mask = pd.Series(day_labels, index=index_5m).isin(window_days).values
        window_index = index_5m[mask]
        ari_df = _compute_ari_matrix(aligned, window_index)
        rolling_rows.append({
            "window_start": window_days[0].strftime("%Y-%m-%d"),
            "window_end": window_days[-1].strftime("%Y-%m-%d"),
            "days_in_window": len(window_days),
            "bars_5m": int(len(window_index)),
            "mean_offdiag_ari": _mean_offdiag(ari_df),
        })
        for i, fa in enumerate(FREQS):
            for j, fb in enumerate(FREQS):
                if j <= i:
                    continue
                rolling_pair_rows.append({
                    "window_start": window_days[0].strftime("%Y-%m-%d"),
                    "window_end": window_days[-1].strftime("%Y-%m-%d"),
                    "freq_a": fa, "freq_b": fb,
                    "ari": ari_df.loc[fa, fb],
                })

    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(daily_pair_rows),
        pd.DataFrame(rolling_rows),
        pd.DataFrame(rolling_pair_rows),
    )


# ── Visualization ────────────────────────────────────────────────────────────

def _plot_timeline(aligned: Dict[str, pd.Series], asset_name: str, out_path: Path) -> None:
    """Gantt-style 4-panel timeline: X=time, Y=1d/1h/15m/5m, red=crisis."""
    matplotlib.use("Agg")
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(4, 1, figsize=(12, 4), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    order = ("1d", "1h", "15m", "5m")
    s_5m = aligned.get("5m")
    t_common = s_5m.index if s_5m is not None and not s_5m.empty else pd.DatetimeIndex([])

    for ax, freq in zip(axes, order):
        s = aligned.get(freq)
        if s is None or s.empty:
            ax.set_ylabel(freq)
            ax.set_yticks([])
            continue
        s = s.reindex(t_common).ffill().bfill().fillna(0) if len(t_common) else s
        t = s.index
        crisis = (s == 1).values
        ax.fill_between(t, 0, 1, where=crisis, color="darkred", alpha=0.8, step="post")
        ax.fill_between(t, 0, 1, where=(s != 1).values, color="skyblue", alpha=0.6, step="post")
        ax.set_ylabel(freq, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.yaxis.set_label_position("right")

    axes[0].set_title(f"{asset_name}: regime by frequency (red = crisis)")
    try:
        tz = t_common[0].tz if len(t_common) else None
    except Exception:
        tz = None
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=tz))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_rolling_ari(
    rolling_df: pd.DataFrame,
    rolling_pair_df: pd.DataFrame,
    asset_name: str,
    out_path: Path,
) -> None:
    """Plot rolling 7-day ARI by day."""
    matplotlib.use("Agg")
    if rolling_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(rolling_df["window_end"])
    ax.plot(x, rolling_df["mean_offdiag_ari"], color="black", linewidth=2, label="mean off-diag")

    for (fa, fb), grp in rolling_pair_df.groupby(["freq_a", "freq_b"]):
        ax.plot(pd.to_datetime(grp["window_end"]), grp["ari"],
                alpha=0.6, linewidth=1.2, label=f"{fa}-{fb}")

    ax.set_title(f"{asset_name}: Rolling {DEFAULT_ROLLING_DAYS}-day ARI")
    ax.set_ylabel("ARI")
    ax.set_xlabel("Window end date")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_ari_heatmap(ari_matrix: pd.DataFrame, asset_name: str, out_path: Path) -> None:
    """Cross-frequency ARI heatmap."""
    n = len(ari_matrix)
    fig, ax = plt.subplots(figsize=(5 + n * 0.3, 4 + n * 0.3))
    data = ari_matrix.values.astype(float)
    im = ax.imshow(data, vmin=-0.1, vmax=1.0, cmap="RdYlGn", aspect="auto")
    labels = list(ari_matrix.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if i != j else "normal",
                    color="white" if v < 0.4 else "black")
    ax.set_title(f"{asset_name} — Cross-Frequency ARI\n(threshold = {ARI_THRESHOLD})",
                 fontsize=12, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("ARI", fontsize=10)
    cbar.ax.axhline(y=ARI_THRESHOLD, color="black", linewidth=1.5, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Core Analysis ────────────────────────────────────────────────────────────

def analyze_asset(
    asset_name: str,
    df_5m: pd.DataFrame,
    model: str = "gmm",
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    rolling_days: int = DEFAULT_ROLLING_DAYS,
    event_window: Optional[Tuple[str, str]] = None,
    calm_window: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Run multi-frequency analysis for one already-loaded asset.

    This is the main analytical function. It:
    1. Resamples 5m OHLC to all frequencies
    2. Builds features and fits regimes per frequency
    3. Aligns everything to 5m timestamps
    4. Computes ARI/AMI/VI matrices, permutation p-values
    5. Runs event/calm window and TOD analyses
    6. Computes daily/rolling summaries
    """
    stem = asset_name

    # Fit regimes per frequency
    regimes_by_freq: Dict[str, pd.Series] = {}
    fallback_flags: Dict[str, bool] = {}
    for freq in FREQS:
        ohlc = resample_ohlc(df_5m, freq)
        feats = _features(ohlc, freq, stem=stem, window_scale=window_scale)
        labels, fb = _fit_regime_model(feats, model=model, n_components=n_components, freq=freq)
        regimes_by_freq[freq] = labels
        fallback_flags[freq] = fb

    # Align to 5m
    aligned = _align_regimes_to_5m(regimes_by_freq, df_5m.index)
    for freq in FREQS:
        pct = 100.0 * (aligned[freq] == 1).mean()
        if pct < 1.0 or pct > 99.0:
            logger.warning("%s %s: crisis share %.1f%% (near trivial)", asset_name, freq, pct)
        else:
            logger.info("%s %s: crisis share %.1f%%", asset_name, freq, pct)

    # TOD analysis
    tod_crisis = _compute_tod_crisis_distribution(aligned)
    tod_regimes_by_freq: Dict[str, pd.Series] = {}
    for freq in FREQS:
        tod_feats = _compute_tod_adjusted_volatility(df_5m, freq, stem, window_scale=window_scale)
        tod_labels, _ = _fit_regime_model(tod_feats, model=model, n_components=n_components, freq=freq)
        tod_regimes_by_freq[freq] = tod_labels
    tod_aligned = _align_regimes_to_5m(tod_regimes_by_freq, df_5m.index)
    tod_ari_df = _compute_ari_matrix(tod_aligned)

    # Metrics
    ari_df = _compute_ari_matrix(aligned)
    extra_metrics = _compute_extra_metrics(aligned)
    perm_p, perm_ci = _permute_pvalue_mean_offdiag_ari(aligned, n_perm=500, seed=42)

    # Event/calm windows
    event_ari_df = pd.DataFrame()
    calm_ari_df = pd.DataFrame()
    if event_window:
        event_index = _subset_index_by_dates(df_5m.index, event_window[0], event_window[1])
        if len(event_index):
            event_ari_df = _compute_ari_matrix(aligned, event_index)
    if calm_window:
        calm_index = _subset_index_by_dates(df_5m.index, calm_window[0], calm_window[1])
        if len(calm_index):
            calm_ari_df = _compute_ari_matrix(aligned, calm_index)

    # Daily/rolling
    daily_df, daily_pair_df, rolling_df, rolling_pair_df = _compute_daily_outputs(
        aligned, rolling_days=rolling_days)

    crisis_shares = {freq: float(100.0 * (aligned[freq] == 1).mean()) for freq in FREQS}

    # Block-permutation test (preserves autocorrelation)
    block_perm_p, block_perm_ci = _block_permute_pvalue_mean_offdiag_ari(aligned, n_perm=500, seed=42)

    # GMM fit diagnostics per resolution
    gmm_diagnostics: Dict[str, Dict[str, Any]] = {}
    for freq in FREQS:
        ohlc = resample_ohlc(df_5m, freq)
        gmm_diagnostics[freq] = _gmm_fit_diagnostics(
            ohlc, freq, stem=stem, n_components=n_components, window_scale=window_scale)

    # Expanding-window regime fitting (no look-ahead)
    expanding_regimes: Dict[str, pd.Series] = {}
    expanding_diag: Dict[str, Dict[str, Any]] = {}
    for freq in FREQS:
        ohlc = resample_ohlc(df_5m, freq)
        exp_labels, exp_info = _fit_regime_expanding(
            ohlc, freq, stem=stem, model=model,
            n_components=n_components, window_scale=window_scale,
        )
        expanding_regimes[freq] = exp_labels
        expanding_diag[freq] = exp_info
    expanding_aligned = _align_regimes_to_5m(expanding_regimes, df_5m.index)
    expanding_ari_df = _compute_ari_matrix(expanding_aligned)

    # CL roll-week analysis
    cl_roll_result: Optional[Dict[str, Any]] = None
    if stem == "CL":
        cl_roll_result = _cl_roll_week_analysis(aligned, df_5m.index)

    # Rolling ARI distribution (median, IQR)
    rolling_ari_median = np.nan
    rolling_ari_q25 = np.nan
    rolling_ari_q75 = np.nan
    if not rolling_df.empty:
        ari_vals = rolling_df["mean_offdiag_ari"].dropna()
        if len(ari_vals):
            rolling_ari_median = float(ari_vals.median())
            rolling_ari_q25 = float(ari_vals.quantile(0.25))
            rolling_ari_q75 = float(ari_vals.quantile(0.75))

    return {
        "asset_name": asset_name,
        "stem": stem,
        "model": str(model),
        "n_components": int(n_components),
        "window_scale": float(window_scale),
        "rolling_days": int(rolling_days),
        "ari_matrix": ari_df,
        "ami_matrix": extra_metrics["ami"],
        "vi_matrix": extra_metrics["vi"],
        "event_ari_matrix": event_ari_df,
        "calm_ari_matrix": calm_ari_df,
        "event_window": event_window,
        "calm_window": calm_window,
        "regimes_aligned": aligned,
        "daily_df": daily_df,
        "daily_pair_df": daily_pair_df,
        "rolling_df": rolling_df,
        "rolling_pair_df": rolling_pair_df,
        "crisis_shares": crisis_shares,
        "fallback_flags": fallback_flags,
        "tod_crisis_distribution": tod_crisis,
        "tod_adjusted_ari_matrix": tod_ari_df,
        "tod_adjusted_mean_ari": _mean_offdiag(tod_ari_df),
        "overall_mean_ari": _mean_offdiag(ari_df),
        "overall_mean_ari_pvalue_perm": perm_p,
        "overall_mean_ari_null_ci": perm_ci,
        "block_perm_pvalue": block_perm_p,
        "block_perm_null_ci": block_perm_ci,
        "event_mean_ari": _mean_offdiag(event_ari_df) if not event_ari_df.empty else None,
        "calm_mean_ari": _mean_offdiag(calm_ari_df) if not calm_ari_df.empty else None,
        "latest_rolling_7d_mean_ari": None if rolling_df.empty else float(rolling_df["mean_offdiag_ari"].iloc[-1]),
        "rolling_ari_median": rolling_ari_median,
        "rolling_ari_q25": rolling_ari_q25,
        "rolling_ari_q75": rolling_ari_q75,
        "gmm_diagnostics": gmm_diagnostics,
        "expanding_ari_matrix": expanding_ari_df,
        "expanding_mean_ari": _mean_offdiag(expanding_ari_df),
        "expanding_diagnostics": expanding_diag,
        "cl_roll_analysis": cl_roll_result,
    }


def run_robustness(
    df_5m: pd.DataFrame,
    asset_name: str,
    k_values: Tuple[int, ...] = (2, 3),
    window_scales: Tuple[float, ...] = (0.5, 1.0, 2.0),
    rolling_days: int = DEFAULT_ROLLING_DAYS,
) -> pd.DataFrame:
    """Run parameter sensitivity sweeps over K and window scales."""
    rows: List[Dict[str, Any]] = []
    for n_components in k_values:
        for window_scale in window_scales:
            analysis = analyze_asset(
                asset_name, df_5m,
                n_components=n_components,
                window_scale=window_scale,
                rolling_days=rolling_days,
            )
            row: Dict[str, Any] = {
                "asset": asset_name,
                "k": int(n_components),
                "window_scale": float(window_scale),
                "overall_mean_ari": analysis["overall_mean_ari"],
                "latest_rolling_mean_ari": analysis["latest_rolling_7d_mean_ari"],
            }
            for freq, share in analysis["crisis_shares"].items():
                row[f"crisis_share_{freq}"] = share
            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    baseline = (
        summary[(summary["k"] == DEFAULT_GMM_K) & (summary["window_scale"] == DEFAULT_WINDOW_SCALE)]
        .set_index("asset")
        .rename(columns={
            "overall_mean_ari": "baseline_overall_mean_ari",
            "latest_rolling_mean_ari": "baseline_latest_rolling_mean_ari",
        })[["baseline_overall_mean_ari", "baseline_latest_rolling_mean_ari"]]
    )
    summary = summary.join(baseline, on="asset")
    summary["delta_overall_mean_ari"] = summary["overall_mean_ari"] - summary["baseline_overall_mean_ari"]
    return summary.sort_values(["asset", "k", "window_scale"]).reset_index(drop=True)


# ── Validator Class ──────────────────────────────────────────────────────────

class ResValidator(BaseValidator):
    """Resolution Invariance validator (Paper 2).

    Tests whether regime labels agree across time frequencies (5m/15m/1h/1d).
    """

    name = "res"

    def validate(
        self,
        prices: Optional[Dict[str, pd.DataFrame]] = None,
        labels: Optional[Dict[str, Dict[str, pd.Series]]] = None,
    ) -> Dict[str, Any]:
        """
        Run resolution invariance test.

        Parameters
        ----------
        prices : dict, optional
            ``{asset: ohlcv_5m_dataframe}``. If None, loaded from config.
        labels : dict, optional
            ``{asset: {freq: labels_series}}``. If None, computed internally.
        """
        res_cfg = self.test_cfg
        model_name = res_cfg.get("model", "gmm")
        n_components = res_cfg.get("n_states", DEFAULT_GMM_K)
        episode = res_cfg.get("episode")
        start = res_cfg.get("start")
        end = res_cfg.get("end")

        # Resolve episode windows
        event_window = None
        calm_window = None
        if episode and episode in EPISODES:
            event_window, calm_window = EPISODES[episode]
        elif res_cfg.get("event_window") and res_cfg.get("calm_window"):
            event_window = tuple(res_cfg["event_window"])
            calm_window = tuple(res_cfg["calm_window"])

        run_dir = self._make_run_dir()

        # Load data if not provided
        if prices is None:
            prices = self._load_5m_data(res_cfg, start, end)

        if not prices:
            raise ValueError("No price data available for resolution invariance test")

        logger.info("=== Resolution Invariance ===")
        logger.info("Assets: %s, Model: %s (K=%d)", list(prices.keys()), model_name, n_components)

        all_results: Dict[str, Dict] = {}

        for asset_name, df_5m in prices.items():
            logger.info("--- %s ---", asset_name)

            if labels and asset_name in labels:
                # Pre-computed labels: wrap in aligned dict
                aligned = labels[asset_name]
                analysis = self._analyze_with_labels(asset_name, df_5m, aligned,
                                                      event_window, calm_window)
            else:
                analysis = analyze_asset(
                    asset_name, df_5m,
                    model=model_name,
                    n_components=n_components,
                    event_window=event_window,
                    calm_window=calm_window,
                )

            # Attribution (if enabled)
            if res_cfg.get("attribution", False) and "ari_matrix" in analysis:
                from mrv.validator.attribution import (
                    freq_pair_attribution, temporal_attribution, generate_attribution_summary,
                )
                pair_attr = freq_pair_attribution(analysis["ari_matrix"])
                attr_result: Dict[str, Any] = {"freq_pairs": pair_attr}
                # Temporal hotspot for the worst frequency pair
                if pair_attr and "regimes_aligned" in analysis:
                    worst = pair_attr[0]
                    aligned = analysis["regimes_aligned"]
                    if worst["freq_a"] in aligned and worst["freq_b"] in aligned:
                        temp = temporal_attribution(
                            aligned[worst["freq_a"]], aligned[worst["freq_b"]])
                        if not temp.empty:
                            attr_result["temporal"] = temp
                            temp.to_csv(run_dir / f"{asset_name}_attribution_timeline.csv", index=False)
                attr_result["summary"] = generate_attribution_summary(attr_result, "res")
                analysis["attribution"] = attr_result
                logger.info("  Attribution: worst pair=%s-%s (ARI=%.3f)",
                            pair_attr[0]["freq_a"], pair_attr[0]["freq_b"], pair_attr[0]["ari"])

            # Business impact (if impact_fn provided)
            if self.impact_fn is not None and "regimes_aligned" in analysis:
                aligned = analysis["regimes_aligned"]
                price = df_5m["Close"] if "Close" in df_5m.columns else df_5m["close"]
                freq_labels = {freq: aligned[freq].values for freq in FREQS if freq in aligned}
                impact = self._compute_impact_matrix(freq_labels, price)
                if impact is not None:
                    analysis["impact"] = impact
                    logger.info("  Impact: max_delta=%.4f worst_pair=%s",
                                impact["max_delta"], impact["worst_pair"])

            all_results[asset_name] = analysis

            # Save per-asset outputs
            self._save_asset_outputs(asset_name, analysis, run_dir)

            # HMM robustness
            if not (labels and asset_name in labels):
                hmm_analysis = analyze_asset(
                    asset_name, df_5m, model="hmm",
                    n_components=n_components,
                    event_window=event_window, calm_window=calm_window,
                )
                hmm_analysis["ari_matrix"].to_csv(run_dir / f"{asset_name}_hmm_cross_freq_ari.csv")
                analysis["hmm_overall_mean_ari"] = hmm_analysis["overall_mean_ari"]

                # Calendar-window robustness
                cal_result = self._calendar_window_robustness(
                    asset_name, df_5m, run_dir, model=model_name, n_components=n_components)
                analysis["calendar_window_mean_ari"] = cal_result.get("mean_offdiag_ari")

        # Save JSON
        json_path = run_dir / "result.json"
        json_data = self._build_json(all_results, model_name, n_components, res_cfg)
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.json_path = json_path
        logger.info("JSON -> %s", json_path)

        # Save text summary
        self._write_text_report(run_dir / "summary.txt", all_results, model_name, n_components, res_cfg)

        # Pipeline summary CSV
        self._save_pipeline_summary(run_dir, all_results)

        logger.info("=== Output: %s ===", run_dir)
        self.results = all_results
        return {"run_dir": str(run_dir), "json_path": str(json_path), "assets": all_results}

    # ── Internal helpers ─────────────────────────────────────────────────

    def _load_5m_data(
        self, res_cfg: Dict, start: Optional[str], end: Optional[str],
    ) -> Dict[str, pd.DataFrame]:
        """Load 5m OHLCV data from config paths.

        For res validator, assets map to a list of paths (one per freq),
        or a single 5m path that gets resampled.
        """
        assets_map = res_cfg.get("assets", {})
        prices: Dict[str, pd.DataFrame] = {}

        for name, paths in assets_map.items():
            # Support both single path and list of paths
            if isinstance(paths, list):
                # Use the first (5m) path
                path = Path(paths[0])
            else:
                path = Path(paths)

            if not path.exists():
                logger.warning("Skip %s: %s not found", name, path)
                continue

            df = load_ohlcv(path)
            if start:
                df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
            if end:
                df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]

            if len(df) < 50:
                logger.warning("Skip %s: too few data (%d)", name, len(df))
                continue

            prices[name] = df
        return prices

    def _analyze_with_labels(
        self,
        asset_name: str,
        df_5m: pd.DataFrame,
        aligned: Dict[str, pd.Series],
        event_window: Optional[Tuple[str, str]],
        calm_window: Optional[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Build analysis result from pre-computed aligned labels."""
        ari_df = _compute_ari_matrix(aligned)
        extra = _compute_extra_metrics(aligned)
        perm_p, perm_ci = _permute_pvalue_mean_offdiag_ari(aligned)
        daily_df, daily_pair_df, rolling_df, rolling_pair_df = _compute_daily_outputs(aligned)
        crisis_shares = {freq: float(100.0 * (aligned[freq] == 1).mean()) for freq in FREQS}

        event_ari_df = pd.DataFrame()
        calm_ari_df = pd.DataFrame()
        if event_window:
            idx = _subset_index_by_dates(df_5m.index, event_window[0], event_window[1])
            if len(idx):
                event_ari_df = _compute_ari_matrix(aligned, idx)
        if calm_window:
            idx = _subset_index_by_dates(df_5m.index, calm_window[0], calm_window[1])
            if len(idx):
                calm_ari_df = _compute_ari_matrix(aligned, idx)

        return {
            "asset_name": asset_name,
            "ari_matrix": ari_df,
            "ami_matrix": extra["ami"],
            "vi_matrix": extra["vi"],
            "event_ari_matrix": event_ari_df,
            "calm_ari_matrix": calm_ari_df,
            "regimes_aligned": aligned,
            "daily_df": daily_df,
            "daily_pair_df": daily_pair_df,
            "rolling_df": rolling_df,
            "rolling_pair_df": rolling_pair_df,
            "crisis_shares": crisis_shares,
            "overall_mean_ari": _mean_offdiag(ari_df),
            "overall_mean_ari_pvalue_perm": perm_p,
            "overall_mean_ari_null_ci": perm_ci,
            "event_mean_ari": _mean_offdiag(event_ari_df) if not event_ari_df.empty else None,
            "calm_mean_ari": _mean_offdiag(calm_ari_df) if not calm_ari_df.empty else None,
            "latest_rolling_7d_mean_ari": None if rolling_df.empty else float(rolling_df["mean_offdiag_ari"].iloc[-1]),
        }

    def _save_asset_outputs(self, asset_name: str, analysis: Dict, run_dir: Path) -> None:
        """Save CSVs and plots for one asset."""
        analysis["ari_matrix"].to_csv(run_dir / f"{asset_name}_cross_freq_ari.csv")
        analysis["ami_matrix"].to_csv(run_dir / f"{asset_name}_cross_freq_ami.csv")
        analysis["vi_matrix"].to_csv(run_dir / f"{asset_name}_cross_freq_vi.csv")

        if isinstance(analysis.get("tod_crisis_distribution"), pd.DataFrame):
            if not analysis["tod_crisis_distribution"].empty:
                analysis["tod_crisis_distribution"].to_csv(
                    run_dir / f"{asset_name}_tod_crisis_distribution.csv", index=False)
        if isinstance(analysis.get("tod_adjusted_ari_matrix"), pd.DataFrame):
            analysis["tod_adjusted_ari_matrix"].to_csv(
                run_dir / f"{asset_name}_tod_adjusted_cross_freq_ari.csv")

        if isinstance(analysis.get("event_ari_matrix"), pd.DataFrame) and not analysis["event_ari_matrix"].empty:
            analysis["event_ari_matrix"].to_csv(run_dir / f"{asset_name}_event_cross_freq_ari.csv")
        if isinstance(analysis.get("calm_ari_matrix"), pd.DataFrame) and not analysis["calm_ari_matrix"].empty:
            analysis["calm_ari_matrix"].to_csv(run_dir / f"{asset_name}_calm_cross_freq_ari.csv")

        # Plots
        if "regimes_aligned" in analysis:
            _plot_timeline(analysis["regimes_aligned"], asset_name, run_dir / f"{asset_name}_timeline.png")
            _plot_ari_heatmap(analysis["ari_matrix"], asset_name, run_dir / f"{asset_name}_ari_heatmap.png")

        # Daily/rolling CSVs
        if isinstance(analysis.get("daily_df"), pd.DataFrame) and not analysis["daily_df"].empty:
            analysis["daily_df"].to_csv(run_dir / f"{asset_name}_daily_summary.csv", index=False)
        if isinstance(analysis.get("daily_pair_df"), pd.DataFrame) and not analysis["daily_pair_df"].empty:
            analysis["daily_pair_df"].to_csv(run_dir / f"{asset_name}_daily_pairwise_ari.csv", index=False)
        if isinstance(analysis.get("rolling_df"), pd.DataFrame) and not analysis["rolling_df"].empty:
            analysis["rolling_df"].to_csv(run_dir / f"{asset_name}_rolling_7d_ari.csv", index=False)
            if isinstance(analysis.get("rolling_pair_df"), pd.DataFrame):
                analysis["rolling_pair_df"].to_csv(
                    run_dir / f"{asset_name}_rolling_7d_pairwise_ari.csv", index=False)
                _plot_rolling_ari(analysis["rolling_df"], analysis["rolling_pair_df"],
                                  asset_name, run_dir / f"{asset_name}_rolling_7d_ari.png")

        # Fallback triggers
        fallback_rows = []
        for freq, fb in analysis.get("fallback_flags", {}).items():
            fallback_rows.append({"freq": freq, "model": analysis.get("model", "gmm"), "fallback_triggered": fb})
        if fallback_rows:
            pd.DataFrame(fallback_rows).to_csv(run_dir / f"{asset_name}_fallback_triggers.csv", index=False)

        # Impact matrix
        if "impact" in analysis and analysis["impact"] is not None:
            analysis["impact"]["delta_matrix"].to_csv(run_dir / f"{asset_name}_impact_matrix.csv")

        # GMM diagnostics
        if analysis.get("gmm_diagnostics"):
            diag_rows = []
            for freq, diag in analysis["gmm_diagnostics"].items():
                row = {"freq": freq, **{k: v for k, v in diag.items() if k not in ("means", "stds", "weights")}}
                if "means" in diag:
                    for ci, m in enumerate(diag["means"]):
                        row[f"mean_{ci}"] = m
                if "stds" in diag:
                    for ci, s in enumerate(diag["stds"]):
                        row[f"std_{ci}"] = s
                if "weights" in diag:
                    for ci, w in enumerate(diag["weights"]):
                        row[f"weight_{ci}"] = w
                diag_rows.append(row)
            pd.DataFrame(diag_rows).to_csv(run_dir / f"{asset_name}_gmm_diagnostics.csv", index=False)

        # Expanding-window ARI
        if isinstance(analysis.get("expanding_ari_matrix"), pd.DataFrame):
            analysis["expanding_ari_matrix"].to_csv(run_dir / f"{asset_name}_expanding_cross_freq_ari.csv")

        # CL roll-week analysis
        if analysis.get("cl_roll_analysis"):
            roll_info = analysis["cl_roll_analysis"]
            roll_summary = pd.DataFrame([{
                "roll_week_mean_ari": roll_info["roll_week_mean_ari"],
                "nonroll_week_mean_ari": roll_info["nonroll_week_mean_ari"],
                "roll_week_bars": roll_info["roll_week_bars"],
                "nonroll_week_bars": roll_info["nonroll_week_bars"],
            }])
            roll_summary.to_csv(run_dir / f"{asset_name}_roll_week_ari.csv", index=False)

        logger.info("Saved %s outputs -> %s", asset_name, run_dir)

    def _calendar_window_robustness(
        self,
        asset_name: str,
        df_5m: pd.DataFrame,
        run_dir: Path,
        calendar_window: str = "6h",
        model: str = "gmm",
        n_components: int = DEFAULT_GMM_K,
    ) -> Dict[str, Any]:
        """Fixed calendar-time window for all frequencies."""
        stem = asset_name
        regimes_by_freq: Dict[str, pd.Series] = {}
        for freq in FREQS:
            ohlc = resample_ohlc(df_5m, freq)
            feats = _features(ohlc, freq, stem=stem, calendar_window=calendar_window)
            labels, _ = _fit_regime_model(feats, model=model, n_components=n_components, freq=freq)
            regimes_by_freq[freq] = labels

        aligned = _align_regimes_to_5m(regimes_by_freq, df_5m.index)
        ari_df = _compute_ari_matrix(aligned)
        ari_df.to_csv(run_dir / f"{asset_name}_calendar_window_cross_freq_ari.csv")
        logger.info("Saved %s calendar-window (%s) ARI", asset_name, calendar_window)
        return {"calendar_window": calendar_window, "ari_matrix": ari_df, "mean_offdiag_ari": _mean_offdiag(ari_df)}

    def _build_json(self, results: Dict, model: str, n_components: int, res_cfg: Dict) -> Dict:
        all_ari = [r["overall_mean_ari"] for r in results.values()
                   if r.get("overall_mean_ari") is not None and np.isfinite(r["overall_mean_ari"])]
        overall_ari = float(np.mean(all_ari)) if all_ari else None

        assets_json = {}
        for name, r in results.items():
            ari_df = r["ari_matrix"]
            assets_json[name] = {
                "n_obs": int(ari_df.shape[0]) if not ari_df.empty else 0,
                "overall_mean_ari": round(r["overall_mean_ari"], 6) if r.get("overall_mean_ari") is not None else None,
                "partition_pass": r.get("overall_mean_ari") is not None and r["overall_mean_ari"] >= ARI_THRESHOLD,
                "pvalue_perm": r.get("overall_mean_ari_pvalue_perm"),
                "null_ci": r.get("overall_mean_ari_null_ci"),
                "block_perm_pvalue": r.get("block_perm_pvalue"),
                "block_perm_null_ci": r.get("block_perm_null_ci"),
                "event_mean_ari": r.get("event_mean_ari"),
                "calm_mean_ari": r.get("calm_mean_ari"),
                "crisis_shares": r.get("crisis_shares", {}),
                "rolling_ari_median": r.get("rolling_ari_median"),
                "rolling_ari_q25": r.get("rolling_ari_q25"),
                "rolling_ari_q75": r.get("rolling_ari_q75"),
                "expanding_mean_ari": r.get("expanding_mean_ari"),
                "ari_matrix": {
                    "labels": list(ari_df.columns),
                    "values": [[round(v, 6) for v in row] for row in ari_df.values.tolist()],
                },
                "heatmap_png": f"{name}_ari_heatmap.png",
                "timeline_png": f"{name}_timeline.png",
            }

        return {
            "test": "resolution_invariance",
            "generated": datetime.now().isoformat(),
            "model": model.upper(),
            "n_components": n_components,
            "frequencies": list(FREQS),
            "date_range": {"start": res_cfg.get("start"), "end": res_cfg.get("end")},
            "ari_threshold": ARI_THRESHOLD,
            "overall_mean_ari": round(overall_ari, 6) if overall_ari is not None else None,
            "partition_pass": overall_ari is not None and overall_ari >= ARI_THRESHOLD,
            "assets": assets_json,
        }

    def _write_text_report(self, path: Path, results: Dict, model: str, n_components: int, res_cfg: Dict) -> None:
        lines = [
            "=" * 60, "MRV Resolution Invariance Report", "=" * 60, "",
            f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Model:        {model.upper()} (K={n_components})",
            f"Frequencies:  {', '.join(FREQS)}",
            f"Period:       {res_cfg.get('start', '?')} -> {res_cfg.get('end', '?')}", "",
        ]
        for asset, r in results.items():
            mean_ari = r.get("overall_mean_ari")
            status = "PASS" if mean_ari is not None and mean_ari >= ARI_THRESHOLD else "FAIL"
            lines += [
                f"--- {asset} ---",
                f"  Mean off-diag ARI: {mean_ari:.3f} [{status}]" if mean_ari is not None else f"  Mean off-diag ARI: N/A",
                f"  Perm p-value:      {r.get('overall_mean_ari_pvalue_perm', 'N/A')}",
                f"  Block perm p-val:  {r.get('block_perm_pvalue', 'N/A')}",
                f"  Expanding ARI:     {r.get('expanding_mean_ari', 'N/A')}",
                f"  Rolling ARI med:   {r.get('rolling_ari_median', 'N/A')}",
                f"  Crisis shares:     {r.get('crisis_shares', {})}",
                f"  Event mean ARI:    {r.get('event_mean_ari', 'N/A')}",
                f"  Calm mean ARI:     {r.get('calm_mean_ari', 'N/A')}",
                "",
            ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _save_pipeline_summary(self, run_dir: Path, results: Dict) -> None:
        rows = []
        for asset, r in results.items():
            rows.append({
                "asset": asset,
                "overall_mean_ari": r.get("overall_mean_ari"),
                "pvalue_perm": r.get("overall_mean_ari_pvalue_perm"),
                "null_ci_low": r["overall_mean_ari_null_ci"][0] if r.get("overall_mean_ari_null_ci") else None,
                "null_ci_high": r["overall_mean_ari_null_ci"][1] if r.get("overall_mean_ari_null_ci") else None,
                "latest_rolling_7d_mean_ari": r.get("latest_rolling_7d_mean_ari"),
                "rolling_ari_median": r.get("rolling_ari_median"),
                "rolling_ari_q25": r.get("rolling_ari_q25"),
                "rolling_ari_q75": r.get("rolling_ari_q75"),
                "block_perm_pvalue": r.get("block_perm_pvalue"),
                "block_perm_null_ci_low": r["block_perm_null_ci"][0] if r.get("block_perm_null_ci") else None,
                "block_perm_null_ci_high": r["block_perm_null_ci"][1] if r.get("block_perm_null_ci") else None,
                "expanding_mean_ari": r.get("expanding_mean_ari"),
                "event_mean_ari": r.get("event_mean_ari"),
                "calm_mean_ari": r.get("calm_mean_ari"),
            })
        if rows:
            pd.DataFrame(rows).to_csv(run_dir / "pipeline_summary.csv", index=False)
