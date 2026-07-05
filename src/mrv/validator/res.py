"""
mrv.validator.res -- Resolution Invariance validator.

Validates whether regime labels agree across time frequencies
(e.g. 5m / 15m / 1h / 1d).  Users supply pre-computed labels at each
frequency -- this module only measures cross-frequency agreement.

Usage::

    from mrv.validator.res import ResValidator

    v = ResValidator()
    result = v.validate(labels={
        "SPY": {
            "5m":  labels_5m,   # pd.Series with DatetimeIndex
            "15m": labels_15m,
            "1h":  labels_1h,
            "1d":  labels_1d,
        }
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from mrv.validator.base import BaseValidator
from mrv.validator.metrics import ARI_THRESHOLD
from mrv.validator.metrics import variation_of_information as _vi_metric

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────
TZ = "America/New_York"
DEFAULT_ROLLING_DAYS = 7


# ── Alignment ────────────────────────────────────────────────────────────────

def align_labels_to_finest(
    labels_by_freq: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    """Forward-fill each frequency's labels onto the finest-resolution index.

    The finest resolution is determined by the series with the most
    observations.  All other series are forward-filled onto that index.
    """
    if not labels_by_freq:
        return {}
    finest_key = max(labels_by_freq, key=lambda k: len(labels_by_freq[k]))
    finest_idx = labels_by_freq[finest_key].index
    return {
        freq: ser.reindex(finest_idx, method="ffill").fillna(0).astype(int)
        for freq, ser in labels_by_freq.items()
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

def _resolve_index_subset(
    aligned: Dict[str, pd.Series],
    index_subset: Optional[pd.DatetimeIndex] = None,
) -> pd.DatetimeIndex:
    """Default to the first series' index if not provided."""
    if index_subset is not None:
        return index_subset
    first = next(iter(aligned.values()), None)
    return first.index if first is not None else pd.DatetimeIndex([])


def _aligned_pair_values(
    aligned: Dict[str, pd.Series],
    fa: str,
    fb: str,
    index_subset: pd.DatetimeIndex,
    min_obs: int = 10,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Reindex two aligned series and return common values, or None if too few."""
    a = aligned[fa].reindex(index_subset)
    b = aligned[fb].reindex(index_subset)
    common = a.index.intersection(b.index).dropna()
    if len(common) < min_obs:
        return None
    return a.loc[common].astype(int).values, b.loc[common].astype(int).values


def compute_ari_matrix(
    aligned: Dict[str, pd.Series],
    index_subset: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Compute cross-frequency ARI matrix."""
    index_subset = _resolve_index_subset(aligned, index_subset)
    freqs = list(aligned.keys())
    n = len(freqs)
    mat = np.eye(n, dtype=float)
    for i, fa in enumerate(freqs):
        for j, fb in enumerate(freqs):
            if j <= i:
                continue
            pair = _aligned_pair_values(aligned, fa, fb, index_subset)
            if pair is None:
                mat[i, j] = mat[j, i] = np.nan
            else:
                score = adjusted_rand_score(pair[0], pair[1])
                mat[i, j] = mat[j, i] = float(np.clip(round(score, 6), -1.0, 1.0))
    return pd.DataFrame(mat, index=freqs, columns=freqs)


def compute_all_metrics(
    aligned: Dict[str, pd.Series],
    index_subset: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute ARI, AMI, and VI matrices in a single pass."""
    index_subset = _resolve_index_subset(aligned, index_subset)
    freqs = list(aligned.keys())
    n = len(freqs)
    ari_mat = np.eye(n, dtype=float)
    ami_mat = np.eye(n, dtype=float)
    vi_mat = np.zeros((n, n), dtype=float)
    for i, fa in enumerate(freqs):
        for j, fb in enumerate(freqs):
            if j <= i:
                continue
            pair = _aligned_pair_values(aligned, fa, fb, index_subset)
            if pair is None:
                ari_mat[i, j] = ari_mat[j, i] = np.nan
                ami_mat[i, j] = ami_mat[j, i] = np.nan
                vi_mat[i, j] = vi_mat[j, i] = np.nan
            else:
                av, bv = pair
                ari_mat[i, j] = ari_mat[j, i] = float(np.clip(
                    round(adjusted_rand_score(av, bv), 6), -1.0, 1.0))
                ami_mat[i, j] = ami_mat[j, i] = float(np.clip(
                    round(adjusted_mutual_info_score(av, bv), 6), -1.0, 1.0))
                vi_mat[i, j] = vi_mat[j, i] = float(round(
                    _vi_metric(av, bv), 6))
    return {
        "ari": pd.DataFrame(ari_mat, index=freqs, columns=freqs),
        "ami": pd.DataFrame(ami_mat, index=freqs, columns=freqs),
        "vi": pd.DataFrame(vi_mat, index=freqs, columns=freqs),
    }


def mean_offdiag(mat: pd.DataFrame) -> Optional[float]:
    """Mean of off-diagonal entries."""
    if mat is None or mat.empty:
        return None
    vals = mat.values.astype(float)
    if vals.shape[0] != vals.shape[1]:
        return None
    mask = ~np.eye(vals.shape[0], dtype=bool)
    offdiag = vals[mask]
    return float(np.nanmean(offdiag)) if offdiag.size else None


# ── Permutation Tests ────────────────────────────────────────────────────────

def permute_pvalue_mean_offdiag_ari(
    aligned: Dict[str, pd.Series],
    index_subset: Optional[pd.DatetimeIndex] = None,
    n_perm: int = 500,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Permutation test for mean off-diagonal ARI."""
    freqs = list(aligned.keys())
    first = next(iter(aligned.values()), None)
    if first is None or first.empty:
        return None, None
    if index_subset is None:
        index_subset = first.index
    if len(index_subset) < 50:
        return None, None

    obs_df = compute_ari_matrix(aligned, index_subset=index_subset)
    obs = mean_offdiag(obs_df)
    if obs is None or not np.isfinite(obs):
        return None, None

    rng = np.random.default_rng(seed)
    y = {freq: aligned[freq].reindex(index_subset).astype(int).to_numpy() for freq in freqs}

    null_stats = np.empty(int(n_perm), dtype=float)
    for k in range(int(n_perm)):
        y_perm = {freq: rng.permutation(arr) for freq, arr in y.items()}
        vals = []
        for i, fa in enumerate(freqs):
            for j, fb in enumerate(freqs):
                if j <= i:
                    continue
                vals.append(adjusted_rand_score(y_perm[fa], y_perm[fb]))
        null_stats[k] = float(np.mean(vals)) if vals else np.nan

    ge = np.mean(null_stats >= obs)
    p = float((ge * n_perm + 1.0) / (n_perm + 1.0))
    ci = (float(np.nanpercentile(null_stats, 2.5)), float(np.nanpercentile(null_stats, 97.5)))
    return p, ci


# ── Window Subsetting ────────────────────────────────────────────────────────

def subset_index_by_dates(
    index: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
    tz: str = TZ,
) -> pd.DatetimeIndex:
    """Select timestamps whose calendar date is in [start, end].

    Uses ``ambiguous="NaT"`` to avoid AmbiguousTimeError on DST transitions
    (autumn clock-change produces duplicate hour that ``ambiguous="infer"``
    cannot always resolve for sub-minute data).  NaT entries are dropped
    before masking.
    """
    if index.tz is None:
        idx = index.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        index = index[~idx.isna()]
        idx = idx.dropna()
    else:
        idx = index.tz_convert(tz)
    d = idx.normalize()
    start = pd.Timestamp(start_date).tz_localize(tz)
    end = pd.Timestamp(end_date).tz_localize(tz)
    mask = (d >= start.normalize()) & (d <= end.normalize())
    return index[mask]


# ── Daily & Rolling Summaries ────────────────────────────────────────────────

def compute_daily_outputs(
    aligned: Dict[str, pd.Series],
    rolling_days: int = DEFAULT_ROLLING_DAYS,
    tz: str = TZ,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build daily summaries and rolling ARI tables."""
    first = next(iter(aligned.values()), None)
    if first is None or first.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    freqs = list(aligned.keys())
    index_fine = first.index
    day_labels = index_fine.tz_convert(tz).normalize() if index_fine.tz else index_fine.normalize()
    days = pd.DatetimeIndex(day_labels.unique()).sort_values()

    daily_rows: List[Dict[str, Any]] = []
    daily_pair_rows: List[Dict[str, Any]] = []
    rolling_rows: List[Dict[str, Any]] = []
    rolling_pair_rows: List[Dict[str, Any]] = []

    for day in days:
        mask = day_labels == day
        day_index = index_fine[mask]
        ari_df = compute_ari_matrix(aligned, day_index)
        row: Dict[str, Any] = {
            "date": day.strftime("%Y-%m-%d"),
            "bars": int(len(day_index)),
            "mean_offdiag_ari": mean_offdiag(ari_df),
        }
        for freq in freqs:
            sub = aligned[freq].reindex(day_index)
            row[f"crisis_share_{freq}"] = (
                float(100.0 * (sub == 1).mean()) if len(day_index) else np.nan
            )
        daily_rows.append(row)
        for i, fa in enumerate(freqs):
            for j, fb in enumerate(freqs):
                if j <= i:
                    continue
                daily_pair_rows.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "freq_a": fa, "freq_b": fb,
                    "ari": ari_df.loc[fa, fb],
                })

    for end_idx in range(rolling_days - 1, len(days)):
        window_days = days[end_idx - rolling_days + 1: end_idx + 1]
        mask = pd.Series(day_labels, index=index_fine).isin(window_days).values
        window_index = index_fine[mask]
        ari_df = compute_ari_matrix(aligned, window_index)
        rolling_rows.append({
            "window_start": window_days[0].strftime("%Y-%m-%d"),
            "window_end": window_days[-1].strftime("%Y-%m-%d"),
            "days_in_window": len(window_days),
            "bars": int(len(window_index)),
            "mean_offdiag_ari": mean_offdiag(ari_df),
        })
        for i, fa in enumerate(freqs):
            for j, fb in enumerate(freqs):
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
    """Gantt-style timeline: X=time, coloured by regime label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    freqs = list(aligned.keys())
    first = next(iter(aligned.values()))
    t_common = first.index

    fig, axes = plt.subplots(len(freqs), 1, figsize=(12, len(freqs)),
                              sharex=True,
                              gridspec_kw={"height_ratios": [1] * len(freqs)})
    if len(freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, freqs):
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


# ── Core Analysis ────────────────────────────────────────────────────────────

def analyze_labels(
    asset_name: str,
    labels_by_freq: Dict[str, pd.Series],
    rolling_days: int = DEFAULT_ROLLING_DAYS,
    event_window: Optional[Tuple[str, str]] = None,
    calm_window: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Run cross-frequency analysis on pre-computed labels.

    Parameters
    ----------
    asset_name : str
        Name of the asset (for logging / output).
    labels_by_freq : dict
        ``{freq: pd.Series}`` -- regime labels at each frequency.
        Each Series must have a DatetimeIndex.
    rolling_days : int
        Window for rolling ARI summary.
    event_window, calm_window : tuple of (start, end) date strings, optional
        If provided, compute ARI separately for event/calm periods.
    """
    freqs = list(labels_by_freq.keys())
    aligned = align_labels_to_finest(labels_by_freq)

    for freq in freqs:
        pct = 100.0 * (aligned[freq] == 1).mean()
        logger.info("%s %s: crisis share %.1f%%", asset_name, freq, pct)

    # Metrics
    all_metrics = compute_all_metrics(aligned)
    ari_df = all_metrics["ari"]
    perm_p, perm_ci = permute_pvalue_mean_offdiag_ari(aligned, n_perm=500, seed=42)

    # Event/calm windows
    first = next(iter(aligned.values()))
    fine_index = first.index
    event_ari_df = pd.DataFrame()
    calm_ari_df = pd.DataFrame()
    if event_window:
        event_index = subset_index_by_dates(fine_index, event_window[0], event_window[1])
        if len(event_index):
            event_ari_df = compute_ari_matrix(aligned, event_index)
    if calm_window:
        calm_index = subset_index_by_dates(fine_index, calm_window[0], calm_window[1])
        if len(calm_index):
            calm_ari_df = compute_ari_matrix(aligned, calm_index)

    # Daily/rolling
    daily_df, daily_pair_df, rolling_df, rolling_pair_df = compute_daily_outputs(
        aligned, rolling_days=rolling_days)

    crisis_shares = {freq: float(100.0 * (aligned[freq] == 1).mean()) for freq in freqs}

    # Rolling ARI distribution
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
        "frequencies": freqs,
        "rolling_days": int(rolling_days),
        "ari_matrix": ari_df,
        "ami_matrix": all_metrics["ami"],
        "vi_matrix": all_metrics["vi"],
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
        "overall_mean_ari": mean_offdiag(ari_df),
        "overall_mean_ari_pvalue_perm": perm_p,
        "overall_mean_ari_null_ci": perm_ci,
        "event_mean_ari": mean_offdiag(event_ari_df) if not event_ari_df.empty else None,
        "calm_mean_ari": mean_offdiag(calm_ari_df) if not calm_ari_df.empty else None,
        "latest_rolling_mean_ari": (
            None if rolling_df.empty
            else float(rolling_df["mean_offdiag_ari"].iloc[-1])
        ),
        "rolling_ari_median": rolling_ari_median,
        "rolling_ari_q25": rolling_ari_q25,
        "rolling_ari_q75": rolling_ari_q75,
    }


# ── Validator Class ──────────────────────────────────────────────────────────

class ResValidator(BaseValidator):
    """Resolution Invariance validator.

    Measures whether regime labels agree across time frequencies.
    Users provide their own labels -- no model fitting is performed.
    """

    name = "res"

    def validate(  # type: ignore[override]
        self,
        labels: Dict[str, Dict[str, pd.Series]],
        event_window: Optional[Tuple[str, str]] = None,
        calm_window: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run resolution invariance test on user-provided labels.

        Parameters
        ----------
        labels : dict
            ``{asset_name: {freq: pd.Series}}``.
            Each Series must have a DatetimeIndex and contain integer labels.
            At least 2 frequencies per asset are required.
        event_window : tuple, optional
            ``(start_date, end_date)`` for event-period analysis.
        calm_window : tuple, optional
            ``(start_date, end_date)`` for calm-period analysis.
        """
        if not labels:
            raise ValueError("labels dict is empty -- provide at least one asset")

        for asset, freqs in labels.items():
            if len(freqs) < 2:
                raise ValueError(
                    f"Asset '{asset}' has {len(freqs)} frequency(ies), need >= 2"
                )

        # Resolve episode from config if not passed directly
        res_cfg = self.test_cfg
        if event_window is None and res_cfg.get("event_window"):
            raw_ew = res_cfg["event_window"]
            if not (isinstance(raw_ew, (list, tuple)) and len(raw_ew) == 2):
                raise ValueError(
                    f"event_window must be a 2-element [start, end] list; got {raw_ew!r}"
                )
            event_window = (str(raw_ew[0]), str(raw_ew[1]))
        if calm_window is None and res_cfg.get("calm_window"):
            raw_cw = res_cfg["calm_window"]
            if not (isinstance(raw_cw, (list, tuple)) and len(raw_cw) == 2):
                raise ValueError(
                    f"calm_window must be a 2-element [start, end] list; got {raw_cw!r}"
                )
            calm_window = (str(raw_cw[0]), str(raw_cw[1]))

        run_dir = self._make_run_dir()

        logger.info("=== Resolution Invariance ===")
        logger.info("Assets: %s", list(labels.keys()))

        all_results: Dict[str, Dict] = {}

        for asset_name, freq_labels in labels.items():
            logger.info("--- %s ---", asset_name)

            analysis = analyze_labels(
                asset_name, freq_labels,
                event_window=event_window,
                calm_window=calm_window,
            )

            # Attribution (if enabled)
            if res_cfg.get("attribution", False) and "ari_matrix" in analysis:
                from mrv.validator.attribution import (
                    freq_pair_attribution,
                    generate_attribution_summary,
                    temporal_attribution,
                )
                pair_attr = freq_pair_attribution(analysis["ari_matrix"])
                attr_result: Dict[str, Any] = {"freq_pairs": pair_attr}
                if pair_attr and "regimes_aligned" in analysis:
                    worst = pair_attr[0]
                    aligned = analysis["regimes_aligned"]
                    if worst["freq_a"] in aligned and worst["freq_b"] in aligned:
                        temp = temporal_attribution(
                            aligned[worst["freq_a"]], aligned[worst["freq_b"]])
                        if not temp.empty:
                            attr_result["temporal"] = temp
                            temp.to_csv(
                                run_dir / f"{asset_name}_attribution_timeline.csv",
                                index=False,
                            )
                attr_result["summary"] = generate_attribution_summary(attr_result, "res")
                analysis["attribution"] = attr_result

            all_results[asset_name] = analysis
            self._save_asset_outputs(asset_name, analysis, run_dir)

        # Save JSON
        json_path = run_dir / "result.json"
        json_data = self._build_json(all_results, res_cfg)
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.json_path = json_path
        logger.info("JSON -> %s", json_path)

        # Save text summary
        self._write_text_report(run_dir / "summary.txt", all_results, res_cfg)

        # Pipeline summary CSV
        self._save_pipeline_summary(run_dir, all_results)

        logger.info("=== Output: %s ===", run_dir)
        self.results = all_results
        return {"run_dir": str(run_dir), "json_path": str(json_path), "assets": all_results}

    # ── Internal helpers ─────────────────────────────────────────────────

    def _save_asset_outputs(self, asset_name: str, analysis: Dict, run_dir: Path) -> None:
        """Save CSVs and plots for one asset."""
        analysis["ari_matrix"].to_csv(run_dir / f"{asset_name}_cross_freq_ari.csv")
        analysis["ami_matrix"].to_csv(run_dir / f"{asset_name}_cross_freq_ami.csv")
        analysis["vi_matrix"].to_csv(run_dir / f"{asset_name}_cross_freq_vi.csv")

        event_mat = analysis.get("event_ari_matrix")
        if isinstance(event_mat, pd.DataFrame) and not event_mat.empty:
            event_mat.to_csv(run_dir / f"{asset_name}_event_cross_freq_ari.csv")
        calm_mat = analysis.get("calm_ari_matrix")
        if isinstance(calm_mat, pd.DataFrame) and not calm_mat.empty:
            calm_mat.to_csv(run_dir / f"{asset_name}_calm_cross_freq_ari.csv")

        # Plots
        if "regimes_aligned" in analysis:
            try:
                _plot_timeline(
                    analysis["regimes_aligned"], asset_name,
                    run_dir / f"{asset_name}_timeline.png",
                )
                from mrv.validator.plots import plot_ari_heatmap
                plot_ari_heatmap(
                    analysis["ari_matrix"], asset_name,
                    run_dir / f"{asset_name}_ari_heatmap.png",
                    title_prefix="Cross-Frequency",
                )
            except ImportError:
                logger.debug("matplotlib not available, skipping plots")

        # Daily/rolling CSVs
        daily_df = analysis.get("daily_df")
        if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
            daily_df.to_csv(run_dir / f"{asset_name}_daily_summary.csv", index=False)
        daily_pair_df = analysis.get("daily_pair_df")
        if isinstance(daily_pair_df, pd.DataFrame) and not daily_pair_df.empty:
            daily_pair_df.to_csv(
                run_dir / f"{asset_name}_daily_pairwise_ari.csv", index=False
            )
        rolling_df = analysis.get("rolling_df")
        if isinstance(rolling_df, pd.DataFrame) and not rolling_df.empty:
            rolling_df.to_csv(run_dir / f"{asset_name}_rolling_ari.csv", index=False)
            if isinstance(analysis.get("rolling_pair_df"), pd.DataFrame):
                analysis["rolling_pair_df"].to_csv(
                    run_dir / f"{asset_name}_rolling_pairwise_ari.csv", index=False)

        logger.info("Saved %s outputs -> %s", asset_name, run_dir)

    def _build_json(self, results: Dict, res_cfg: Dict) -> Dict:
        all_ari = [r["overall_mean_ari"] for r in results.values()
                   if r.get("overall_mean_ari") is not None and np.isfinite(r["overall_mean_ari"])]
        overall_ari = float(np.mean(all_ari)) if all_ari else None

        assets_json = {}
        for name, r in results.items():
            ari_df = r["ari_matrix"]
            assets_json[name] = {
                "frequencies": r.get("frequencies", []),
                "overall_mean_ari": (
                    round(r["overall_mean_ari"], 6)
                    if r.get("overall_mean_ari") is not None else None
                ),
                "partition_pass": (
                    r.get("overall_mean_ari") is not None
                    and r["overall_mean_ari"] >= ARI_THRESHOLD
                ),
                "pvalue_perm": r.get("overall_mean_ari_pvalue_perm"),
                "null_ci": r.get("overall_mean_ari_null_ci"),
                "event_mean_ari": r.get("event_mean_ari"),
                "calm_mean_ari": r.get("calm_mean_ari"),
                "crisis_shares": r.get("crisis_shares", {}),
                "rolling_ari_median": r.get("rolling_ari_median"),
                "rolling_ari_q25": r.get("rolling_ari_q25"),
                "rolling_ari_q75": r.get("rolling_ari_q75"),
                "ari_matrix": {
                    "labels": list(ari_df.columns),
                    "values": [[round(v, 6) for v in row] for row in ari_df.values.tolist()],
                },
            }

        return {
            "test": "resolution_invariance",
            "generated": datetime.now().isoformat(),
            "date_range": {"start": res_cfg.get("start"), "end": res_cfg.get("end")},
            "ari_threshold": ARI_THRESHOLD,
            "overall_mean_ari": round(overall_ari, 6) if overall_ari is not None else None,
            "partition_pass": overall_ari is not None and overall_ari >= ARI_THRESHOLD,
            "assets": assets_json,
        }

    def _write_text_report(self, path: Path, results: Dict, res_cfg: Dict) -> None:
        lines = [
            "=" * 60, "MRV Resolution Invariance Report", "=" * 60, "",
            f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Period:       {res_cfg.get('start', '?')} -> {res_cfg.get('end', '?')}", "",
        ]
        for asset, r in results.items():
            mean_ari = r.get("overall_mean_ari")
            status = "PASS" if mean_ari is not None and mean_ari >= ARI_THRESHOLD else "FAIL"
            lines += [
                f"--- {asset} ---",
                f"  Frequencies: {', '.join(r.get('frequencies', []))}",
                (f"  Mean off-diag ARI: {mean_ari:.3f} [{status}]"
                 if mean_ari is not None else "  Mean off-diag ARI: N/A"),
                f"  Perm p-value:      {r.get('overall_mean_ari_pvalue_perm', 'N/A')}",
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
                "null_ci_low": (
                    r["overall_mean_ari_null_ci"][0]
                    if r.get("overall_mean_ari_null_ci") else None
                ),
                "null_ci_high": (
                    r["overall_mean_ari_null_ci"][1]
                    if r.get("overall_mean_ari_null_ci") else None
                ),
                "latest_rolling_mean_ari": r.get("latest_rolling_mean_ari"),
                "rolling_ari_median": r.get("rolling_ari_median"),
            })
        if rows:
            pd.DataFrame(rows).to_csv(run_dir / "pipeline_summary.csv", index=False)
