"""
mrv.validator.attribution — Disagreement attribution & root cause analysis.

Three attribution methods:
1. Leave-one-out factor attribution (rep validator)
2. Frequency-pair decomposition (res validator)
3. Temporal hotspot detection (both validators)
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)

MIN_SAMPLES = 10


# ── Leave-one-out factor attribution (rep) ───────────────────────────────────

def loo_factor_attribution(
    labels_dict: Dict[str, np.ndarray],
    baseline_mean_ari: float,
) -> Dict[str, Any]:
    """Leave-one-out factor attribution for representation invariance.

    For each factor set *i*, remove it and recompute mean pairwise ARI
    from the remaining sets.

    Returns::

        {
            "baseline_mean_ari": 0.45,
            "scores": {"set_label": delta_ari, ...},
            "worst_contributor": "set_label",
            "summary": "..."
        }

    A positive delta means removing set *i* **improves** ARI → set *i*
    is a disagreement driver.
    """
    keys = list(labels_dict.keys())
    n = len(keys)
    if n < 3:
        return {
            "baseline_mean_ari": baseline_mean_ari,
            "scores": {},
            "worst_contributor": None,
            "summary": "Need >= 3 factor sets for LOO attribution.",
        }

    scores: Dict[str, float] = {}
    for drop_key in keys:
        remaining = {k: v for k, v in labels_dict.items() if k != drop_key}
        rem_keys = list(remaining.keys())
        ari_vals = []
        for ka, kb in combinations(rem_keys, 2):
            a, b = remaining[ka], remaining[kb]
            nc = min(len(a), len(b))
            if nc >= MIN_SAMPLES:
                ari_vals.append(adjusted_rand_score(a[:nc], b[:nc]))
        loo_ari = float(np.mean(ari_vals)) if ari_vals else float("nan")
        scores[drop_key] = round(loo_ari - baseline_mean_ari, 6)

    worst = max(scores, key=scores.get) if scores else None
    summary = ""
    if worst and scores[worst] > 0.01:
        summary = (
            f"Removing '{worst}' improves mean ARI by {scores[worst]:+.3f}, "
            f"indicating it is the primary disagreement driver."
        )
    elif worst:
        summary = "No single factor set dominates the disagreement."

    return {
        "baseline_mean_ari": baseline_mean_ari,
        "scores": scores,
        "worst_contributor": worst,
        "summary": summary,
    }


# ── Frequency-pair decomposition (res) ───────────────────────────────────────

def freq_pair_attribution(
    ari_matrix: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Rank frequency pairs by pairwise ARI (ascending = worst first).

    Returns a list of dicts with keys: freq_a, freq_b, ari, rank.
    """
    freqs = list(ari_matrix.index)
    pairs = []
    for i, fa in enumerate(freqs):
        for j, fb in enumerate(freqs):
            if j <= i:
                continue
            pairs.append({
                "freq_a": fa, "freq_b": fb,
                "ari": float(ari_matrix.loc[fa, fb]),
            })
    pairs.sort(key=lambda x: x["ari"])
    for rank, p in enumerate(pairs, 1):
        p["rank"] = rank
    return pairs


# ── Temporal hotspot detection ───────────────────────────────────────────────

def temporal_attribution(
    labels_a: pd.Series,
    labels_b: pd.Series,
    window: str = "1D",
    ari_threshold: float = 0.3,
) -> pd.DataFrame:
    """Per-window ARI between two label sequences.

    Groups timestamps by *window* (default: 1 calendar day), computes ARI
    per group, and flags hotspots where ARI < ``ari_threshold``.

    Returns DataFrame: window_start, n_obs, ari, is_hotspot.
    """
    # Align
    common = labels_a.index.intersection(labels_b.index)
    if len(common) < MIN_SAMPLES:
        return pd.DataFrame(columns=["window_start", "n_obs", "ari", "is_hotspot"])

    a = labels_a.reindex(common).astype(int)
    b = labels_b.reindex(common).astype(int)

    # Group by window
    if window == "1D":
        tz = common.tz
        if tz is not None:
            groups = common.tz_convert("America/New_York").normalize()
        else:
            groups = common.normalize()
    else:
        groups = pd.Grouper(freq=window)

    rows = []
    if window == "1D":
        unique_days = pd.DatetimeIndex(groups.unique()).sort_values()
        for day in unique_days:
            mask = groups == day
            a_sub = a[mask].values
            b_sub = b[mask].values
            if len(a_sub) < MIN_SAMPLES:
                continue
            ari_val = float(adjusted_rand_score(a_sub, b_sub))
            rows.append({
                "window_start": day.strftime("%Y-%m-%d"),
                "n_obs": len(a_sub),
                "ari": round(ari_val, 6),
                "is_hotspot": ari_val < ari_threshold,
            })
    else:
        combined = pd.DataFrame({"a": a, "b": b})
        for name, grp in combined.resample(window):
            if len(grp) < MIN_SAMPLES:
                continue
            ari_val = float(adjusted_rand_score(grp["a"].values, grp["b"].values))
            rows.append({
                "window_start": str(name),
                "n_obs": len(grp),
                "ari": round(ari_val, 6),
                "is_hotspot": ari_val < ari_threshold,
            })

    return pd.DataFrame(rows)


# ── Summary generation ───────────────────────────────────────────────────────

def generate_attribution_summary(
    attr_results: Dict[str, Any],
    validator_type: str,
) -> str:
    """Generate a plain-language attribution summary."""
    lines = []

    if validator_type == "rep":
        scores = attr_results.get("scores", {})
        worst = attr_results.get("worst_contributor")
        if worst and scores.get(worst, 0) > 0.01:
            lines.append(
                f"Primary disagreement driver: factor set '{worst}' "
                f"(removing it improves mean ARI by {scores[worst]:+.3f})."
            )
        else:
            lines.append("No single factor set dominates the disagreement.")

    elif validator_type == "res":
        freq_pairs = attr_results.get("freq_pairs", [])
        temporal = attr_results.get("temporal")
        if freq_pairs:
            worst = freq_pairs[0]
            lines.append(
                f"Weakest frequency pair: {worst['freq_a']} vs {worst['freq_b']} "
                f"(ARI = {worst['ari']:.3f})."
            )
        if isinstance(temporal, pd.DataFrame) and not temporal.empty:
            hotspots = temporal[temporal["is_hotspot"]]
            if not hotspots.empty:
                dates = hotspots["window_start"].tolist()
                lines.append(
                    f"Temporal hotspots ({len(dates)} days): "
                    f"{', '.join(dates[:5])}{'...' if len(dates) > 5 else ''}."
                )

    return " ".join(lines) if lines else "No attribution anomalies detected."
