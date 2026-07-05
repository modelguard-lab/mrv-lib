"""
mrv.invariance.rep -- High-level representation invariance API (Paper 1).

Wraps mrv.validator.RepValidator with a functional interface and a typed
result object so callers do not need to understand the validator config layer.

Source: Paper 1 (Zheng, Low & Wang, 2026)
  - ARI: Table 2 (cross-representation ARI, Adjusted Rand Index metric)
  - Matching-free ordering: posthoc_rank_aligned_ordering.py, Supplement app:ordering
  - 1/K null: Supplement app:ordering, text around Table 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RepInvarianceResult:
    """Result of a representation-invariance check (Paper 1).

    Attributes
    ----------
    ari_per_pair : dict
        ``{(spec_a, spec_b): float}`` -- pairwise ARI for each specification
        pair per asset. Outer key is asset name.
    ordering_per_pair : dict
        ``{(spec_a, spec_b): float}`` -- Spearman ordering consistency per pair
        per asset. ``nan`` when ``returns`` is not provided.
    mean_ari : dict
        ``{asset_name: float}`` -- mean off-diagonal ARI per asset.
    min_ari : dict
        ``{asset_name: float}`` -- minimum pairwise ARI per asset.
    null_1_over_K : float
        ``1 / K`` -- the ordering null under random assignment of K states.
    K : int
        Number of states passed by the caller.
    ari_threshold : float
        Library threshold for "acceptable partition recovery" (Steinley 2004).
    spearman_threshold : float
        Library threshold for stable ordinal risk ordering.
    passes_partition : dict
        ``{asset_name: bool}`` -- True iff mean ARI >= ari_threshold.
    passes_ordering : dict
        ``{asset_name: bool}`` -- True iff mean Spearman >= spearman_threshold.
    """

    ari_per_pair: Dict[str, Dict[tuple[str, str], float]] = field(default_factory=dict)
    ordering_per_pair: Dict[str, Dict[tuple[str, str], float]] = field(default_factory=dict)
    mean_ari: Dict[str, float] = field(default_factory=dict)
    min_ari: Dict[str, float] = field(default_factory=dict)
    null_1_over_K: float = 0.0
    K: int = 2
    ari_threshold: float = ARI_THRESHOLD
    spearman_threshold: float = SPEARMAN_THRESHOLD
    passes_partition: Dict[str, bool] = field(default_factory=dict)
    passes_ordering: Dict[str, bool] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a short text summary."""
        lines = ["RepInvarianceResult", f"  K={self.K}  null_1/K={self.null_1_over_K:.3f}"]
        for asset in self.mean_ari:
            status_p = "PASS" if self.passes_partition.get(asset) else "FAIL"
            status_o = "PASS" if self.passes_ordering.get(asset) else "FAIL"
            pair_vals = list(self.ordering_per_pair.get(asset, {}).values())
            finite = [v for v in pair_vals if v is not None and np.isfinite(v)]
            sp_str = f"{float(np.mean(finite)):.3f}" if finite else "n/a"
            lines.append(
                f"  {asset}: mean_ARI={self.mean_ari[asset]:.3f} [{status_p}]"
                f"  mean_Spearman={sp_str} [{status_o}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------


def rep_invariance_validator(
    model_fn: Callable[[np.ndarray], np.ndarray],
    admissible_class: Dict[str, np.ndarray],
    returns: Optional[np.ndarray] = None,
    K: int = 2,
) -> RepInvarianceResult:
    """Run the Paper 1 representation-invariance check.

    Parameters
    ----------
    model_fn : callable
        ``(features: np.ndarray) -> np.ndarray`` of integer regime labels.
        Called once per specification in ``admissible_class``.
    admissible_class : dict
        ``{spec_name: feature_matrix}`` where each feature matrix is a 2-D
        array of shape ``(n_obs, n_features)``.  At least 2 specifications
        are required.
    returns : np.ndarray, optional
        1-D float array of log-returns aligned with the feature rows.
        When provided, ordering consistency (Spearman) is computed.
    K : int, default 2
        Number of regime states.  Used only to compute ``null_1_over_K``.

    Returns
    -------
    RepInvarianceResult
    """
    if len(admissible_class) < 2:
        raise ValueError(
            "rep_invariance_validator: admissible_class must have >= 2 specifications"
        )

    # Fit labels for every specification.
    labels: Dict[str, np.ndarray] = {}
    for spec_name, features in admissible_class.items():
        labels[spec_name] = model_fn(np.asarray(features))

    # Delegate to RepValidator.
    from mrv.validator.rep import RepValidator

    v = RepValidator()
    asset_name = "asset"
    risk_proxy = None
    if returns is not None:
        risk_proxy = {asset_name: np.asarray(returns)}

    raw = v.validate(
        labels={asset_name: labels},
        risk_proxy=risk_proxy,
    )
    asset_result = raw["assets"][asset_name]
    ari_df = asset_result["ari_matrix"]
    spec_names = list(labels.keys())

    # Build per-pair dicts.
    ari_pairs: Dict[tuple[str, str], float] = {}
    ordering_pairs: Dict[tuple[str, str], float] = {}
    # No direct sp_mat in raw result -- recompute ordering per pair below.
    for i, sa in enumerate(spec_names):
        for j, sb in enumerate(spec_names):
            if j <= i:
                continue
            pair = (sa, sb)
            ari_pairs[pair] = float(ari_df.loc[sa, sb])
            ordering_pairs[pair] = float("nan")
            if returns is not None:
                from mrv.validator.metrics import ordering_consistency
                nc = min(len(labels[sa]), len(labels[sb]), len(returns))
                sp_val = ordering_consistency(
                    labels[sa][:nc], labels[sb][:nc], returns[:nc]
                )
                ordering_pairs[pair] = sp_val

    mean_ari_val = asset_result["mean_ari"]
    min_ari_val = asset_result["min_ari"]

    return RepInvarianceResult(
        ari_per_pair={asset_name: ari_pairs},
        ordering_per_pair={asset_name: ordering_pairs},
        mean_ari={asset_name: mean_ari_val},
        min_ari={asset_name: min_ari_val},
        null_1_over_K=1.0 / max(K, 1),
        K=K,
        ari_threshold=ARI_THRESHOLD,
        spearman_threshold=SPEARMAN_THRESHOLD,
        passes_partition={asset_name: mean_ari_val >= ARI_THRESHOLD},
        passes_ordering={
            asset_name: (
                not np.isnan(asset_result.get("mean_spearman", float("nan")))
                and asset_result.get("mean_spearman", float("nan")) >= SPEARMAN_THRESHOLD
            )
        },
    )
