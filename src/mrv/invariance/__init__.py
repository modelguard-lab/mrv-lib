"""
mrv.invariance -- High-level invariance API.

Wraps Paper 1 (representation) and Paper 2 (resolution) invariance checks
with functional interfaces and typed result objects.

Representation invariance (Paper 1)::

    from mrv.invariance import rep_invariance_validator

    result = rep_invariance_validator(
        model_fn=your_clustering_fn,   # (features: np.ndarray) -> np.ndarray of int labels
        admissible_class={             # {spec_name: feature_matrix}
            "rep_a": feat_a,
            "rep_b": feat_b,
            "rep_c": feat_c,
        },
        returns=log_returns,           # 1-D float array aligned with features
        K=3,
    )

    result.summary()
    print(result.ari_per_pair)
    print(result.ordering_per_pair)
    print(result.null_1_over_K)

    Source: Paper 1 (Zheng, Low & Wang, 2026)
      - ARI: Table 3 (cross-representation ARI, Adjusted Rand Index metric)
      - Matching-free ordering: posthoc_rank_aligned_ordering.py, Supplement app:ordering
      - 1/K null: Supplement app:ordering, text around Table 3

Resolution invariance (Paper 2)::

    from mrv.invariance import res_invariance_validator, ResolutionSpec

    result = res_invariance_validator(
        model_fn=your_regime_fn,       # (prices: pd.Series) -> pd.Series of int labels
        resolution_set={               # {asset_name: {freq: price_series}}
            "SPY": {"5m": spy_5m, "15m": spy_15m, "1h": spy_1h, "1d": spy_1d},
            "CL":  {"5m": cl_5m,  "15m": cl_15m,  "1h": cl_1h,  "1d": cl_1d},
        },
        spec=ResolutionSpec(),         # default: 4-freq Paper 2 panel
    )

    result.summary()
    print(result.ari_matrix["SPY"])         # cross-freq ARI DataFrame
    print(result.ami_matrix["SPY"])         # cross-freq AMI DataFrame
    print(result.intraday_overall_ari_gap)  # {asset: intraday_ARI - overall_ARI}
    print(result.perm_pvalue)               # permutation p-values

    Source: Paper 2 (Zheng, Low & Wang, 2026)
      - Cross-frequency ARI matrix: main Table 1 / Table S1
      - AMI: reported as a per-asset column in main Table 1; the full
        cross-frequency AMI matrix here is a library extension, not a paper
        artefact.
      - intraday_overall_ari_gap: intraday-only mean off-diag ARI minus the
        overall 4-freq mean off-diag ARI (an intraday-vs-daily agreement gap on
        the same data). This library field is NOT Paper 2's simulated-baseline
        "within-intraday excess"; it does not reproduce the paper's +0.03 to
        +0.24 figure.
"""

from mrv.exceptions import MrvConfigError, MrvError, MrvValidationError
from mrv.invariance.rep import (
    RepInvarianceResult,
    rep_invariance_validator,
)
from mrv.invariance.res import (
    PAPER2_FREQS,
    PAPER2_INTRADAY_FREQS,
    ResInvarianceResult,
    ResolutionSpec,
    res_invariance_validator,
)

__all__ = [
    # Exceptions
    "MrvError",
    "MrvValidationError",
    "MrvConfigError",
    # Paper 1
    "RepInvarianceResult",
    "rep_invariance_validator",
    # Paper 2
    "ResInvarianceResult",
    "ResolutionSpec",
    "res_invariance_validator",
    "PAPER2_FREQS",
    "PAPER2_INTRADAY_FREQS",
]
