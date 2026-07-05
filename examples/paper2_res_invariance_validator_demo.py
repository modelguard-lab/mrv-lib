"""
paper2_res_invariance_validator_demo.py
=======================================
Canonical SPY / CL / USDJPY fixture demo for mrv.invariance.res_invariance_validator.

Demonstrates the Paper 2 resolution-invariance check using synthetic regime
labels at four frequencies (5m / 15m / 1h / 1d).  No live data is needed.

Run::

    python examples/paper2_res_invariance_validator_demo.py

What this shows
---------------
1. Build a resolution_set dict: {asset: {freq: label_series}}.
2. Call res_invariance_validator with a passthrough model (labels supplied
   directly) and the default Paper 2 four-frequency spec.
3. Print:
   - Cross-frequency ARI matrix per asset (Paper 2 Table 2 analogue).
   - Cross-frequency AMI matrix per asset (Supplement S.2 robustness).
   - Overall mean off-diagonal ARI and permutation p-value.
   - Within-intraday excess = intraday_mean_ARI - overall_mean_ARI.
     Positive = intraday freqs agree more with each other than with daily.

Fixture design
--------------
Three assets: SPY (US equity ETF), CL (WTI crude front-month future),
USDJPY (FX spot).

Labels are drawn as i.i.d. Bernoulli(0.3): each frequency's labels are
independent Bernoulli(0.3) draws to mimic a "crisis state = 1" regime model.
This produces low cross-frequency ARI and a small within-intraday excess,
consistent with the null of no structured regime persistence.

To test the "perfect invariance" case, all four frequencies share the same
label series; ARI matrix should be all-ones off-diagonal with zero excess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrv.invariance import (
    ResolutionSpec,
    res_invariance_validator,
    PAPER2_FREQS,
    PAPER2_INTRADAY_FREQS,
)


# ---------------------------------------------------------------------------
# Synthetic label factory
# ---------------------------------------------------------------------------


def _make_labels(n: int, p_crisis: float = 0.3, seed: int = 0) -> np.ndarray:
    """Bernoulli(p_crisis) integer labels: 1 = crisis, 0 = calm."""
    rng = np.random.RandomState(seed)
    return rng.choice([0, 1], size=n, p=[1 - p_crisis, p_crisis])


def _make_resolution_set(
    assets: list[str],
    n_bars: int = 400,
    freqs: tuple = PAPER2_FREQS,
    seed: int = 42,
    share_labels: bool = False,
) -> dict[str, dict[str, pd.Series]]:
    """Build a synthetic resolution_set.

    Parameters
    ----------
    assets : list of str
        Asset names.
    n_bars : int
        Number of bars per (asset, freq) series.
    freqs : tuple of str
        Frequency labels to include.
    seed : int
        Base random seed; each (asset, freq) gets a distinct sub-seed.
    share_labels : bool
        When True, all freqs share the same label sequence per asset,
        producing an all-ones ARI matrix (perfect invariance scenario).
    """
    resolution_set: dict = {}
    for i, asset in enumerate(assets):
        resolution_set[asset] = {}
        base_labels = _make_labels(n_bars, seed=seed + i * 100)
        for j, freq in enumerate(freqs):
            idx = pd.date_range(
                "2026-01-05 09:30",
                periods=n_bars,
                freq="5min",
                tz="America/New_York",
            )
            if share_labels:
                lbls = base_labels.copy()
            else:
                lbls = _make_labels(n_bars, seed=seed + i * 100 + j)
            resolution_set[asset][freq] = pd.Series(lbls, index=idx, dtype=int)
    return resolution_set


# Passthrough: labels are already in the resolution_set.
def passthrough(s: pd.Series) -> pd.Series:
    return s


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _print_matrix(mat: pd.DataFrame, title: str) -> None:
    print(f"\n  {title}")
    print("  " + mat.round(4).to_string().replace("\n", "\n  "))


def run_demo(share_labels: bool = False) -> None:
    label = "PERFECT INVARIANCE (shared labels)" if share_labels else "NULL CASE (independent labels)"
    print("=" * 64)
    print(f"Demo: {label}")
    print("=" * 64)

    assets = ["SPY", "CL", "USDJPY"]
    spec = ResolutionSpec(
        freqs=PAPER2_FREQS,
        intraday_freqs=PAPER2_INTRADAY_FREQS,
    )

    resolution_set = _make_resolution_set(
        assets,
        n_bars=400,
        freqs=PAPER2_FREQS,
        seed=42,
        share_labels=share_labels,
    )

    result = res_invariance_validator(
        model_fn=passthrough,
        resolution_set=resolution_set,
        spec=spec,
        run_permutation=True,
        n_perm=200,
        seed=42,
    )

    print(result.summary())

    for asset in assets:
        print(f"\n--- {asset} ---")
        _print_matrix(result.ari_matrix[asset], "Cross-frequency ARI matrix")
        _print_matrix(result.ami_matrix[asset], "Cross-frequency AMI matrix")

        overall = result.overall_mean_ari[asset]
        intraday = result.intraday_mean_ari[asset]
        excess = result.within_intraday_excess[asset]
        pval = result.perm_pvalue[asset]
        ci = result.perm_null_ci[asset]

        print(f"\n  Metrics for {asset}:")
        print(f"    overall_mean_ARI  = {overall:.4f}" if overall is not None else "    overall_mean_ARI  = n/a")
        print(f"    intraday_mean_ARI = {intraday:.4f}" if intraday is not None else "    intraday_mean_ARI = n/a")
        print(f"    within_intraday_excess = {excess:+.4f}" if excess is not None else "    within_intraday_excess = n/a")
        print(f"    perm_p-value      = {pval:.4f}" if pval is not None else "    perm_p-value      = n/a")
        if ci is not None:
            print(f"    perm_null_ci (95%) = [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"    passes_partition  = {result.passes_partition[asset]}")


def main() -> None:
    # Case 1: independent random labels (null scenario, low ARI expected)
    run_demo(share_labels=False)

    print()

    # Case 2: all freqs share same labels (perfect invariance, ARI ~ 1)
    run_demo(share_labels=True)


if __name__ == "__main__":
    main()
