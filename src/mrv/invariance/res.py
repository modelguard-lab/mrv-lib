"""
mrv.invariance.res -- High-level resolution invariance API (Paper 2).

Provides a typed functional interface that delegates to the shared
ResValidator cross-frequency orchestration (mrv.validator.res.analyze_labels,
the per-asset core that mrv.validator.ResValidator.validate runs) rather than
re-implementing the alignment / metric / permutation pipeline.  Takes a model
callable and a resolution-set spec; returns the cross-frequency ARI matrix,
the AMI matrix, and an intraday-vs-daily ARI agreement gap
(``intraday_overall_ari_gap``; see below).

Source: Paper 2 (Zheng, Low & Wang, 2026)
  - Cross-frequency ARI matrix: main Table 1 / Table S1
  - AMI: Paper 2 reports AMI as a per-asset column in main Table 1; the full
    cross-frequency AMI matrix computed here is a library extension, not a
    paper artefact.
  - intraday_overall_ari_gap = intraday_mean_ari - overall_mean_ari: the
    intraday-only mean off-diagonal ARI minus the overall (4-frequency) mean
    off-diagonal ARI, i.e. an intraday-vs-daily agreement gap measured on the
    same data. The two operand fields (``overall_mean_ari``, 4-freq mean
    off-diag ARI over 5m/15m/1h/1d; ``intraday_mean_ari``, 3-freq intraday-only
    over 5m/15m/1h) correspond to the SimReplicationResult component fields of
    the same name. A positive value means intraday frequencies agree more with
    each other than with the daily scale. NOTE: this library field is NOT
    Paper 2's headline "within-intraday excess", which the paper defines as the
    empirical intraday ARI minus a simulated MS-Gaussian baseline of ~0.195
    (giving +0.03 to +0.24). These are different quantities; the
    ``intraday_overall_ari_gap`` field does not reproduce the paper's +0.03 to
    +0.24 figure.

Canonical resolution-set spec
------------------------------
Paper 2 uses FREQS = ("5m", "15m", "1h", "1d") as the four-frequency panel
for each of SPY (equities), CL (WTI futures), and USDJPY (FX).
The ``ResolutionSpec`` helper class encodes this convention so callers need
not repeat the frequency labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mrv.exceptions import MrvValidationError
from mrv.validator.metrics import ARI_THRESHOLD

# ---------------------------------------------------------------------------
# Canonical Paper 2 frequency sets
# ---------------------------------------------------------------------------

#: Default four-frequency panel (Paper 2 main Table 1 panel).
# Source: Paper 2 src/core/config.py FREQS
PAPER2_FREQS: Tuple[str, ...] = ("5m", "15m", "1h", "1d")

#: Intraday-only subset (Paper 2 intraday_mean_ari).
# Source: Paper 2 src/core/sim_dgp.py line 435
PAPER2_INTRADAY_FREQS: Tuple[str, ...] = ("5m", "15m", "1h")


# ---------------------------------------------------------------------------
# ResolutionSpec
# ---------------------------------------------------------------------------


@dataclass
class ResolutionSpec:
    """Describes which resolution levels to test and which are intraday.

    Parameters
    ----------
    freqs : tuple of str
        Ordered frequency labels. Must have >= 2 entries.
        Labels must match the keys in the ``labels`` dict passed to
        :func:`res_invariance_validator`.
    intraday_freqs : tuple of str, optional
        Subset of ``freqs`` to use for the intraday-vs-overall ARI gap metric.
        Defaults to all frequencies that are not ``"1d"``.

    Examples
    --------
    Default Paper 2 four-frequency spec::

        spec = ResolutionSpec()   # ("5m", "15m", "1h", "1d")

    Three-frequency intraday-only spec::

        spec = ResolutionSpec(freqs=("5m", "15m", "1h"), intraday_freqs=("5m", "15m", "1h"))
    """

    freqs: Tuple[str, ...] = PAPER2_FREQS
    intraday_freqs: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        self.freqs = tuple(self.freqs)
        if len(self.freqs) < 2:
            raise MrvValidationError("ResolutionSpec: freqs must have >= 2 entries")
        if self.intraday_freqs is None:
            self.intraday_freqs = tuple(f for f in self.freqs if f != "1d")
        else:
            self.intraday_freqs = tuple(self.intraday_freqs)
        for f in self.intraday_freqs:
            if f not in self.freqs:
                raise MrvValidationError(
                    f"ResolutionSpec: intraday_freq {f!r} not in freqs {self.freqs}"
                )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResInvarianceResult:
    """Result of a resolution-invariance check (Paper 2).

    Attributes
    ----------
    ari_matrix : dict
        ``{asset_name: pd.DataFrame}`` -- symmetric cross-frequency ARI matrix
        per asset. Rows and columns are frequency labels.
    ami_matrix : dict
        ``{asset_name: pd.DataFrame}`` -- symmetric cross-frequency AMI matrix
        per asset.
    overall_mean_ari : dict
        ``{asset_name: float | None}`` -- mean of off-diagonal ARI entries.
        None when the aligned label set is too short to compute.
    intraday_mean_ari : dict
        ``{asset_name: float | None}`` -- mean off-diagonal ARI on the
        intraday-only frequency subset (omits pairs involving "1d").
    intraday_overall_ari_gap : dict
        ``{asset_name: float | None}`` -- ``intraday_mean_ari - overall_mean_ari``:
        the intraday-only mean off-diagonal ARI minus the overall (4-frequency)
        mean off-diagonal ARI, an intraday-vs-daily agreement gap on the same
        data. Positive when intraday frequencies agree more with each other than
        with the daily scale. This library field is NOT Paper 2's
        simulated-baseline "within-intraday excess" and does not reproduce the
        paper's +0.03 to +0.24 figure.
    passes_partition : dict
        ``{asset_name: bool}`` -- True iff overall_mean_ari >= ari_threshold.
        This is a library-provided convenience for the user, not a Paper 2
        verdict: Paper 2 does not define a 0.65 (or any) ARI pass/fail cutoff.
    ari_threshold : float
        The library-default ARI threshold used for ``passes_partition``,
        surfaced here for reference. It is the module constant
        ``mrv.validator.metrics.ARI_THRESHOLD`` (the library convention 0.65,
        Steinley 2004 ARI-substantiality guideline, adopted by Paper 1). The
        functional validator does not take a per-call threshold argument;
        ``passes_partition`` always uses this default.
    freqs : tuple of str
        Frequency labels in the order they appear in the matrices.
    intraday_freqs : tuple of str
        Frequency subset used for ``intraday_mean_ari``.
    perm_pvalue : dict
        ``{asset_name: float | None}`` -- permutation p-value for the
        overall mean off-diagonal ARI (available if ``run_permutation=True``).
    perm_null_ci : dict
        ``{asset_name: tuple[float, float] | None}`` -- 2.5/97.5 null CI.
    """

    ari_matrix: Dict[str, pd.DataFrame] = field(default_factory=dict)
    ami_matrix: Dict[str, pd.DataFrame] = field(default_factory=dict)
    overall_mean_ari: Dict[str, Optional[float]] = field(default_factory=dict)
    intraday_mean_ari: Dict[str, Optional[float]] = field(default_factory=dict)
    intraday_overall_ari_gap: Dict[str, Optional[float]] = field(default_factory=dict)
    passes_partition: Dict[str, bool] = field(default_factory=dict)
    ari_threshold: float = ARI_THRESHOLD
    freqs: Tuple[str, ...] = PAPER2_FREQS
    intraday_freqs: Tuple[str, ...] = PAPER2_INTRADAY_FREQS
    perm_pvalue: Dict[str, Optional[float]] = field(default_factory=dict)
    perm_null_ci: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a short text summary."""
        lines = [
            "ResInvarianceResult",
            f"  freqs={self.freqs}  intraday_freqs={self.intraday_freqs}",
            f"  ARI threshold={self.ari_threshold}",
        ]
        for asset in self.ari_matrix:
            mean_ari = self.overall_mean_ari.get(asset)
            intra = self.intraday_mean_ari.get(asset)
            excess = self.intraday_overall_ari_gap.get(asset)
            status = "PASS" if self.passes_partition.get(asset) else "FAIL"
            pval = self.perm_pvalue.get(asset)

            def _finite(x: Optional[float]) -> bool:
                return x is not None and np.isfinite(x)

            if not (_finite(mean_ari) and _finite(intra)):
                lines.append(f"  {asset}: insufficient data")
                continue
            pval_str = f"{pval:.4f}" if _finite(pval) else "n/a"
            excess_str = f"{excess:+.3f}" if _finite(excess) else "n/a"
            lines.append(
                f"  {asset}:"
                f"  overall_ARI={mean_ari:.3f} [{status}]"
                f"  intraday_ARI={intra:.3f}"
                f"  intraday_overall_ari_gap={excess_str}"
                f"  perm_p={pval_str}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Intraday-vs-overall ARI gap helper
# ---------------------------------------------------------------------------


def _mean_offdiag(mat: pd.DataFrame) -> Optional[float]:
    """Mean of upper-triangle entries of a square DataFrame.

    Returns ``None`` when the matrix is empty, non-square, smaller than 2x2,
    or contains only NaN entries above the diagonal.
    """
    if mat is None or mat.empty:
        return None
    v = mat.values.astype(float)
    n = v.shape[0]
    if n < 2 or v.shape[0] != v.shape[1]:
        return None
    idx = np.triu_indices(n, k=1)
    offdiag = v[idx]
    if offdiag.size == 0:
        return None
    finite = offdiag[np.isfinite(offdiag)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _intraday_mean_ari(
    ari_matrix: pd.DataFrame,
    intraday_freqs: Tuple[str, ...],
) -> Optional[float]:
    """Extract the mean off-diagonal ARI for the intraday-only sub-matrix.

    Source: Paper 2 src/core/sim_dgp.py lines 435-438 -- the 3-freq
    intraday sub-matrix is built as a separate cross_freq_ari_matrix call
    on aligned_intraday = {f: aligned[f] for f in ("5m","15m","1h")}.
    Here we extract the same submatrix from the already-computed full matrix.
    """
    available = [f for f in intraday_freqs if f in ari_matrix.index]
    if len(available) < 2:
        return None
    sub = ari_matrix.loc[available, available]
    return _mean_offdiag(sub)


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------


def res_invariance_validator(
    model_fn: Callable[[pd.Series], pd.Series],
    resolution_set: Dict[str, Dict[str, pd.Series]],
    spec: Optional[ResolutionSpec] = None,
    run_permutation: bool = True,
    n_perm: int = 500,
    seed: int = 42,
) -> ResInvarianceResult:
    """Run the Paper 2 resolution-invariance check.

    Parameters
    ----------
    model_fn : callable
        ``(prices: pd.Series) -> pd.Series`` of integer regime labels.
        The output Series must have the same DatetimeIndex as the input.
        Called once per (asset, frequency) combination.

        When labels are already available, pass a pre-fitted callable.
        To supply labels directly, use ``resolution_set`` with pre-labelled
        Series and wrap them with ``lambda s: s`` as the model function.
    resolution_set : dict
        ``{asset_name: {freq: pd.Series}}`` where each inner Series is a
        price (or feature) Series with a DatetimeIndex at that frequency.
        The model_fn is called on each inner Series to produce regime labels.

        Alternatively, supply integer-labelled Series directly and use
        ``model_fn = lambda s: s`` to pass labels through.
    spec : ResolutionSpec, optional
        Controls which frequencies are tested and which are considered intraday.
        Defaults to the Paper 2 four-frequency panel ("5m","15m","1h","1d").
    run_permutation : bool, default True
        Whether to compute the permutation p-value for the mean off-diagonal ARI.
    n_perm : int, default 500
        Number of permutations (Paper 2 default; Paper 2 src/core/config.py
        DEFAULT_PERM_N = 500).
    seed : int, default 42
        Random seed for permutation test (Paper 2 DEFAULT_PERM_SEED = 42).

    Returns
    -------
    ResInvarianceResult

    Examples
    --------
    SPY demo across two frequencies with synthetic labels (see
    ``examples/paper2_resolution_invariance.ipynb`` for the full walkthrough)::

        from mrv.invariance import res_invariance_validator, ResolutionSpec
        import pandas as pd
        import numpy as np

        rng = np.random.default_rng(0)
        def make_labels(s):
            return pd.Series(rng.integers(0, 2, len(s)), index=s.index, dtype=int)

        idx = pd.date_range("2026-01-05 09:30", periods=400, freq="5min",
                            tz="America/New_York")
        prices_5m = pd.Series(100 + rng.standard_normal(400).cumsum(), index=idx)
        prices_15m = pd.Series(100 + rng.standard_normal(400).cumsum(), index=idx)
        result = res_invariance_validator(
            model_fn=make_labels,
            resolution_set={"SPY": {"5m": prices_5m, "15m": prices_15m}},
            spec=ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "15m")),
            run_permutation=False,
        )
    """
    if spec is None:
        spec = ResolutionSpec()
    # __post_init__ guarantees intraday_freqs is set; narrow for type-checkers.
    assert spec.intraday_freqs is not None
    intraday_freqs: Tuple[str, ...] = spec.intraday_freqs

    if not resolution_set:
        raise MrvValidationError("res_invariance_validator: resolution_set is empty")

    # 1. Apply model_fn to produce label Series for each (asset, freq).
    labels: Dict[str, Dict[str, pd.Series]] = {}
    for asset_name, freq_inputs in resolution_set.items():
        if len(freq_inputs) < 2:
            raise MrvValidationError(
                f"res_invariance_validator: asset '{asset_name}' has "
                f"{len(freq_inputs)} frequency(ies), need >= 2"
            )
        asset_labels: Dict[str, pd.Series] = {}
        for freq, prices_or_labels in freq_inputs.items():
            raw = model_fn(prices_or_labels)
            if not isinstance(raw, pd.Series):
                raw = pd.Series(raw, index=prices_or_labels.index)
            asset_labels[freq] = raw.astype(int)
        labels[asset_name] = asset_labels

    # 2. Delegate to the shared ResValidator cross-frequency orchestration.
    #    ``analyze_labels`` is the per-asset core that ResValidator.validate
    #    loops over (alignment + ARI/AMI matrices + permutation). Routing the
    #    functional API through it keeps a single orchestration implementation
    #    instead of re-running align + compute_all_metrics + permute here.
    from mrv.validator.res import analyze_labels

    result_ari: Dict[str, pd.DataFrame] = {}
    result_ami: Dict[str, pd.DataFrame] = {}
    result_overall: Dict[str, Optional[float]] = {}
    result_intraday: Dict[str, Optional[float]] = {}
    result_excess: Dict[str, Optional[float]] = {}
    result_passes: Dict[str, bool] = {}
    result_pval: Dict[str, Optional[float]] = {}
    result_ci: Dict[str, Optional[Tuple[float, float]]] = {}

    for asset_name, freq_labels in labels.items():
        analysis = analyze_labels(
            asset_name,
            freq_labels,
            run_permutation=run_permutation,
            n_perm=n_perm,
            seed=seed,
            compute_daily=False,  # wrapper only needs the matrices + permutation
        )
        ari_df = analysis["ari_matrix"]
        ami_df = analysis["ami_matrix"]

        # Derive the intraday-only mean ARI and the intraday-vs-overall gap
        # cheaply from the core's ARI matrix. These are lightweight summaries
        # of the orchestration output, not a second pipeline run.
        overall = _mean_offdiag(ari_df)
        intraday = _intraday_mean_ari(ari_df, intraday_freqs)
        if overall is not None and intraday is not None:
            excess: Optional[float] = round(intraday - overall, 6)
        else:
            excess = None

        pval: Optional[float] = analysis["overall_mean_ari_pvalue_perm"]
        ci: Optional[Tuple[float, float]] = analysis["overall_mean_ari_null_ci"]

        result_ari[asset_name] = ari_df
        result_ami[asset_name] = ami_df
        result_overall[asset_name] = overall
        result_intraday[asset_name] = intraday
        result_excess[asset_name] = excess
        result_passes[asset_name] = (
            overall is not None and np.isfinite(overall) and overall >= ARI_THRESHOLD
        )
        result_pval[asset_name] = pval
        result_ci[asset_name] = ci

    # Report the frequencies actually present in the computed matrices, which
    # may be fewer than spec.freqs when the caller supplies a smaller
    # resolution_set than the default 4-panel spec. Fall back to spec.freqs
    # only when no matrix was produced.
    actual_freqs: Tuple[str, ...] = spec.freqs
    for _asset_ari in result_ari.values():
        if _asset_ari is not None and not _asset_ari.empty:
            actual_freqs = tuple(_asset_ari.columns)
            break

    # Narrow the reported intraday_freqs to those actually present in the
    # computed matrices, so a smaller resolution_set than the default 4-panel
    # spec does not report intraday frequencies that were never used.
    actual_intraday_freqs: Tuple[str, ...] = tuple(
        f for f in intraday_freqs if f in actual_freqs
    )

    return ResInvarianceResult(
        ari_matrix=result_ari,
        ami_matrix=result_ami,
        overall_mean_ari=result_overall,
        intraday_mean_ari=result_intraday,
        intraday_overall_ari_gap=result_excess,
        passes_partition=result_passes,
        ari_threshold=ARI_THRESHOLD,
        freqs=actual_freqs,
        intraday_freqs=actual_intraday_freqs,
        perm_pvalue=result_pval,
        perm_null_ci=result_ci,
    )
