"""Tests for mrv.invariance.res_invariance_validator (Paper 2 wrapper).

Canonical fixture: SPY / CL / USDJPY synthetic 5m labels at 4 frequencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_5m_index(n: int = 400, asset: str = "SPY", seed: int = 42) -> pd.DatetimeIndex:
    """Synthetic 5m DatetimeIndex in NY time."""
    return pd.date_range("2026-01-05 09:30", periods=n, freq="5min", tz="America/New_York")


def _make_label_series(n: int, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="5min", tz="America/New_York")
    return pd.Series(rng.randint(0, 2, n), index=idx, dtype=int)


def _make_resolution_set(
    assets: list[str] = None,
    n_bars: int = 400,
    freqs: tuple = ("5m", "15m", "1h", "1d"),
    seed: int = 42,
) -> dict:
    """Build a minimal resolution_set: one label Series per (asset, freq)."""
    if assets is None:
        assets = ["SPY", "CL", "USDJPY"]
    rng = np.random.RandomState(seed)
    resolution_set = {}
    for asset in assets:
        resolution_set[asset] = {}
        for freq in freqs:
            idx = pd.date_range(
                "2026-01-05 09:30", periods=n_bars, freq="5min", tz="America/New_York"
            )
            resolution_set[asset][freq] = pd.Series(
                rng.randint(0, 2, n_bars), index=idx, dtype=int
            )
    return resolution_set


# Passthrough model: labels are supplied directly in the resolution_set.
def _passthrough(s: pd.Series) -> pd.Series:
    return s


# ---------------------------------------------------------------------------
# ResolutionSpec tests
# ---------------------------------------------------------------------------


class TestResolutionSpec:
    def test_defaults(self):
        from mrv.invariance import PAPER2_FREQS, PAPER2_INTRADAY_FREQS, ResolutionSpec
        spec = ResolutionSpec()
        assert spec.freqs == PAPER2_FREQS
        assert spec.intraday_freqs == PAPER2_INTRADAY_FREQS

    def test_custom_freqs(self):
        from mrv.invariance import ResolutionSpec
        spec = ResolutionSpec(freqs=("5m", "15m", "1h"))
        assert spec.freqs == ("5m", "15m", "1h")
        assert "1d" not in spec.intraday_freqs

    def test_requires_two_freqs(self):
        from mrv.invariance import ResolutionSpec
        with pytest.raises(ValueError, match=">= 2"):
            ResolutionSpec(freqs=("5m",))

    def test_intraday_freq_not_in_freqs_raises(self):
        from mrv.invariance import ResolutionSpec
        with pytest.raises(ValueError, match="not in freqs"):
            ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "1h"))


# ---------------------------------------------------------------------------
# ResInvarianceResult tests
# ---------------------------------------------------------------------------


class TestResInvarianceResult:
    def _make_result(self):
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        rs = _make_resolution_set(["SPY"], n_bars=200, freqs=("5m", "15m"))
        spec = ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "15m"))
        return res_invariance_validator(
            model_fn=_passthrough,
            resolution_set=rs,
            spec=spec,
            run_permutation=False,
        )

    def test_ari_matrix_shape(self):
        result = self._make_result()
        assert result.ari_matrix["SPY"].shape == (2, 2)

    def test_ami_matrix_shape(self):
        result = self._make_result()
        assert result.ami_matrix["SPY"].shape == (2, 2)

    def test_diagonal_is_one(self):
        result = self._make_result()
        diag = np.diag(result.ari_matrix["SPY"].values)
        assert all(np.isclose(v, 1.0) for v in diag)

    def test_ari_symmetric(self):
        result = self._make_result()
        mat = result.ari_matrix["SPY"].values
        assert np.allclose(mat, mat.T, equal_nan=True)

    def test_overall_mean_ari_exists(self):
        result = self._make_result()
        v = result.overall_mean_ari["SPY"]
        assert v is not None and np.isfinite(v)

    def test_passes_partition_type(self):
        result = self._make_result()
        assert isinstance(result.passes_partition["SPY"], bool)

    def test_perm_pvalue_none_when_skipped(self):
        result = self._make_result()
        assert result.perm_pvalue["SPY"] is None

    def test_summary_runs(self):
        result = self._make_result()
        s = result.summary()
        assert "ResInvarianceResult" in s


# ---------------------------------------------------------------------------
# res_invariance_validator main interface tests
# ---------------------------------------------------------------------------


class TestResInvarianceValidator:
    def test_import_from_top_level(self):
        from mrv.invariance import res_invariance_validator
        assert callable(res_invariance_validator)

    def test_import_from_mrv_top_level(self):
        from mrv import res_invariance_validator
        assert callable(res_invariance_validator)

    def test_two_freq_minimal(self):
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        rs = _make_resolution_set(["SPY"], n_bars=200, freqs=("5m", "15m"))
        spec = ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "15m"))
        result = res_invariance_validator(_passthrough, rs, spec=spec, run_permutation=False)
        assert "SPY" in result.ari_matrix

    def test_requires_two_freqs(self):
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        idx = pd.date_range("2026-01-05 09:30", periods=100, freq="5min", tz="America/New_York")
        with pytest.raises(ValueError, match="need >= 2"):
            res_invariance_validator(
                _passthrough,
                {"SPY": {"5m": pd.Series(np.zeros(100, dtype=int), index=idx)}},
                spec=ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m",)),
            )

    def test_empty_resolution_set_raises(self):
        from mrv.invariance import res_invariance_validator
        with pytest.raises(ValueError, match="empty"):
            res_invariance_validator(_passthrough, {})

    def test_identical_labels_ari_one(self):
        """When all freqs share the same label sequence, ARI should be ~1."""
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        idx = pd.date_range("2026-01-05 09:30", periods=300, freq="5min", tz="America/New_York")
        base = pd.Series(np.random.RandomState(0).randint(0, 2, 300), index=idx, dtype=int)
        rs = {"SPY": {"5m": base.copy(), "15m": base.copy(), "1h": base.copy()}}
        spec = ResolutionSpec(freqs=("5m", "15m", "1h"), intraday_freqs=("5m", "15m", "1h"))
        result = res_invariance_validator(_passthrough, rs, spec=spec, run_permutation=False)
        ari = result.overall_mean_ari["SPY"]
        assert ari is not None and abs(ari - 1.0) < 0.01

    def test_random_labels_low_ari(self):
        """Independent random labels should yield near-zero ARI."""
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        rs = _make_resolution_set(["SPY"], n_bars=500, freqs=("5m", "15m", "1h"), seed=7)
        spec = ResolutionSpec(freqs=("5m", "15m", "1h"), intraday_freqs=("5m", "15m", "1h"))
        result = res_invariance_validator(_passthrough, rs, spec=spec, run_permutation=False)
        ari = result.overall_mean_ari["SPY"]
        assert ari is not None and abs(ari) < 0.2

    def test_within_intraday_excess_computed(self):
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        # 4-freq: intraday pairs among (5m, 15m, 1h); daily pair adds noise
        rs = _make_resolution_set(["SPY"], n_bars=400, freqs=("5m", "15m", "1h", "1d"), seed=0)
        spec = ResolutionSpec()  # default Paper 2 spec
        result = res_invariance_validator(_passthrough, rs, spec=spec, run_permutation=False)
        excess = result.within_intraday_excess["SPY"]
        # excess should be a finite float (positive or negative)
        assert excess is not None and np.isfinite(excess)

    def test_within_intraday_excess_identity(self):
        """When intraday_freqs == all freqs, excess must be zero."""
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        rs = _make_resolution_set(["SPY"], n_bars=300, freqs=("5m", "15m"), seed=1)
        spec = ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "15m"))
        result = res_invariance_validator(_passthrough, rs, spec=spec, run_permutation=False)
        excess = result.within_intraday_excess["SPY"]
        # overall == intraday when intraday covers all freqs
        assert excess is not None and abs(excess) < 1e-9

    def test_permutation_pvalue_returned(self):
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        rs = _make_resolution_set(["SPY"], n_bars=300, freqs=("5m", "15m"), seed=42)
        spec = ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "15m"))
        result = res_invariance_validator(
            _passthrough, rs, spec=spec, run_permutation=True, n_perm=50, seed=42
        )
        pval = result.perm_pvalue["SPY"]
        # 300 obs is well above the permutation-test minimum, so pval is defined.
        assert pval is not None
        assert 0.0 < pval <= 1.0

    def test_three_asset_fixture(self):
        """Canonical SPY / CL / USDJPY fixture: all three assets processed."""
        from mrv.invariance import ResolutionSpec, res_invariance_validator
        rs = _make_resolution_set(["SPY", "CL", "USDJPY"], n_bars=300,
                                   freqs=("5m", "15m", "1h"), seed=99)
        spec = ResolutionSpec(freqs=("5m", "15m", "1h"), intraday_freqs=("5m", "15m", "1h"))
        result = res_invariance_validator(
            _passthrough, rs, spec=spec, run_permutation=False
        )
        for asset in ["SPY", "CL", "USDJPY"]:
            assert asset in result.ari_matrix
            assert asset in result.ami_matrix
            assert asset in result.overall_mean_ari
            assert result.ari_matrix[asset].shape == (3, 3)

    def test_model_fn_called_per_asset_freq(self):
        """Verify model_fn is invoked on each (asset, freq) input Series."""
        call_log = []

        def logging_model(s: pd.Series) -> pd.Series:
            call_log.append(len(s))
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index)

        from mrv.invariance import ResolutionSpec, res_invariance_validator
        rs = _make_resolution_set(["SPY", "CL"], n_bars=100, freqs=("5m", "15m"), seed=3)
        spec = ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "15m"))
        res_invariance_validator(logging_model, rs, spec=spec, run_permutation=False)
        # 2 assets x 2 freqs = 4 calls
        assert len(call_log) == 4

    def test_ari_threshold_exposed(self):
        from mrv.invariance import ResInvarianceResult
        from mrv.validator.metrics import ARI_THRESHOLD
        r = ResInvarianceResult()
        assert r.ari_threshold == ARI_THRESHOLD


# ---------------------------------------------------------------------------
# Integration: top-level mrv.invariance import surface
# ---------------------------------------------------------------------------


class TestInvariancePublicSurface:
    def test_all_names_importable(self):
        from mrv import invariance
        for name in ["ResInvarianceResult", "ResolutionSpec", "res_invariance_validator",
                     "RepInvarianceResult", "rep_invariance_validator",
                     "PAPER2_FREQS", "PAPER2_INTRADAY_FREQS"]:
            assert hasattr(invariance, name), f"missing: {name}"

    def test_paper2_freqs_constant(self):
        from mrv.invariance import PAPER2_FREQS
        assert PAPER2_FREQS == ("5m", "15m", "1h", "1d")

    def test_paper2_intraday_freqs_constant(self):
        from mrv.invariance import PAPER2_INTRADAY_FREQS
        assert PAPER2_INTRADAY_FREQS == ("5m", "15m", "1h")
