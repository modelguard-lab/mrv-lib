"""Tests for mrv.validator.res (Resolution Invariance / Paper 2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv_5m


class TestResFeatures:
    def test_window_spec(self):
        from mrv.validator.res import _window_spec
        assert _window_spec("5m") == "120min"
        assert _window_spec("15m") == "120min"
        assert _window_spec("1h") == 24
        assert _window_spec("1d") == 5
        assert _window_spec("5m", 2.0) == "240min"
        assert _window_spec("1h", 0.5) == 12

    def test_window_spec_invalid(self):
        from mrv.validator.res import _window_spec
        with pytest.raises(ValueError, match="Unsupported"):
            _window_spec("3m")

    def test_features(self):
        from mrv.validator.res import _features
        df = make_ohlcv_5m(10)
        feats = _features(df, "5m")
        assert "ret" in feats.columns and "vol" in feats.columns
        assert feats["vol"].notna().sum() > 0

    def test_features_calendar_window(self):
        from mrv.validator.res import _features
        assert _features(make_ohlcv_5m(10), "5m", calendar_window="6h")["vol"].notna().sum() > 0

    def test_robust_filter_returns_non_cl(self):
        from mrv.validator.res import _robust_filter_returns
        ret = pd.Series(np.random.randn(100) * 0.01)
        pd.testing.assert_series_equal(_robust_filter_returns(ret, "SPY", "5m"), ret)

    def test_robust_filter_returns_cl(self):
        from mrv.validator.res import _robust_filter_returns
        np.random.seed(42)
        ret = pd.Series(np.random.randn(200) * 0.001)
        ret.iloc[50] = 0.5
        assert abs(_robust_filter_returns(ret, "CL", "5m").iloc[50]) < 0.5


class TestResRegimeFitting:
    def test_fit_regime_gmm(self):
        from mrv.validator.res import _fit_regime_gmm, _features
        feats = _features(make_ohlcv_5m(15), "5m")
        labels, fallback = _fit_regime_gmm(feats, n_components=2, freq="5m")
        assert len(labels) == len(feats)
        assert set(labels.unique()).issubset({0, 1})

    def test_fit_regime_gmm_constant_vol(self):
        from mrv.validator.res import _fit_regime_gmm
        labels, _ = _fit_regime_gmm(pd.DataFrame({"ret": np.zeros(100), "vol": np.ones(100) * 0.01}), n_components=2)
        assert (labels == 0).all()

    def test_fit_regime_model_dispatch(self):
        from mrv.validator.res import _fit_regime_model, _features
        labels, _ = _fit_regime_model(_features(make_ohlcv_5m(15), "5m"), model="gmm", n_components=2, freq="5m")
        assert len(labels) > 0


class TestResAlignment:
    def test_align_regimes_to_5m(self):
        from mrv.validator.res import _align_regimes_to_5m
        idx_5m = pd.date_range("2026-01-05 09:30", periods=100, freq="5min", tz="America/New_York")
        idx_1h = pd.date_range("2026-01-05 10:00", periods=7, freq="1h", tz="America/New_York")
        aligned = _align_regimes_to_5m({
            "5m": pd.Series(np.random.randint(0, 2, 100), index=idx_5m),
            "1h": pd.Series(np.random.randint(0, 2, 7), index=idx_1h),
        }, idx_5m)
        assert len(aligned["5m"]) == 100
        assert len(aligned["1h"]) == 100

    def test_compute_ari_matrix_identical(self):
        from mrv.validator.res import _compute_ari_matrix, FREQS
        idx = pd.date_range("2026-01-05 09:30", periods=200, freq="5min", tz="America/New_York")
        base = pd.Series(np.random.randint(0, 2, 200), index=idx)
        aligned = {freq: base.copy() for freq in FREQS}
        mat = _compute_ari_matrix(aligned)
        assert mat.shape == (4, 4)
        assert all(v == pytest.approx(1.0) for v in mat.values[np.triu_indices(4, k=1)])

    def test_compute_ari_matrix_random(self):
        from mrv.validator.res import _compute_ari_matrix, FREQS
        np.random.seed(42)
        idx = pd.date_range("2026-01-05 09:30", periods=500, freq="5min", tz="America/New_York")
        mat = _compute_ari_matrix({freq: pd.Series(np.random.randint(0, 2, 500), index=idx) for freq in FREQS})
        assert all(-0.1 < v < 0.15 for v in mat.values[np.triu_indices(4, k=1)])


class TestResExtraMetrics:
    def test_compute_extra_metrics(self):
        from mrv.validator.res import _compute_extra_metrics, FREQS
        idx = pd.date_range("2026-01-05 09:30", periods=200, freq="5min", tz="America/New_York")
        base = pd.Series(np.random.randint(0, 2, 200), index=idx)
        extra = _compute_extra_metrics({freq: base.copy() for freq in FREQS})
        assert extra["ami"].iloc[0, 1] == pytest.approx(1.0, abs=0.01)
        assert extra["vi"].iloc[0, 1] == pytest.approx(0.0, abs=0.01)

    def test_mean_offdiag(self):
        from mrv.validator.res import _mean_offdiag
        assert _mean_offdiag(pd.DataFrame(np.eye(3))) == pytest.approx(0.0)
        assert _mean_offdiag(pd.DataFrame()) is None
        assert _mean_offdiag(None) is None

    def test_permute_pvalue(self):
        from mrv.validator.res import _permute_pvalue_mean_offdiag_ari, FREQS
        np.random.seed(42)
        idx = pd.date_range("2026-01-05 09:30", periods=200, freq="5min", tz="America/New_York")
        base = pd.Series(np.random.randint(0, 2, 200), index=idx)
        p, ci = _permute_pvalue_mean_offdiag_ari({freq: base.copy() for freq in FREQS}, n_perm=50, seed=42)
        assert p is not None and p < 0.1
        assert ci is not None and len(ci) == 2


class TestResBlockPermutation:
    def test_block_permute_identical(self):
        from mrv.validator.res import _block_permute_pvalue_mean_offdiag_ari, FREQS
        idx = pd.date_range("2026-01-05 09:30", periods=300, freq="5min", tz="America/New_York")
        base = pd.Series(np.random.randint(0, 2, 300), index=idx)
        p, ci = _block_permute_pvalue_mean_offdiag_ari(
            {freq: base.copy() for freq in FREQS}, n_perm=50, block_size=20, seed=42)
        assert p is not None and p < 0.1
        assert ci is not None and len(ci) == 2

    def test_block_permute_too_short(self):
        from mrv.validator.res import _block_permute_pvalue_mean_offdiag_ari, FREQS
        idx = pd.date_range("2026-01-05 09:30", periods=50, freq="5min", tz="America/New_York")
        base = pd.Series(np.random.randint(0, 2, 50), index=idx)
        p, ci = _block_permute_pvalue_mean_offdiag_ari(
            {freq: base.copy() for freq in FREQS}, block_size=50)
        assert p is None and ci is None


class TestResGmmDiagnostics:
    def test_gmm_fit_diagnostics(self):
        from mrv.validator.res import _gmm_fit_diagnostics
        df = make_ohlcv_5m(15)
        diag = _gmm_fit_diagnostics(df, "5m", n_components=2)
        assert "bic" in diag and "aic" in diag
        assert "separation" in diag and "overlap" in diag
        assert "means" in diag and len(diag["means"]) == 2
        assert "stds" in diag and len(diag["stds"]) == 2
        assert "weights" in diag and len(diag["weights"]) == 2

    def test_gmm_fit_diagnostics_too_few(self):
        from mrv.validator.res import _gmm_fit_diagnostics
        # Create a DataFrame with only 2 rows — too few for K=2 GMM (needs >= 4)
        idx = pd.date_range("2026-01-05 09:30", periods=2, freq="5min", tz="America/New_York")
        df = pd.DataFrame({"Open": [100, 101], "High": [102, 103], "Low": [99, 100], "Close": [101, 102]}, index=idx)
        diag = _gmm_fit_diagnostics(df, "5m", n_components=2)
        assert np.isnan(diag["bic"])


class TestResExpandingWindow:
    def test_fit_regime_expanding(self):
        from mrv.validator.res import _fit_regime_expanding
        df = make_ohlcv_5m(15)
        labels, diag = _fit_regime_expanding(df, "5m", n_components=2, min_train_bars=50)
        assert len(labels) == len(df)
        assert set(labels.unique()).issubset({0, 1})
        assert "refit_count" in diag
        assert "full_vs_expanding_agreement" in diag
        assert 0.0 <= diag["full_vs_expanding_agreement"] <= 1.0


class TestResClRollWeek:
    def test_cl_roll_week_analysis(self):
        from mrv.validator.res import _cl_roll_week_analysis, FREQS
        idx = pd.date_range("2026-01-05 09:30", periods=5000, freq="5min", tz="America/New_York")
        aligned = {freq: pd.Series(np.random.randint(0, 2, 5000), index=idx) for freq in FREQS}
        result = _cl_roll_week_analysis(aligned, idx)
        assert "roll_week_mean_ari" in result
        assert "nonroll_week_mean_ari" in result
        assert result["roll_week_bars"] + result["nonroll_week_bars"] == len(idx)

    def test_cl_roll_week_with_dates(self):
        from mrv.validator.res import _cl_roll_week_analysis, FREQS
        idx = pd.date_range("2026-01-05 09:30", periods=5000, freq="5min", tz="America/New_York")
        aligned = {freq: pd.Series(np.random.randint(0, 2, 5000), index=idx) for freq in FREQS}
        result = _cl_roll_week_analysis(aligned, idx, roll_dates=["2026-01-15"])
        assert result["roll_week_bars"] > 0


class TestResAnalyzeAsset:
    def test_analyze_asset(self):
        from mrv.validator.res import analyze_asset
        result = analyze_asset("TEST", make_ohlcv_5m(15), model="gmm", n_components=2)
        assert result["ari_matrix"].shape == (4, 4)
        assert all(f in result["crisis_shares"] for f in ("5m", "15m", "1h", "1d"))
        assert isinstance(result["tod_crisis_distribution"], pd.DataFrame)
        # New Paper 2 fields
        assert "block_perm_pvalue" in result
        assert "gmm_diagnostics" in result
        assert "expanding_ari_matrix" in result
        assert "expanding_mean_ari" in result
        assert "expanding_diagnostics" in result
        assert "rolling_ari_median" in result
        assert "cl_roll_analysis" in result  # None for non-CL

    def test_analyze_asset_with_event_window(self):
        from mrv.validator.res import analyze_asset
        df = make_ohlcv_5m(15)
        start = df.index[0].strftime("%Y-%m-%d")
        end = (df.index[0] + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        result = analyze_asset("TEST", df, event_window=(start, end), calm_window=(start, end))
        assert result.get("event_mean_ari") is not None or result.get("event_ari_matrix") is not None

    def test_analyze_asset_cl_roll(self):
        from mrv.validator.res import analyze_asset
        result = analyze_asset("CL", make_ohlcv_5m(15), model="gmm", n_components=2)
        assert result["cl_roll_analysis"] is not None
        assert "roll_week_mean_ari" in result["cl_roll_analysis"]


class TestResValidator:
    def test_validator_with_data(self, tmp_path):
        from mrv.validator.res import ResValidator
        result = ResValidator({
            "validator": {"report_dir": str(tmp_path), "report_name": "test_{date}", "res": {"model": "gmm", "n_states": 2}},
        }).validate(prices={"TEST": make_ohlcv_5m(15)})
        run_dir = Path(result["run_dir"])
        assert (run_dir / "result.json").exists()
        assert (run_dir / "TEST_timeline.png").exists()

    def test_validator_imported(self):
        from mrv.validator import ResValidator
        assert ResValidator.name == "res"

    def test_robustness_sweep(self):
        from mrv.validator.res import run_robustness
        summary = run_robustness(make_ohlcv_5m(15), "TEST", k_values=(2,), window_scales=(1.0,))
        assert not summary.empty
        assert "overall_mean_ari" in summary.columns


class TestResSubsetAndTOD:
    def test_subset_index_by_dates(self):
        from mrv.validator.res import _subset_index_by_dates
        idx = pd.date_range("2026-01-05 09:30", periods=1000, freq="5min", tz="America/New_York")
        subset = _subset_index_by_dates(idx, "2026-01-05", "2026-01-05")
        assert 0 < len(subset) < len(idx)

    def test_tod_crisis_distribution(self):
        from mrv.validator.res import _compute_tod_crisis_distribution, FREQS
        idx = pd.date_range("2026-01-05 09:30", periods=500, freq="5min", tz="America/New_York")
        tod = _compute_tod_crisis_distribution({freq: pd.Series(np.random.randint(0, 2, 500), index=idx) for freq in FREQS})
        assert {"freq", "hour", "crisis_share"}.issubset(tod.columns)

    def test_daily_outputs(self):
        from mrv.validator.res import _compute_daily_outputs, FREQS
        idx = pd.date_range("2026-01-05 09:30", periods=1000, freq="5min", tz="America/New_York")
        daily, daily_pair, rolling, rolling_pair = _compute_daily_outputs(
            {freq: pd.Series(np.random.randint(0, 2, 1000), index=idx) for freq in FREQS}, rolling_days=3)
        assert not daily.empty
        assert "mean_offdiag_ari" in daily.columns
