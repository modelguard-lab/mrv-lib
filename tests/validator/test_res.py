"""Tests for mrv.validator.res (Resolution Invariance)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_aligned_labels(n=200, freqs=("5m", "15m", "1h", "1d"), seed=42):
    """Create test label Series at multiple frequencies."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="5min", tz="America/New_York")
    return {freq: pd.Series(rng.randint(0, 2, n), index=idx) for freq in freqs}


class TestResAlignment:
    def test_align_labels_to_finest(self):
        from mrv.validator.res import align_labels_to_finest
        idx_5m = pd.date_range("2026-01-05 09:30", periods=100, freq="5min", tz="America/New_York")
        idx_1h = pd.date_range("2026-01-05 10:00", periods=7, freq="1h", tz="America/New_York")
        aligned = align_labels_to_finest({
            "5m": pd.Series(np.random.randint(0, 2, 100), index=idx_5m),
            "1h": pd.Series(np.random.randint(0, 2, 7), index=idx_1h),
        })
        assert len(aligned["5m"]) == 100
        assert len(aligned["1h"]) == 100

    def test_compute_ari_matrix_identical(self):
        from mrv.validator.res import compute_ari_matrix
        idx = pd.date_range("2026-01-05 09:30", periods=200, freq="5min", tz="America/New_York")
        base = pd.Series(np.random.randint(0, 2, 200), index=idx)
        freqs = ("5m", "15m", "1h", "1d")
        aligned = {freq: base.copy() for freq in freqs}
        mat = compute_ari_matrix(aligned)
        assert mat.shape == (4, 4)
        assert all(v == pytest.approx(1.0) for v in mat.values[np.triu_indices(4, k=1)])

    def test_compute_ari_matrix_random(self):
        from mrv.validator.res import compute_ari_matrix
        np.random.seed(42)
        aligned = _make_aligned_labels(500)
        mat = compute_ari_matrix(aligned)
        assert all(-0.1 < v < 0.15 for v in mat.values[np.triu_indices(4, k=1)])


class TestResMetrics:
    def test_compute_all_metrics(self):
        from mrv.validator.res import compute_all_metrics
        idx = pd.date_range("2026-01-05 09:30", periods=200, freq="5min", tz="America/New_York")
        base = pd.Series(np.random.randint(0, 2, 200), index=idx)
        freqs = ("5m", "15m", "1h", "1d")
        all_metrics = compute_all_metrics({freq: base.copy() for freq in freqs})
        assert all_metrics["ami"].iloc[0, 1] == pytest.approx(1.0, abs=0.01)
        assert all_metrics["vi"].iloc[0, 1] == pytest.approx(0.0, abs=0.01)

    def test_mean_offdiag(self):
        from mrv.validator.res import mean_offdiag
        assert mean_offdiag(pd.DataFrame(np.eye(3))) == pytest.approx(0.0)
        assert mean_offdiag(pd.DataFrame()) is None
        assert mean_offdiag(None) is None

    def test_permute_pvalue(self):
        from mrv.validator.res import permute_pvalue_mean_offdiag_ari
        np.random.seed(42)
        aligned = _make_aligned_labels(200)
        base = next(iter(aligned.values()))
        identical = {k: base.copy() for k in aligned}
        p, ci = permute_pvalue_mean_offdiag_ari(identical, n_perm=50, seed=42)
        assert p is not None and p < 0.1
        assert ci is not None and len(ci) == 2


class TestResSubset:
    def test_subset_index_by_dates(self):
        from mrv.validator.res import subset_index_by_dates
        idx = pd.date_range("2026-01-05 09:30", periods=1000, freq="5min", tz="America/New_York")
        subset = subset_index_by_dates(idx, "2026-01-05", "2026-01-05")
        assert 0 < len(subset) < len(idx)

    def test_daily_outputs(self):
        from mrv.validator.res import compute_daily_outputs
        aligned = _make_aligned_labels(1000)
        daily, daily_pair, rolling, rolling_pair = compute_daily_outputs(
            aligned, rolling_days=3)
        assert not daily.empty
        assert "mean_offdiag_ari" in daily.columns


class TestResAnalyzeLabels:
    def test_analyze_labels(self):
        from mrv.validator.res import analyze_labels
        labels = _make_aligned_labels(500)
        result = analyze_labels("TEST", labels)
        assert result["ari_matrix"].shape == (4, 4)
        assert all(f in result["crisis_shares"] for f in ("5m", "15m", "1h", "1d"))
        assert result.get("overall_mean_ari") is not None
        assert "rolling_ari_median" in result

    def test_analyze_labels_with_event_window(self):
        from mrv.validator.res import analyze_labels
        labels = _make_aligned_labels(500)
        # The 500-bar 5m fixture starts 2026-01-05 09:30 and spans ~1.7 days,
        # so this event window overlaps the fixture dates and yields real data.
        result = analyze_labels("TEST", labels,
                               event_window=("2026-01-05", "2026-01-06"),
                               calm_window=("2026-01-07", "2026-01-08"))
        event_ari = result.get("event_mean_ari")
        assert isinstance(event_ari, float) and np.isfinite(event_ari)
        event_mat = result.get("event_ari_matrix")
        assert isinstance(event_mat, pd.DataFrame) and not event_mat.empty
        assert event_mat.shape == (4, 4)


class TestResValidator:
    def test_validator_with_labels(self, tmp_path):
        from mrv.validator.res import ResValidator
        labels = _make_aligned_labels(500)
        result = ResValidator(cfg={
            "validator": {"report_dir": str(tmp_path), "report_name": "test_{date}", "res": {}},
        }).validate(labels={"TEST": labels})
        run_dir = Path(result["run_dir"])
        assert (run_dir / "result.json").exists()

    def test_validator_imported(self):
        from mrv.validator import ResValidator
        assert ResValidator.name == "res"

    def test_validator_requires_two_freqs(self, tmp_path):
        from mrv.validator.res import ResValidator
        idx = pd.date_range("2026-01-05 09:30", periods=100, freq="5min", tz="America/New_York")
        with pytest.raises(ValueError, match="need >= 2"):
            ResValidator(cfg={
                "validator": {"report_dir": str(tmp_path), "report_name": "t_{date}", "res": {}},
            }).validate(labels={"A": {"5m": pd.Series(np.zeros(100, dtype=int), index=idx)}})
