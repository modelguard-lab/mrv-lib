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


def _make_multiday_labels(n_days=8, bars_per_day=12, freqs=("5m", "15m", "1h", "1d"), seed=3):
    """Create test labels spanning ``n_days`` distinct calendar days.

    The daily/rolling summaries in ``compute_daily_outputs`` group by calendar
    day and need at least ``rolling_days`` distinct days for the rolling frames
    to be non-empty, so tests that exercise that path use this fixture.
    """
    rng = np.random.RandomState(seed)
    idx = None
    for d in range(n_days):
        start = pd.Timestamp("2026-01-05 09:30", tz="America/New_York") + pd.Timedelta(days=d)
        day_idx = pd.date_range(start, periods=bars_per_day, freq="5min")
        idx = day_idx if idx is None else idx.append(day_idx)
    return {freq: pd.Series(rng.randint(0, 2, len(idx)), index=idx) for freq in freqs}


class TestResAlignment:
    def test_align_labels_to_finest(self):
        from mrv.validator.res import align_labels_to_finest
        idx_5m = pd.date_range("2026-01-05 09:30", periods=100, freq="5min", tz="America/New_York")
        idx_1h = pd.date_range("2026-01-05 10:00", periods=7, freq="1h", tz="America/New_York")
        s_1h = pd.Series([1, 0, 1, 0, 1, 0, 1], index=idx_1h)
        aligned = align_labels_to_finest({
            # Seeded for determinism (zeros would also work; the 5m values are
            # not asserted on here, only alignment and the 1h ffill).
            "5m": pd.Series(np.random.RandomState(0).randint(0, 2, 100), index=idx_5m),
            "1h": s_1h,
        })
        # The common index starts at the latest first-observation (the 1h
        # start, 10:00), so the 5m rows before 10:00 are dropped rather than
        # paired against a zero-filled 1h head. 100 5m bars from 09:30 leave 94
        # from 10:00 onward.
        assert aligned["5m"].index.equals(aligned["1h"].index)
        assert aligned["5m"].index[0] == idx_1h[0]
        assert len(aligned["5m"]) == len(aligned["1h"]) == 94
        # No spurious regime-0 head: the first aligned 1h label is its real
        # first observation, not a fillna(0).
        assert aligned["1h"].iloc[0] == 1
        # Interior ffill check: a 5m bar strictly between the first (10:00) and
        # second (11:00) hourly boundary must carry the first hourly label (1),
        # not the second (0). Guards an interior ffill off-by-one.
        ts_mid = idx_1h[0] + pd.Timedelta(minutes=30)
        assert aligned["1h"].loc[ts_mid] == 1
        # The bar exactly at the second boundary takes the second label.
        assert aligned["1h"].loc[idx_1h[1]] == 0

    def test_align_empty_dict(self):
        from mrv.validator.res import align_labels_to_finest
        assert align_labels_to_finest({}) == {}

    def test_align_single_series_passthrough(self):
        from mrv.validator.res import align_labels_to_finest
        idx = pd.date_range("2026-01-05 09:30", periods=50, freq="5min",
                            tz="America/New_York")
        s = pd.Series(np.arange(50) % 2, index=idx, dtype=int)
        aligned = align_labels_to_finest({"5m": s})
        assert list(aligned.keys()) == ["5m"]
        assert aligned["5m"].index.equals(s.index)
        assert aligned["5m"].tolist() == s.tolist()

    def test_align_disjoint_start_dates_empty(self):
        """Coarse series begins after the last fine bar: no overlap, so the
        common index (>= the latest first-observation) is empty and every
        aligned series is empty.
        """
        from mrv.validator.res import align_labels_to_finest
        idx_5m = pd.date_range("2026-01-05 09:30", periods=20, freq="5min",
                               tz="America/New_York")  # ends 11:05
        idx_1h = pd.date_range("2026-01-05 12:00", periods=5, freq="1h",
                               tz="America/New_York")  # starts after last 5m bar
        aligned = align_labels_to_finest({
            "5m": pd.Series(np.ones(20, dtype=int), index=idx_5m),
            "1h": pd.Series(np.ones(5, dtype=int), index=idx_1h),
        })
        assert len(aligned["5m"]) == 0
        assert len(aligned["1h"]) == 0

    def test_disjoint_surfaces_overall_mean_ari_none(self):
        """The res-invariance wrapper must surface overall_mean_ari as None on
        fully-disjoint inputs (insufficient data), not a misleading pass/fail
        computed from an empty alignment.
        """
        from mrv.invariance import ResolutionSpec, res_invariance_validator

        idx_5m = pd.date_range("2026-01-05 09:30", periods=20, freq="5min",
                               tz="America/New_York")
        idx_1h = pd.date_range("2026-01-05 12:00", periods=5, freq="1h",
                               tz="America/New_York")
        rs = {
            "SPY": {
                "5m": pd.Series(np.ones(20, dtype=int), index=idx_5m),
                "1h": pd.Series(np.ones(5, dtype=int), index=idx_1h),
            }
        }
        spec = ResolutionSpec(freqs=("5m", "1h"), intraday_freqs=("5m", "1h"))
        result = res_invariance_validator(
            lambda s: s, rs, spec=spec, run_permutation=False
        )
        assert result.overall_mean_ari["SPY"] is None
        assert result.passes_partition["SPY"] is False

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
        from mrv.validator.res import analyze_labels, mean_offdiag
        labels = _make_aligned_labels(500)
        # run_permutation=False: this test asserts the deterministic metric
        # outputs; skipping the 500-perm null keeps it fast.
        result = analyze_labels("TEST", labels, run_permutation=False)
        assert result["ari_matrix"].shape == (4, 4)
        assert all(f in result["crisis_shares"] for f in ("5m", "15m", "1h", "1d"))
        # Value check: overall_mean_ari is exactly the ARI-matrix off-diagonal mean.
        overall = result["overall_mean_ari"]
        assert overall is not None
        assert overall == pytest.approx(mean_offdiag(result["ari_matrix"]), abs=1e-12)
        # rolling_ari_median is a float (NaN when the fixture is shorter than one
        # rolling window), value-strengthening the old presence-only check.
        assert isinstance(result["rolling_ari_median"], float)

    def test_analyze_labels_perm_gating_and_seed(self):
        """T-P2a: run_permutation=False nulls the perm fields; identical seeds
        give identical perm p-values."""
        from mrv.validator.res import analyze_labels
        labels = _make_aligned_labels(300, seed=5)

        off = analyze_labels("TEST", labels, run_permutation=False)
        assert off["overall_mean_ari_pvalue_perm"] is None
        assert off["overall_mean_ari_null_ci"] is None

        r1 = analyze_labels("TEST", labels, run_permutation=True, n_perm=50, seed=42)
        r2 = analyze_labels("TEST", labels, run_permutation=True, n_perm=50, seed=42)
        assert r1["overall_mean_ari_pvalue_perm"] is not None
        assert (
            r1["overall_mean_ari_pvalue_perm"]
            == r2["overall_mean_ari_pvalue_perm"]
        )
        assert r1["overall_mean_ari_null_ci"] == r2["overall_mean_ari_null_ci"]

    def test_compute_daily_gates_only_daily_outputs(self):
        """T-P2: ``compute_daily`` gates ONLY the daily/rolling frames. The four
        per-day frames must be empty when False and non-empty when True, while
        the headline cross-frequency numbers stay bit-identical either way."""
        from pandas.testing import assert_frame_equal

        from mrv.validator.res import analyze_labels

        labels = _make_multiday_labels(n_days=6)
        common = dict(run_permutation=False, rolling_days=3)
        on = analyze_labels("TEST", labels, compute_daily=True, **common)
        off = analyze_labels("TEST", labels, compute_daily=False, **common)

        for key in ("daily_df", "daily_pair_df", "rolling_df", "rolling_pair_df"):
            assert off[key].empty, f"{key} must be empty when compute_daily=False"
            assert not on[key].empty, f"{key} must be non-empty when compute_daily=True"

        # The flag must not move any headline number.
        assert_frame_equal(on["ari_matrix"], off["ari_matrix"])
        assert_frame_equal(on["ami_matrix"], off["ami_matrix"])
        assert_frame_equal(on["vi_matrix"], off["vi_matrix"])
        assert on["overall_mean_ari"] == off["overall_mean_ari"]

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
        # Multi-day fixture + small rolling_days so the daily/rolling CSV path is
        # actually exercised (compute_daily=True is the validator default).
        labels = _make_multiday_labels(n_days=6)
        result = ResValidator(cfg={
            "validator": {
                "report_dir": str(tmp_path), "report_name": "test_{date}",
                "res": {"rolling_days": 3},
            },
        }).validate(labels={"TEST": labels})
        run_dir = Path(result["run_dir"])
        assert (run_dir / "result.json").exists()

        daily_csv = run_dir / "TEST_daily_summary.csv"
        rolling_csv = run_dir / "TEST_rolling_ari.csv"
        assert daily_csv.exists(), "daily summary CSV should be written"
        assert rolling_csv.exists(), "rolling ARI CSV should be written"
        assert len(pd.read_csv(daily_csv)) > 0
        assert len(pd.read_csv(rolling_csv)) > 0

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
