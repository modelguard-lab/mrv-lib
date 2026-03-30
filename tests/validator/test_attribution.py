"""Tests for mrv.validator.attribution."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_price_series


class TestAttribution:
    def test_loo_factor_attribution(self):
        from mrv.validator.attribution import loo_factor_attribution
        np.random.seed(42)
        base = np.random.randint(0, 3, 200)
        result = loo_factor_attribution(
            {"A": base.copy(), "B": base.copy(), "C": np.random.randint(0, 3, 200)},
            baseline_mean_ari=0.5,
        )
        assert result["worst_contributor"] == "C"
        assert result["scores"]["C"] > 0

    def test_loo_too_few_sets(self):
        from mrv.validator.attribution import loo_factor_attribution
        assert loo_factor_attribution({"A": np.array([0, 1]), "B": np.array([0, 1])}, 1.0)["worst_contributor"] is None

    def test_freq_pair_attribution(self):
        from mrv.validator.attribution import freq_pair_attribution
        mat = pd.DataFrame(
            [[1.0, 0.8, 0.3, 0.1], [0.8, 1.0, 0.5, 0.2], [0.3, 0.5, 1.0, 0.6], [0.1, 0.2, 0.6, 1.0]],
            index=["5m", "15m", "1h", "1d"], columns=["5m", "15m", "1h", "1d"],
        )
        pairs = freq_pair_attribution(mat)
        assert len(pairs) == 6
        assert pairs[0]["ari"] == pytest.approx(0.1)
        assert pairs[0]["rank"] == 1

    def test_temporal_attribution(self):
        from mrv.validator.attribution import temporal_attribution
        np.random.seed(42)
        idx = pd.date_range("2026-01-05 09:30", periods=500, freq="5min", tz="America/New_York")
        a = pd.Series(np.zeros(500, dtype=int), index=idx)
        b = pd.Series(np.zeros(500, dtype=int), index=idx)
        day2_mask = idx.normalize() != idx[0].normalize()
        b[day2_mask] = np.random.randint(0, 2, day2_mask.sum())
        result = temporal_attribution(a, b, ari_threshold=0.5)
        assert not result.empty
        assert "is_hotspot" in result.columns

    def test_attribution_in_rep_validator(self, tmp_path):
        from mrv.validator.rep import RepValidator
        v = RepValidator({
            "validator": {"report_dir": str(tmp_path), "report_name": "t_{date}",
                          "rep": {"model": "gmm", "n_states": 2, "attribution": True,
                                  "factors": [["vol", "drawdown"], ["var", "cvar"], ["vol", "var", "cvar"]]}},
        })
        result = v.validate(prices={"TEST": make_price_series(500)})
        assert "attribution" in result["assets"]["TEST"]
