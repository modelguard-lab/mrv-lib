"""Tests for impact_fn (business impact function interface)."""

import numpy as np
import pandas as pd
import pytest

from mrv.validator.base import BaseValidator
from tests.conftest import make_price_series, make_ohlcv_5m


class _Dummy(BaseValidator):
    name = "test"
    def validate(self, prices=None, labels=None):
        return {}


class TestImpactFn:
    def test_compute_impact_matrix(self):
        fn = lambda labels, prices: float(np.mean(labels))
        v = _Dummy({"validator": {}}, impact_fn=fn)
        result = v._compute_impact_matrix(
            {"A": np.array([0, 0, 0, 1, 1]), "B": np.array([1, 1, 1, 1, 1]), "C": np.array([0, 0, 0, 0, 0])},
            pd.Series([100.0] * 5),
        )
        assert result is not None
        assert result["impacts"]["B"] == pytest.approx(1.0)
        assert result["impacts"]["C"] == pytest.approx(0.0)
        assert result["max_delta"] == pytest.approx(1.0)
        assert result["delta_matrix"].shape == (3, 3)

    def test_no_impact_fn(self):
        assert _Dummy({"validator": {}})._compute_impact_matrix({}, pd.Series()) is None

    def test_impact_fn_in_rep_validator(self, tmp_path):
        from mrv.validator.rep import RepValidator
        v = RepValidator({
            "validator": {"report_dir": str(tmp_path), "report_name": "t_{date}",
                          "rep": {"model": "gmm", "n_states": 2, "factors": [["vol", "drawdown"], ["var", "cvar"]]}},
        }, impact_fn=lambda labels, prices: float(np.std(labels)))
        result = v.validate(prices={"TEST": make_price_series(200)})
        assert "impact" in result["assets"]["TEST"]

    def test_impact_fn_in_res_validator(self, tmp_path):
        from mrv.validator.res import ResValidator
        v = ResValidator({
            "validator": {"report_dir": str(tmp_path), "report_name": "t_{date}", "res": {"model": "gmm", "n_states": 2}},
        }, impact_fn=lambda labels, prices: float(np.mean(labels)))
        result = v.validate(prices={"TEST": make_ohlcv_5m(15)})
        assert "impact" in result["assets"]["TEST"]
