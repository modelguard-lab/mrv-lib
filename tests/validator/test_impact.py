"""Tests for impact_fn (business impact function interface)."""

import numpy as np
import pandas as pd
import pytest

from mrv.validator.base import BaseValidator


class _Dummy(BaseValidator):
    name = "test"
    def validate(self, labels, **kwargs):
        return {}


class TestImpactFn:
    def test_compute_impact_matrix(self):
        def fn(labels, prices):
            return float(np.mean(labels))
        v = _Dummy(cfg={"validator": {}}, impact_fn=fn)
        result = v._compute_impact_matrix(
            {"A": np.array([0, 0, 0, 1, 1]), "B": np.array([1, 1, 1, 1, 1]),
             "C": np.array([0, 0, 0, 0, 0])},
            pd.Series([100.0] * 5),
        )
        assert result is not None
        assert result["impacts"]["B"] == pytest.approx(1.0)
        assert result["impacts"]["C"] == pytest.approx(0.0)
        assert result["max_delta"] == pytest.approx(1.0)
        assert result["delta_matrix"].shape == (3, 3)

    def test_no_impact_fn(self):
        assert _Dummy(cfg={"validator": {}})._compute_impact_matrix({}, pd.Series()) is None

    def test_impact_fn_in_rep_validator(self, tmp_path):
        from mrv.validator.rep import RepValidator
        np.random.seed(42)
        labels_a = np.random.randint(0, 2, 200)
        labels_b = np.random.randint(0, 2, 200)
        prices = pd.Series(np.random.uniform(90, 110, 200))
        v = RepValidator(
            cfg={
                "validator": {"report_dir": str(tmp_path), "report_name": "t_{date}",
                              "rep": {}},
            },
            impact_fn=lambda labels, prices: float(np.std(labels)),
        )
        result = v.validate(
            labels={"TEST": {"rep_a": labels_a, "rep_b": labels_b}},
            prices={"TEST": prices},
        )
        assert "impact" in result["assets"]["TEST"]
