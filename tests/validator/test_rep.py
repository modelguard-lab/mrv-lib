"""Tests for mrv.validator.rep (Representation Invariance)."""

from pathlib import Path

import numpy as np
import pytest


class TestRepValidator:
    def test_rep_imported_from_package(self):
        from mrv.validator import RepValidator
        assert RepValidator.name == "rep"

    def test_validate_with_labels(self, tmp_path):
        """Smoke test: validate() runs to completion when labels are supplied directly."""
        from mrv.validator.rep import RepValidator

        rng = np.random.RandomState(0)
        labels_a = rng.randint(0, 2, 300)
        labels_b = rng.randint(0, 2, 300)

        cfg = {
            "validator": {
                "report_dir": str(tmp_path),
                "report_name": "test_{date}",
                "rep": {},
            }
        }
        v = RepValidator(cfg)
        result = v.validate(
            labels={"TEST": {"set_0": labels_a, "set_1": labels_b}},
        )
        assert "assets" in result
        assert "TEST" in result["assets"]
        r = result["assets"]["TEST"]
        assert "mean_ari" in r
        assert "mean_spearman" in r
        assert -1.0 <= r["mean_ari"] <= 1.0
        run_dir = Path(result["run_dir"])
        assert (run_dir / "result.json").exists()
        assert (run_dir / "summary.txt").exists()

    def test_validate_identical_labels_gives_ari_one(self, tmp_path):
        """Identical label sets should yield ARI = 1.0."""
        from mrv.validator.rep import RepValidator

        labels = np.random.RandomState(1).randint(0, 2, 200)

        cfg = {
            "validator": {
                "report_dir": str(tmp_path),
                "report_name": "ari_{date}",
                "rep": {},
            }
        }
        result = RepValidator(cfg).validate(
            labels={"A": {"s0": labels, "s1": labels}},
        )
        assert result["assets"]["A"]["mean_ari"] == pytest.approx(1.0, abs=1e-6)

    def test_validate_with_risk_proxy(self, tmp_path):
        """Ordering consistency should be computed when risk_proxy is provided."""
        from mrv.validator.rep import RepValidator

        rng = np.random.RandomState(42)
        labels_a = rng.randint(0, 3, 200)
        labels_b = rng.randint(0, 3, 200)
        risk = rng.uniform(0, 1, 200)

        cfg = {
            "validator": {
                "report_dir": str(tmp_path),
                "report_name": "t_{date}",
                "rep": {},
            }
        }
        result = RepValidator(cfg).validate(
            labels={"A": {"rep_a": labels_a, "rep_b": labels_b}},
            risk_proxy={"A": risk},
        )
        sp = result["assets"]["A"]["mean_spearman"]
        assert np.isfinite(sp)

    def test_validate_requires_two_specs(self, tmp_path):
        """validate() should raise if fewer than 2 specs are provided."""
        from mrv.validator.rep import RepValidator

        cfg = {
            "validator": {
                "report_dir": str(tmp_path),
                "report_name": "t_{date}",
                "rep": {},
            }
        }
        with pytest.raises(ValueError, match="need >= 2"):
            RepValidator(cfg).validate(labels={"A": {"only_one": np.array([0, 1, 0])}})
