"""Tests for mrv.pipeline."""

import numpy as np
import pandas as pd
import pytest


class TestPipeline:
    def test_validators_registered(self):
        from mrv.pipeline import _VALIDATORS
        assert "rep" in _VALIDATORS
        assert "res" in _VALIDATORS

    def test_validate_rep_exposed(self):
        from mrv.pipeline import validate_rep
        assert callable(validate_rep)

    def test_validate_res_exposed(self):
        from mrv.pipeline import validate_res
        assert callable(validate_res)

    def test_report_exposed(self):
        from mrv.pipeline import report
        assert callable(report)

    def test_sr_report_symbols_removed(self):
        """The SR-26-2 / SR-11-7 governance-report symbols are gone (v0.6.1)."""
        import mrv.pipeline as pipeline
        for name in ("sr26_2_report", "sr11_7_report"):
            assert not hasattr(pipeline, name), f"{name} should be removed"

    def test_validate_rep_runs_end_to_end(self, tmp_path):
        """validate_rep must call through without raising and return a result dict."""
        from mrv.pipeline import validate_rep

        rng = np.random.default_rng(42)
        n = 200
        labels_a = rng.integers(0, 2, size=n)
        labels_b = rng.integers(0, 2, size=n)
        result = validate_rep(
            labels={"SPY": {"spec_a": labels_a, "spec_b": labels_b}},
            cfg={"validator": {"rep": {}, "report_dir": str(tmp_path)}},
        )
        assert "assets" in result
        assert "SPY" in result["assets"]
        asset = result["assets"]["SPY"]
        assert "mean_ari" in asset
        assert isinstance(asset["mean_ari"], float)
        assert "n_obs" in asset
        assert asset["n_obs"] == n
        assert "n_specs" in asset  # n_factor_sets alias is in the JSON; n_specs in live result
        # Verify run_dir is under tmp_path (isolation: no leak to working-tree reports/).
        from pathlib import Path
        assert tmp_path in Path(result["run_dir"]).parents, (
            f"run_dir {result['run_dir']!r} is not under tmp_path {tmp_path}"
        )

    def test_validate_res_runs_end_to_end(self, tmp_path):
        """validate_res must call through without raising and return a result dict."""
        from mrv.pipeline import validate_res

        rng = np.random.default_rng(42)
        dates_1d = pd.bdate_range("2022-01-03", periods=200)
        labels_1d = pd.Series(rng.integers(0, 2, size=200), index=dates_1d)
        labels_2d = pd.Series(rng.integers(0, 2, size=200), index=dates_1d)

        result = validate_res(
            labels={"SPY": {"1d": labels_1d, "2d": labels_2d}},
            cfg={"validator": {"res": {}, "report_dir": str(tmp_path)}},
        )
        assert "assets" in result
        assert "SPY" in result["assets"]

    def test_validate_function_callable_and_invocable(self, tmp_path):
        """validate() wrapper used by monitor() must exist and not raise on an unknown validator."""
        import pytest

        from mrv.pipeline import validate

        assert callable(validate)
        cfg = {"validator": {"rep": {}, "report_dir": str(tmp_path)}}
        # validate() dispatches on the validator string; an unknown name raises ValueError.
        # This confirms the body executes (not just the import) so a future
        # signature breakage is caught here rather than silently at monitor() runtime.
        with pytest.raises(ValueError, match="Unknown validator"):
            validate(cfg=cfg, validator="__nonexistent__")


class TestPipelineErrors:
    def test_validate_rep_empty_labels_raises(self):
        from mrv.pipeline import validate_rep
        with pytest.raises(ValueError, match="empty"):
            validate_rep(labels={})

    def test_validate_res_empty_labels_raises(self):
        from mrv.pipeline import validate_res
        with pytest.raises(ValueError, match="empty"):
            validate_res(labels={})

    def test_validate_res_malformed_event_window_raises(self, tmp_path):
        from mrv.pipeline import validate_res
        idx = pd.date_range("2026-01-05 09:30", periods=120, freq="5min",
                            tz="America/New_York")
        labels = {
            "SPY": {
                "5m": pd.Series(np.zeros(120, dtype=int), index=idx),
                "15m": pd.Series(np.ones(120, dtype=int), index=idx),
            }
        }
        cfg = {
            "validator": {
                "report_dir": str(tmp_path),
                "res": {"event_window": "not-a-2-element-list"},
            }
        }
        with pytest.raises(ValueError, match="event_window must be a 2-element"):
            validate_res(labels=labels, cfg=cfg)
