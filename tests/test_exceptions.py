"""Tests for the mrv exception hierarchy (backward-compatible typed errors).

Every retrofitted public-facing input/config error must:
1. raise the typed mrv error (MrvValidationError / MrvConfigError), and
2. remain catchable as the builtin it replaces (ValueError), so existing
   ``except ValueError`` code keeps working (non-breaking change).
"""

import numpy as np
import pandas as pd
import pytest

import mrv
import mrv.pipeline  # noqa: F401  (registers the mrv.pipeline attribute for access below)
from mrv.exceptions import MrvConfigError, MrvError, MrvValidationError


class TestHierarchy:
    def test_subclassing(self):
        # Typed errors derive from the shared base and from the builtin
        # they replace, so both catch styles work.
        assert issubclass(MrvValidationError, MrvError)
        assert issubclass(MrvValidationError, ValueError)
        assert issubclass(MrvConfigError, MrvError)
        assert issubclass(MrvConfigError, ValueError)
        assert issubclass(MrvError, Exception)

    def test_top_level_exports(self):
        assert mrv.MrvError is MrvError
        assert mrv.MrvValidationError is MrvValidationError
        assert mrv.MrvConfigError is MrvConfigError
        for name in ("MrvError", "MrvValidationError", "MrvConfigError"):
            assert name in mrv.__all__

    def test_invariance_reexports(self):
        from mrv.invariance import MrvConfigError as C
        from mrv.invariance import MrvError as E
        from mrv.invariance import MrvValidationError as V
        assert (E, V, C) == (MrvError, MrvValidationError, MrvConfigError)


class TestValidationErrors:
    def test_res_empty_resolution_set(self):
        with pytest.raises(MrvValidationError, match="empty"):
            mrv.res_invariance_validator(lambda s: s, resolution_set={})

    def test_res_empty_resolution_set_backward_compat(self):
        # Same trigger, caught as the builtin -> non-breaking.
        with pytest.raises(ValueError, match="empty"):
            mrv.res_invariance_validator(lambda s: s, resolution_set={})

    def test_res_single_frequency(self):
        idx = pd.date_range("2026-01-05", periods=60, freq="5min")
        rs = {"SPY": {"5m": pd.Series(np.zeros(60, dtype=int), index=idx)}}
        with pytest.raises(MrvValidationError, match="need >= 2"):
            mrv.res_invariance_validator(lambda s: s, resolution_set=rs)
        with pytest.raises(ValueError, match="need >= 2"):
            mrv.res_invariance_validator(lambda s: s, resolution_set=rs)

    def test_resolution_spec_too_few_freqs(self):
        with pytest.raises(MrvValidationError, match=">= 2"):
            mrv.ResolutionSpec(freqs=("5m",))
        with pytest.raises(ValueError, match=">= 2"):
            mrv.ResolutionSpec(freqs=("5m",))

    def test_resolution_spec_bad_intraday(self):
        with pytest.raises(MrvValidationError, match="not in freqs"):
            mrv.ResolutionSpec(freqs=("5m", "1d"), intraday_freqs=("15m",))
        with pytest.raises(ValueError, match="not in freqs"):
            mrv.ResolutionSpec(freqs=("5m", "1d"), intraday_freqs=("15m",))

    def test_rep_too_few_specs(self):
        with pytest.raises(MrvValidationError, match=">= 2 specifications"):
            mrv.rep_invariance_validator(lambda x: x, admissible_class={"a": np.zeros((10, 2))})
        with pytest.raises(ValueError, match=">= 2 specifications"):
            mrv.rep_invariance_validator(lambda x: x, admissible_class={"a": np.zeros((10, 2))})


class TestConfigErrors:
    def test_unknown_download_source(self):
        from mrv.pipeline import download
        cfg = {"download": {"source": "bogus"}}
        with pytest.raises(MrvConfigError, match="Unknown download source"):
            download(cfg=cfg)
        with pytest.raises(ValueError, match="Unknown download source"):
            download(cfg=cfg)

    def test_invalid_config_not_a_dict(self, tmp_path):
        from mrv.utils.config import load
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(MrvConfigError, match="Invalid config"):
            load(bad)
        with pytest.raises(ValueError, match="Invalid config"):
            load(bad)

    def test_unknown_model(self):
        # Model name is config-derived; unknown -> MrvConfigError (catchable as ValueError).
        from mrv.models import fit
        features = pd.DataFrame({"x": np.arange(10.0)})
        with pytest.raises(MrvConfigError, match="Unknown model"):
            fit(features, model="does_not_exist")
        with pytest.raises(ValueError, match="Unknown model"):
            fit(features, model="does_not_exist")

    def test_unknown_normalization_mode(self):
        from mrv.data.normalize import normalize
        df = pd.DataFrame({"x": np.arange(10.0)})
        with pytest.raises(MrvConfigError, match="Unknown normalization mode"):
            normalize(df, mode="bogus")
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            normalize(df, mode="bogus")


class TestPipelineLabelsFirst:
    """Labels-first backend raises typed errors that stay ValueError-catchable."""

    def test_validate_res_empty_labels(self):
        with pytest.raises(MrvError, match="empty"):
            mrv.pipeline.validate_res(labels={})
        with pytest.raises(ValueError, match="empty"):
            mrv.pipeline.validate_res(labels={})

    def test_validate_rep_empty_labels(self):
        with pytest.raises(MrvError, match="empty"):
            mrv.pipeline.validate_rep(labels={})
        with pytest.raises(ValueError, match="empty"):
            mrv.pipeline.validate_rep(labels={})

    def test_validate_res_typed_as_validation_error(self):
        # Specifically MrvValidationError (bad user data), not just the base.
        with pytest.raises(MrvValidationError):
            mrv.pipeline.validate_res(labels={})

    def test_validate_rep_single_spec(self):
        with pytest.raises(MrvValidationError, match="need >= 2"):
            mrv.pipeline.validate_rep(labels={"SPY": {"only_one": np.zeros(10, dtype=int)}})

    def test_non_positive_prices_build_factors(self):
        from mrv.data.factors import build_factors
        price = pd.Series([1.0, 2.0, -3.0, 4.0])
        with pytest.raises(MrvValidationError, match="non-positive"):
            build_factors(price)
        with pytest.raises(ValueError, match="non-positive"):
            build_factors(price)
