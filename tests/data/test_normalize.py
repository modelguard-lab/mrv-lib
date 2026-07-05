"""Tests for mrv.data.normalize."""

import numpy as np
import pandas as pd
import pytest


class TestNormalize:
    def test_rolling_zscore(self):
        from mrv.data.normalize import rolling_zscore
        df = pd.DataFrame({"a": np.arange(200, dtype=float), "b": np.arange(200, dtype=float) * 2})
        result = rolling_zscore(df, window=50)
        assert result.shape == df.shape
        assert result.iloc[:49].isna().all().all()
        assert result.iloc[49:].notna().all().all()

    def test_minmax(self):
        from mrv.data.normalize import minmax
        df = pd.DataFrame({"a": np.arange(200, dtype=float)})
        result = minmax(df, window=50)
        valid = result.iloc[50:].dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()

    def test_normalize_dispatch(self):
        from mrv.data.normalize import normalize
        df = pd.DataFrame({"a": np.random.randn(200)})
        r1 = normalize(df, mode="none")
        pd.testing.assert_frame_equal(r1, df)
        r2 = normalize(df, mode="rolling_zscore", window=50)
        assert r2.iloc[50:].notna().all().all()

    def test_normalize_with_config(self):
        from mrv.data.normalize import normalize
        df = pd.DataFrame({"a": np.random.randn(200)})
        cfg = {"normalize": {"mode": "minmax", "window": 50}}
        result = normalize(df, cfg=cfg)
        valid = result.iloc[50:].dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()

    def test_normalize_unknown_mode(self):
        from mrv.data.normalize import normalize
        df = pd.DataFrame({"a": np.random.randn(200)})
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            normalize(df, mode="invalid_mode")

    def test_rolling_zscore_constant_column(self):
        from mrv.data.normalize import rolling_zscore
        df = pd.DataFrame({"a": np.ones(200)})
        result = rolling_zscore(df, window=50)
        assert result.iloc[50:].isna().all().all()
