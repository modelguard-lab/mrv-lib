"""Tests for mrv.data.factors."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_price_series


class TestFactors:
    def test_log_returns(self):
        from mrv.data.factors import log_returns
        p = make_price_series(50)
        r = log_returns(p)
        assert len(r) == 50
        assert pd.isna(r.iloc[0])
        assert not pd.isna(r.iloc[1])

    def test_log_returns_rejects_non_positive(self):
        from mrv.data.factors import log_returns
        p = pd.Series([100.0, 0.0, 50.0])
        with pytest.raises(ValueError, match="Non-positive"):
            log_returns(p)

    def test_volatility(self):
        from mrv.data.factors import log_returns, volatility
        p = make_price_series(100)
        r = log_returns(p)
        v = volatility(r, window=20)
        assert v.name == "volatility"
        assert v.iloc[:20].isna().all()
        assert v.iloc[20:].notna().all()

    def test_drawdown(self):
        from mrv.data.factors import drawdown
        p = make_price_series(100)
        dd = drawdown(p, window=20)
        assert dd.name == "drawdown"
        assert (dd.dropna() <= 0).all()

    def test_var_cvar(self):
        from mrv.data.factors import log_returns, var, cvar
        p = make_price_series(200)
        r = log_returns(p)
        v = var(r, window=60)
        c = cvar(r, window=60)
        assert v.name == "var"
        assert c.name == "cvar"
        valid = pd.concat([v, c], axis=1).dropna()
        assert (valid["cvar"] <= valid["var"]).all()

    def test_build_factors(self):
        from mrv.data.factors import build_factors
        p = make_price_series(300)
        df = build_factors(p, factors=["volatility", "drawdown", "var", "cvar"])
        assert set(df.columns) == {"volatility", "drawdown", "var", "cvar"}
        assert len(df) == 300

    def test_build_factors_with_config(self):
        from mrv.data.factors import build_factors
        p = make_price_series(300)
        cfg = {"factors": {"vol_window": 10, "tail_window": 30}}
        df = build_factors(p, factors=["volatility", "var"], cfg=cfg)
        assert "volatility" in df.columns
        assert df["volatility"].iloc[10:15].notna().all()

    def test_max_drawdown(self):
        from mrv.data.factors import max_drawdown
        p = make_price_series(200)
        mdd = max_drawdown(p, window=60)
        assert mdd.name == "max_drawdown_window"
        assert (mdd.dropna() <= 0).all()

    def test_realized_skew(self):
        from mrv.data.factors import log_returns, realized_skew
        p = make_price_series(200)
        r = log_returns(p)
        sk = realized_skew(r, window=60)
        assert sk.name == "realized_skew"
        assert sk.dropna().shape[0] > 0

    def test_stability(self):
        from mrv.data.factors import log_returns, volatility, stability
        p = make_price_series(200)
        r = log_returns(p)
        v = volatility(r, window=20)
        s = stability(v, window=60)
        assert s.name == "stability"
        assert (s.dropna() >= 0).all()

    def test_resolve_aliases(self):
        from mrv.data.factors import resolve_name
        assert resolve_name("vol") == "volatility"
        assert resolve_name("maxdd") == "max_drawdown_window"
        assert resolve_name("real_skew") == "realized_skew"
        assert resolve_name("vol_stab") == "stability"
        assert resolve_name("unknown") == "unknown"

    def test_register_custom_factor(self):
        from mrv.data.factors import register_factor, build_factors
        def momentum(returns, price, windows):
            return price.pct_change(windows.get("mom_window", 20)).rename("momentum")
        register_factor("momentum", momentum)
        p = make_price_series(100)
        df = build_factors(p, factors=["momentum"])
        assert "momentum" in df.columns

    def test_build_factors_unknown_skipped(self):
        from mrv.data.factors import build_factors
        p = make_price_series(100)
        df = build_factors(p, factors=["volatility", "nonexistent_factor"])
        assert "volatility" in df.columns
        assert len(df.columns) == 1

    def test_build_factors_default(self):
        from mrv.data.factors import build_factors, DEFAULT_FACTORS
        p = make_price_series(300)
        df = build_factors(p)
        assert len(df.columns) == len(DEFAULT_FACTORS)
