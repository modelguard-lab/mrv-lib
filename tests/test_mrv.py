"""Tests for mrv-lib: factors, normalize, reader, config, log, download."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_series(n: int = 300) -> pd.Series:
    """Synthetic daily price series for testing."""
    np.random.seed(42)
    returns = np.random.randn(n) * 0.01
    price = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(price, index=idx, name="close")


def _ohlcv_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 0.3
    low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 0.3
    vol = np.random.randint(1000, 10000, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------

class TestFactors:
    def test_log_returns(self):
        from mrv.data.factors import log_returns
        p = _price_series(50)
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
        p = _price_series(100)
        r = log_returns(p)
        v = volatility(r, window=20)
        assert v.name == "volatility"
        # First 20 values are NaN (rolling window)
        assert v.iloc[:20].isna().all()
        assert v.iloc[20:].notna().all()

    def test_drawdown(self):
        from mrv.data.factors import drawdown
        p = _price_series(100)
        dd = drawdown(p, window=20)
        assert dd.name == "drawdown"
        # Drawdown should be <= 0
        valid = dd.dropna()
        assert (valid <= 0).all()

    def test_var_cvar(self):
        from mrv.data.factors import log_returns, var, cvar
        p = _price_series(200)
        r = log_returns(p)
        v = var(r, window=60)
        c = cvar(r, window=60)
        assert v.name == "var"
        assert c.name == "cvar"
        # CVaR should be <= VaR (both are negative)
        valid = pd.concat([v, c], axis=1).dropna()
        assert (valid["cvar"] <= valid["var"]).all()

    def test_build_factors(self):
        from mrv.data.factors import build_factors
        p = _price_series(300)
        df = build_factors(p, factors=["volatility", "drawdown", "var", "cvar"])
        assert set(df.columns) == {"volatility", "drawdown", "var", "cvar"}
        assert len(df) == 300

    def test_build_factors_with_config(self):
        from mrv.data.factors import build_factors
        p = _price_series(300)
        cfg = {"factors": {"vol_window": 10, "tail_window": 30}}
        df = build_factors(p, factors=["volatility", "var"], cfg=cfg)
        assert "volatility" in df.columns
        # With window=10, should have values earlier
        assert df["volatility"].iloc[10:15].notna().all()


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_rolling_zscore(self):
        from mrv.data.normalize import rolling_zscore
        df = pd.DataFrame({"a": np.arange(200, dtype=float), "b": np.arange(200, dtype=float) * 2})
        result = rolling_zscore(df, window=50)
        assert result.shape == df.shape
        # First 49 rows NaN (min_periods=50 means index 49 is the first valid)
        assert result.iloc[:49].isna().all().all()
        # After warmup, should be normalized
        valid = result.iloc[49:]
        assert valid.notna().all().all()

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
        # "none" returns copy
        r1 = normalize(df, mode="none")
        pd.testing.assert_frame_equal(r1, df)
        # "rolling_zscore"
        r2 = normalize(df, mode="rolling_zscore", window=50)
        assert r2.iloc[50:].notna().all().all()


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

class TestReader:
    def test_validate_ohlcv_good(self):
        from mrv.data.reader import validate_ohlcv
        df = _ohlcv_df(100)
        issues = validate_ohlcv(df, "TEST")
        assert issues == []

    def test_validate_ohlcv_bad_hl(self):
        from mrv.data.reader import validate_ohlcv
        df = _ohlcv_df(100)
        df.loc[df.index[5], "High"] = df.loc[df.index[5], "Low"] - 1
        issues = validate_ohlcv(df, "TEST")
        assert any("High < Low" in i for i in issues)

    def test_resample_ohlc_passthrough(self):
        from mrv.data.reader import resample_ohlc
        df = _ohlcv_df(50)
        # Fake intraday index with tz
        df.index = pd.date_range("2020-01-01", periods=50, freq="5min", tz="America/New_York")
        result = resample_ohlc(df, "5m")
        assert len(result) == 50

    def test_load_ohlcv_csv(self, tmp_path):
        from mrv.data.reader import load_ohlcv
        df = _ohlcv_df(50)
        df.index.name = "Date"
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path)
        loaded = load_ohlcv(csv_path)
        assert len(loaded) == 50
        assert "Open" in loaded.columns


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_defaults(self):
        from mrv.utils.config import load
        cfg = load()
        assert "download" in cfg
        assert "symbols" in cfg["download"]
        assert "data_dir" in cfg["download"]
        assert "ib" in cfg["download"]
        assert "logging" in cfg
        assert "factors" in cfg
        assert "normalize" in cfg
        assert cfg["download"]["ib"]["port"] == 4002

    def test_get_assets_expands_symbols(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {
            "symbols": ["SPY", "IEF", "GLD"],
            "freq": ["5m", "1h", "1d"],
            "start": "2026-01-01",
            "end": None,
        }}
        all_assets = get_assets(cfg)
        assert len(all_assets) == 3
        assert all_assets[0]["symbol"] == "SPY"
        assert all_assets[0]["freq"] == ["5m", "1h", "1d"]
        assert all_assets[0]["start"] == "2026-01-01"

    def test_get_assets_filter_by_freq(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {
            "symbols": ["SPY", "IEF"],
            "freq": ["5m", "1d"],
            "start": "2026-01-01",
        }}
        intraday = get_assets(cfg, freq="5m")
        assert len(intraday) == 2
        daily = get_assets(cfg, freq="1d")
        assert len(daily) == 2
        none = get_assets(cfg, freq="15m")
        assert len(none) == 0

    def test_get_assets_normalizes_scalar_freq(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {
            "symbols": ["GLD"],
            "freq": "1d",
        }}
        assets = get_assets(cfg)
        assert assets[0]["freq"] == ["1d"]

    def test_load_missing_raises(self, tmp_path):
        from mrv.utils.config import load
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------

class TestLog:
    def test_setup_no_error(self):
        from mrv.utils.log import setup
        cfg = {"logging": {"level": "WARNING", "log_dir": None, "quiet": []}}
        setup(cfg)  # Should not raise

    def test_setup_with_file_logging(self, tmp_path):
        from mrv.utils.log import setup
        import logging
        log_dir = tmp_path / "test_logs"
        cfg = {"logging": {"level": "DEBUG", "log_dir": str(log_dir), "quiet": []}}
        # Reset root handlers for clean test
        root = logging.getLogger()
        root.handlers.clear()
        setup(cfg)
        assert log_dir.exists()
        log_files = list(log_dir.glob("mrv_*.log"))
        assert len(log_files) == 1
        # Cleanup
        root.handlers.clear()


# ---------------------------------------------------------------------------
# Additional factor tests
# ---------------------------------------------------------------------------

class TestFactorsExtended:
    def test_max_drawdown(self):
        from mrv.data.factors import max_drawdown
        p = _price_series(200)
        mdd = max_drawdown(p, window=60)
        assert mdd.name == "max_drawdown_window"
        valid = mdd.dropna()
        assert (valid <= 0).all()

    def test_realized_skew(self):
        from mrv.data.factors import log_returns, realized_skew
        p = _price_series(200)
        r = log_returns(p)
        sk = realized_skew(r, window=60)
        assert sk.name == "realized_skew"
        assert sk.dropna().shape[0] > 0

    def test_stability(self):
        from mrv.data.factors import log_returns, volatility, stability
        p = _price_series(200)
        r = log_returns(p)
        v = volatility(r, window=20)
        s = stability(v, window=60)
        assert s.name == "stability"
        valid = s.dropna()
        assert (valid >= 0).all()



# ---------------------------------------------------------------------------
# canonical_stem
# ---------------------------------------------------------------------------

class TestCanonicalStem:
    def test_basic(self):
        from mrv.utils.download import canonical_stem
        assert canonical_stem("SPY") == "SPY"
        assert canonical_stem("^GSPC") == "GSPC"
        assert canonical_stem("CL=F") == "CL"
        assert canonical_stem("USDJPY") == "USDJPY"


# ---------------------------------------------------------------------------
# reader: load_daily
# ---------------------------------------------------------------------------

class TestLoadDaily:
    def test_load_daily_csv(self, tmp_path):
        from mrv.data.reader import load_daily
        df = _ohlcv_df(50)
        df.index.name = "Date"
        csv_path = tmp_path / "test_daily.csv"
        df.to_csv(csv_path)
        price = load_daily(csv_path)
        assert len(price) == 50
        assert price.name == "test_daily"
