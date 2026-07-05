"""Shared fixtures and helpers for mrv-lib tests."""

import importlib

import numpy as np
import pandas as pd
import pytest


def has_hmmlearn() -> bool:
    return importlib.util.find_spec("hmmlearn") is not None


@pytest.fixture
def price_series():
    """Synthetic daily price series (300 obs)."""
    return make_price_series(300)


@pytest.fixture
def ohlcv_df():
    """Synthetic daily OHLCV DataFrame (300 obs)."""
    return make_ohlcv_df(300)


@pytest.fixture
def ohlcv_5m():
    """Synthetic 5m OHLCV DataFrame (20 bdays × 78 bars)."""
    return make_ohlcv_5m(20)


# ---------------------------------------------------------------------------
# Factory functions (also importable directly)
# ---------------------------------------------------------------------------

def make_price_series(n: int = 300) -> pd.Series:
    np.random.seed(42)
    returns = np.random.randn(n) * 0.01
    price = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(price, index=idx, name="close")


def make_ohlcv_df(n: int = 300) -> pd.DataFrame:
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


def make_ohlcv_5m(n_days: int = 20, bars_per_day: int = 78) -> pd.DataFrame:
    np.random.seed(42)
    bdays = pd.bdate_range("2026-01-05", periods=n_days, tz="America/New_York")
    dates = []
    for day in bdays:
        for b in range(bars_per_day):
            dates.append(day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * b))
    idx = pd.DatetimeIndex(sorted(dates))
    n = len(idx)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 0.05
    low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 0.05
    volume = np.random.randint(100, 5000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
