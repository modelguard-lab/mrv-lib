"""Tests for mrv_lib.core: Scanner, detect_boundary, ordinal_consistency."""

import numpy as np
import pandas as pd
import pytest

from mrv_lib import (
    BoundaryResult,
    RepresentationTestResult,
    Scanner,
    detect_boundary,
    ordinal_consistency,
)


def _ohlcv_df(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n)) + 100.0
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 0.5
    low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 0.5
    volume = np.random.randint(1000, 10000, n)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_scanner_run_representation_test_returns_result():
    data = _ohlcv_df(50)
    scanner = Scanner(resolution=["1h", "1d"])
    result = scanner.run_representation_test(data, model="HMM")
    assert isinstance(result, RepresentationTestResult)
    assert 0 <= result.rss_score <= 1
    assert "1h" in result.resolution_scores
    assert "1d" in result.resolution_scores


def test_detect_boundary_returns_boundary_result():
    data = _ohlcv_df(50)
    boundary = detect_boundary(data)
    assert isinstance(boundary, BoundaryResult)
    assert 0 <= boundary.index <= 1
    assert boundary.is_collapsed == (boundary.index < boundary.threshold)


def test_detect_boundary_short_series_collapsed():
    data = _ohlcv_df(3)
    boundary = detect_boundary(data)
    assert boundary.is_collapsed is True
    assert boundary.index == 0.0


def test_ordinal_consistency_perfect_correlation():
    y = np.array([1, 2, 3, 4, 5])
    assert ordinal_consistency(y, y) == pytest.approx(1.0)


def test_ordinal_consistency_reverse_correlation():
    y = np.array([1, 2, 3, 4, 5])
    assert ordinal_consistency(y, -y) == pytest.approx(-1.0)


def test_ordinal_consistency_short_returns_zero():
    assert ordinal_consistency([1], [2]) == 0.0
