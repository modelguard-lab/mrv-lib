"""Tests for mrv.data.reader."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv_df


class TestReader:
    def test_validate_ohlcv_good(self):
        from mrv.data.reader import validate_ohlcv
        issues = validate_ohlcv(make_ohlcv_df(100), "TEST")
        assert issues == []

    def test_validate_ohlcv_bad_hl(self):
        from mrv.data.reader import validate_ohlcv
        df = make_ohlcv_df(100)
        df.loc[df.index[5], "High"] = df.loc[df.index[5], "Low"] - 1
        assert any("High < Low" in i for i in validate_ohlcv(df, "TEST"))

    def test_validate_ohlcv_missing_column(self):
        from mrv.data.reader import validate_ohlcv
        df = pd.DataFrame({"Open": [1], "High": [2], "Low": [1]})
        assert any("Close" in i for i in validate_ohlcv(df))

    def test_validate_ohlcv_empty(self):
        from mrv.data.reader import validate_ohlcv
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        assert any("empty" in i for i in validate_ohlcv(df))

    def test_validate_ohlcv_duplicate_timestamps(self):
        from mrv.data.reader import validate_ohlcv
        df = make_ohlcv_df(10)
        df.index = [df.index[0]] * 10
        assert any("duplicate" in i for i in validate_ohlcv(df))

    def test_resample_ohlc_passthrough(self):
        from mrv.data.reader import resample_ohlc
        df = make_ohlcv_df(50)
        df.index = pd.date_range("2020-01-01", periods=50, freq="5min", tz="America/New_York")
        assert len(resample_ohlc(df, "5m")) == 50

    def test_resample_ohlc_15m(self):
        from mrv.data.reader import resample_ohlc
        df = make_ohlcv_df(60)
        df.index = pd.date_range("2020-01-02 09:30", periods=60, freq="5min", tz="America/New_York")
        result = resample_ohlc(df, "15m")
        assert 0 < len(result) < 60

    def test_resample_ohlc_1h(self):
        from mrv.data.reader import resample_ohlc
        df = make_ohlcv_df(60)
        df.index = pd.date_range("2020-01-02 09:30", periods=60, freq="5min", tz="America/New_York")
        assert 0 < len(resample_ohlc(df, "1h")) < 60

    def test_resample_ohlc_invalid_freq(self):
        from mrv.data.reader import resample_ohlc
        df = make_ohlcv_df(10)
        df.index = pd.date_range("2020-01-01", periods=10, freq="5min", tz="America/New_York")
        with pytest.raises(ValueError, match="Unsupported frequency"):
            resample_ohlc(df, "3m")

    def test_load_ohlcv_csv(self, tmp_path):
        from mrv.data.reader import load_ohlcv
        df = make_ohlcv_df(50)
        df.index.name = "Date"
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path)
        loaded = load_ohlcv(csv_path)
        assert len(loaded) == 50
        assert "Open" in loaded.columns

    def test_load_daily_csv(self, tmp_path):
        from mrv.data.reader import load_daily
        df = make_ohlcv_df(50)
        df.index.name = "Date"
        csv_path = tmp_path / "test_daily.csv"
        df.to_csv(csv_path)
        price = load_daily(csv_path)
        assert len(price) == 50
        assert price.name == "test_daily"

    def test_infer_price_column(self):
        from mrv.data.reader import _infer_price_column
        assert _infer_price_column(pd.DataFrame({"Adj Close": [1], "Open": [1]})) == "Adj Close"
        assert _infer_price_column(pd.DataFrame({"close": [1], "open": [1]})) == "close"

    def test_infer_price_column_raises(self):
        from mrv.data.reader import _infer_price_column
        with pytest.raises(ValueError, match="Could not infer"):
            _infer_price_column(pd.DataFrame({"volume": [1], "open": [1]}))
