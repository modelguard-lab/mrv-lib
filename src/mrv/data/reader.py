"""
mrv.data.reader — Load and validate OHLCV market data from CSV.

Supports both daily data (Paper 1 / yfinance style) and 5-minute intraday
data (Paper 2 / IB style).  Also provides OHLC resampling from 5m to
coarser frequencies (15m, 1h, 1d).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price column inference
# ---------------------------------------------------------------------------

def _infer_price_column(df: pd.DataFrame) -> str:
    """Find the best price column (Adj Close > Close > close > price)."""
    for candidate in ("Adj Close", "Close", "close", "price"):
        if candidate in df.columns:
            return candidate
    lowered = {col.lower(): col for col in df.columns}
    for token in ("adj close", "close", "price", "last"):
        for key, original in lowered.items():
            if token in key:
                return original
    raise ValueError(f"Could not infer price column from: {df.columns.tolist()}")


# ---------------------------------------------------------------------------
# Daily data (Paper 1 / yfinance)
# ---------------------------------------------------------------------------

def load_daily(path: str | Path, price_col: Optional[str] = None) -> pd.Series:
    """
    Load a daily price series from CSV.

    Returns a ``pd.Series`` indexed by date with the inferred (or specified)
    price column.
    """
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    col = price_col or _infer_price_column(df)
    return df[col].rename(path.stem)


# ---------------------------------------------------------------------------
# Intraday data (Paper 2 / IB 5m)
# ---------------------------------------------------------------------------

def load_ohlcv(
    path: str | Path,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV with a DatetimeIndex.

    Works for both daily and intraday CSVs.  For intraday data the index
    is localized/converted to *tz*.
    """
    path = Path(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Normalize columns if MultiIndex (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    idx = pd.to_datetime(df.index, errors="coerce")
    bad = idx.isna()
    if bad.any():
        df = df.loc[~bad].copy()
        idx = idx[~bad]
    df.index = pd.DatetimeIndex(idx)

    # Localize timezone for intraday data (has time component)
    has_time = not all(df.index == df.index.normalize())
    if has_time:
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz, ambiguous="infer")
        else:
            df.index = df.index.tz_convert(tz)

    return df.sort_index()


def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> List[str]:
    """
    Check an OHLCV DataFrame for common data quality issues.

    Returns a list of issue strings (empty list = OK).
    """
    issues: List[str] = []
    prefix = f"{symbol}: " if symbol else ""
    required = ["Open", "High", "Low", "Close"]

    for col in required:
        if col not in df.columns:
            # Try lowercase
            if col.lower() in df.columns:
                continue
            issues.append(f"{prefix}missing column '{col}'")
    if issues:
        return issues
    if df.empty:
        issues.append(f"{prefix}empty DataFrame")
        return issues

    # Use actual column names (case-insensitive lookup)
    def _col(name: str) -> str:
        return name if name in df.columns else name.lower()

    h, l, o, c = _col("High"), _col("Low"), _col("Open"), _col("Close")

    nan_counts = df[[o, h, l, c]].isna().sum()
    if nan_counts.any():
        issues.append(f"{prefix}NaN in {nan_counts[nan_counts > 0].to_dict()}")
    if (df[h] < df[l]).sum() > 0:
        issues.append(f"{prefix}High < Low in {(df[h] < df[l]).sum()} bars")
    if not df.index.is_monotonic_increasing:
        issues.append(f"{prefix}index not sorted ascending")
    dup = df.index.duplicated().sum()
    if dup > 0:
        issues.append(f"{prefix}{dup} duplicate timestamps")

    return issues


# ---------------------------------------------------------------------------
# OHLC resampling (5m -> 15m / 1h / 1d)
# ---------------------------------------------------------------------------

def resample_ohlc(df: pd.DataFrame, freq: str, tz: str = "America/New_York") -> pd.DataFrame:
    """
    Resample OHLCV data to a coarser frequency.

    Parameters
    ----------
    df : DataFrame
        Source OHLCV (expects Open, High, Low, Close columns; Volume optional).
    freq : str
        Target frequency: ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``.
    tz : str
        Timezone for daily grouping.
    """
    if freq == "5m":
        return df.copy()

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    has_volume = "Volume" in df.columns

    if freq == "15m":
        res = df.resample("15min", label="right", closed="right").agg(agg).dropna(how="all")
        if has_volume:
            res["Volume"] = df["Volume"].resample("15min", label="right", closed="right").sum()
        return res

    if freq == "1h":
        res = df.resample("1h", label="right", closed="right").agg(agg).dropna(how="all")
        if has_volume:
            res["Volume"] = df["Volume"].resample("1h", label="right", closed="right").sum()
        return res

    if freq == "1d":
        g = df.groupby(df.index.tz_convert(tz).normalize())
        res = g.agg(agg)
        if has_volume:
            res["Volume"] = g["Volume"].sum()
        if res.index.tz is None:
            res.index = res.index.tz_localize(tz)
        else:
            res.index = res.index.tz_convert(tz)
        res.index = res.index + pd.Timedelta(hours=16)
        return res

    raise ValueError(f"Unsupported frequency: {freq}")
