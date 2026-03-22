"""
mrv.data — Data reading and normalization.

Modules
-------
- reader: Load and validate OHLCV CSVs, resample frequencies
- normalize: Rolling z-score, min-max standardization
"""

from mrv.data.reader import load_daily, load_ohlcv, resample_ohlc, validate_ohlcv
from mrv.data.normalize import normalize, rolling_zscore, minmax
from mrv.data.factors import build_factors, register_factor, log_returns

__all__ = [
    "load_daily",
    "load_ohlcv",
    "resample_ohlc",
    "validate_ohlcv",
    "normalize",
    "rolling_zscore",
    "minmax",
    "build_factors",
    "register_factor",
    "log_returns",
]
