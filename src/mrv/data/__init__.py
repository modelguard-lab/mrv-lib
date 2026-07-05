"""
mrv.data -- Data reading and normalization.

Modules
-------
- reader: Load and validate OHLCV CSVs, resample frequencies
- normalize: Rolling z-score, min-max standardization
"""

from mrv.data.factors import build_factors, log_returns, register_factor
from mrv.data.normalize import minmax, normalize, rolling_zscore
from mrv.data.reader import load_daily, load_ohlcv, resample_ohlc, validate_ohlcv

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
