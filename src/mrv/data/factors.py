"""
mrv.data.factors — Risk factor computation, registry, and builder.

Built-in factors are registered automatically. Users can add custom factors
via ``register_factor()``.

Usage::

    from mrv.data.factors import build_factors, register_factor

    df = build_factors(price, factors=["volatility", "var", "cvar"])

    # Custom factor
    def momentum(returns, price, windows):
        return price.pct_change(windows.get("mom_window", 20)).rename("momentum")
    register_factor("momentum", momentum)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual factor functions
# ---------------------------------------------------------------------------

def log_returns(price: pd.Series) -> pd.Series:
    """Compute log returns. Raises ValueError on non-positive prices."""
    if (price <= 0).any():
        n_bad = int((price <= 0).sum())
        raise ValueError(f"Non-positive prices detected ({n_bad} values).")
    return np.log(price / price.shift(1))


def volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol.rename("volatility")


def drawdown(price: pd.Series, window: int = 60) -> pd.Series:
    rolling_max = price.rolling(window=window, min_periods=window).max()
    return (price / rolling_max - 1.0).rename("drawdown")


def max_drawdown(price: pd.Series, window: int = 60) -> pd.Series:
    def _mdd(x: np.ndarray) -> float:
        if x.size == 0:
            return np.nan
        running_max = np.maximum.accumulate(x)
        return float(np.min(x / running_max - 1.0))
    return price.rolling(window=window, min_periods=window).apply(_mdd, raw=True).rename("max_drawdown_window")


def var(returns: pd.Series, window: int = 60, alpha: float = 0.05) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).quantile(alpha).rename("var")


def cvar(returns: pd.Series, window: int = 60, alpha: float = 0.05) -> pd.Series:
    def _cvar(x: np.ndarray) -> float:
        if x.size == 0:
            return np.nan
        cutoff = np.quantile(x, alpha)
        tail = x[x <= cutoff]
        return float(np.mean(tail)) if tail.size else np.nan
    return returns.rolling(window=window, min_periods=window).apply(_cvar, raw=True).rename("cvar")


def realized_skew(returns: pd.Series, window: int = 60) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).skew().rename("realized_skew")


def stability(vol_series: pd.Series, window: int = 60) -> pd.Series:
    return vol_series.rolling(window=window, min_periods=window).std().rename("stability")


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

FactorFn = Callable[[pd.Series, pd.Series, Dict[str, Any]], pd.Series]
_REGISTRY: Dict[str, FactorFn] = {}


def register_factor(name: str, fn: FactorFn) -> None:
    """Register a factor: ``(returns, price, windows_dict) -> Series``."""
    _REGISTRY[name] = fn


def resolve_name(name: str) -> str:
    """Resolve short alias to canonical name."""
    return _ALIASES.get(name, name)


# Built-in registrations
register_factor("volatility", lambda r, p, w: volatility(r, w.get("vol_window", 20)))
register_factor("drawdown", lambda r, p, w: drawdown(p, w.get("drawdown_window", 60)))
register_factor("max_drawdown_window", lambda r, p, w: max_drawdown(p, w.get("drawdown_window", 60)))
register_factor("var", lambda r, p, w: var(r, w.get("tail_window", 60), w.get("tail_alpha", 0.05)))
register_factor("cvar", lambda r, p, w: cvar(r, w.get("tail_window", 60), w.get("tail_alpha", 0.05)))
register_factor("realized_skew", lambda r, p, w: realized_skew(r, w.get("skew_window", 60)))
register_factor("stability", lambda r, p, w: stability(
    volatility(r, w.get("vol_window", 20)), w.get("stability_window", 60)
))

DEFAULT_FACTORS = ["volatility", "drawdown", "max_drawdown_window", "var", "cvar"]

_ALIASES = {
    "vol": "volatility",
    "maxdd": "max_drawdown_window",
    "real_skew": "realized_skew",
    "vol_stab": "stability",
}


# ---------------------------------------------------------------------------
# Build factor matrix
# ---------------------------------------------------------------------------

def build_factors(
    price: pd.Series,
    factors: Optional[List[str]] = None,
    windows: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Build a factor matrix from a price series."""
    factor_names = [resolve_name(f) for f in (factors or DEFAULT_FACTORS)]
    base_windows = dict((cfg or {}).get("factors", {}))
    if windows:
        base_windows.update(windows)

    r = log_returns(price)
    parts: List[pd.Series] = []
    for name in factor_names:
        builder = _REGISTRY.get(name)
        if builder is None:
            logger.warning("Unknown factor '%s', skipping", name)
            continue
        parts.append(builder(r, price, base_windows))

    if not parts:
        return pd.DataFrame(index=price.index)
    return pd.concat(parts, axis=1)


