"""
mrv.data.factors -- Risk factor computation, registry, and builder.

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

from mrv.exceptions import MrvValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "log_returns",
    "volatility",
    "drawdown",
    "max_drawdown",
    "var",
    "cvar",
    "realized_skew",
    "stability",
    "build_factors",
    "register_factor",
]


# ---------------------------------------------------------------------------
# Individual factor functions
# ---------------------------------------------------------------------------

def log_returns(price: pd.Series) -> pd.Series:
    """Compute log returns. Raises ValueError on non-positive prices."""
    if (price <= 0).any():
        n_bad = int((price <= 0).sum())
        raise MrvValidationError(f"Non-positive prices detected ({n_bad} values).")
    return np.log(price / price.shift(1))


def volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling realised volatility.

    Parameters
    ----------
    returns : pd.Series
        Log-return series.
    window : int, default 20
        Rolling window in observations.
    annualize : bool, default True
        When True, multiply by ``sqrt(252)`` to annualise.

    Returns
    -------
    pd.Series
        Annualised (or raw) rolling standard deviation, named ``"volatility"``.
    """
    vol = returns.rolling(window=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol.rename("volatility")


def drawdown(price: pd.Series, window: int = 60) -> pd.Series:
    """Rolling drawdown from rolling high-water mark.

    Parameters
    ----------
    price : pd.Series
        Price series (must be strictly positive).
    window : int, default 60
        Rolling window in observations.

    Returns
    -------
    pd.Series
        Rolling drawdown in (-inf, 0], named ``"drawdown"``.
    """
    rolling_max = price.rolling(window=window, min_periods=window).max()
    return (price / rolling_max - 1.0).rename("drawdown")


def max_drawdown(price: pd.Series, window: int = 60) -> pd.Series:
    """Rolling max drawdown -- vectorized using cummax within each window."""
    vals = price.values
    n = len(vals)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = vals[i - window + 1: i + 1]
        running_max = np.maximum.accumulate(w)
        if running_max[0] <= 0:
            continue  # skip window if first price is non-positive
        result[i] = float(np.min(w / running_max - 1.0))
    return pd.Series(result, index=price.index, name="max_drawdown_window")


def var(returns: pd.Series, window: int = 60, alpha: float = 0.05) -> pd.Series:
    """Rolling Value-at-Risk (historical simulation).

    Parameters
    ----------
    returns : pd.Series
        Log-return series.
    window : int, default 60
        Rolling window in observations.
    alpha : float, default 0.05
        Tail quantile level (e.g. 0.05 for 5th percentile).

    Returns
    -------
    pd.Series
        Rolling *alpha* quantile of returns, named ``"var"``.
    """
    return returns.rolling(window=window, min_periods=window).quantile(alpha).rename("var")


def cvar(returns: pd.Series, window: int = 60, alpha: float = 0.05) -> pd.Series:
    """Rolling CVaR -- vectorized: use VaR as cutoff, then mean of tail."""
    var_series = returns.rolling(window=window, min_periods=window).quantile(alpha)
    vals = returns.values
    var_vals = var_series.values
    n = len(vals)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        cutoff = var_vals[i]
        if np.isnan(cutoff):
            continue
        w = vals[i - window + 1: i + 1]
        tail = w[w <= cutoff]
        result[i] = float(np.mean(tail)) if tail.size else np.nan
    return pd.Series(result, index=returns.index, name="cvar")


def realized_skew(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling realised skewness of log-returns.

    Parameters
    ----------
    returns : pd.Series
        Log-return series.
    window : int, default 60
        Rolling window in observations.

    Returns
    -------
    pd.Series
        Rolling skewness, named ``"realized_skew"``.
    """
    return returns.rolling(window=window, min_periods=window).skew().rename("realized_skew")


def stability(vol_series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling volatility-of-volatility (vol stability).

    Parameters
    ----------
    vol_series : pd.Series
        Realised volatility series (output of :func:`volatility`).
    window : int, default 60
        Rolling window in observations.

    Returns
    -------
    pd.Series
        Rolling standard deviation of ``vol_series``, named ``"stability"``.
    """
    return vol_series.rolling(window=window, min_periods=window).std().rename("stability")


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

FactorFn = Callable[[pd.Series, pd.Series, Dict[str, Any]], pd.Series]
_REGISTRY: Dict[str, FactorFn] = {}


def register_factor(name: str, fn: FactorFn) -> None:
    """Register a custom factor function under *name*.

    The factor function signature must be::

        fn(returns: pd.Series, price: pd.Series, windows: dict) -> pd.Series

    Parameters
    ----------
    name : str
        Canonical factor name (e.g. ``"momentum"``). Overwrites any existing
        registration with the same name.
    fn : callable
        Factor builder function. The third argument ``windows`` is the merged
        windows dict from ``build_factors``.

    Examples
    --------
    >>> def momentum(returns, price, windows):
    ...     return price.pct_change(windows.get("mom_window", 20)).rename("momentum")
    >>> register_factor("momentum", momentum)
    """
    _REGISTRY[name] = fn


def resolve_name(name: str) -> str:
    """Resolve short alias to canonical name."""
    return _ALIASES.get(name, name)


# Built-in registrations
register_factor("volatility", lambda r, p, w: volatility(r, w.get("vol_window", 20)))
register_factor("drawdown", lambda r, p, w: drawdown(p, w.get("drawdown_window", 60)))
register_factor(
    "max_drawdown_window",
    lambda r, p, w: max_drawdown(p, w.get("drawdown_window", 60)),
)
register_factor(
    "var", lambda r, p, w: var(r, w.get("tail_window", 60), w.get("tail_alpha", 0.05))
)
register_factor(
    "cvar", lambda r, p, w: cvar(r, w.get("tail_window", 60), w.get("tail_alpha", 0.05))
)
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
    """Build a feature matrix from a price series using registered factors.

    Parameters
    ----------
    price : pd.Series
        Daily (or intraday) price series. Must be strictly positive.
    factors : list of str, optional
        Factor names to compute. Each name must be registered via
        :func:`register_factor` or be a built-in alias. Defaults to
        ``DEFAULT_FACTORS`` when *None*.
    windows : dict, optional
        Override window sizes (e.g. ``{"vol_window": 30}``). Merged with
        any ``cfg["factors"]`` settings; caller-supplied values take
        precedence.
    cfg : dict, optional
        Full mrv config dict. ``cfg["factors"]`` is used as the base
        window dict if provided.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to ``price.index``. Columns correspond to
        the requested factors; rows at the beginning contain NaN due to
        rolling windows.

    Raises
    ------
    ValueError
        If *price* contains non-positive values.
    """
    factor_names = [resolve_name(f) for f in (factors or DEFAULT_FACTORS)]
    base_windows = dict((cfg or {}).get("factors", {}))
    if windows:
        base_windows.update(windows)

    # Validate price series before any computation
    n_bad = int((price <= 0).sum())
    if n_bad > 0:
        raise MrvValidationError(
            f"build_factors: price series contains {n_bad} non-positive value(s). "
            "All prices must be strictly positive for log-return computation."
        )

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


