"""
mrv.pipeline — Standard pipeline: data → factors → model → validate → report.

Each step is a plain function, easy to replace or skip.

Usage::

    from mrv.pipeline import run, download

    download("config.yaml")      # fetch data from IB
    run("config.yaml")           # validate → report
    run("config.yaml", "rep")    # explicit validator

    # Step by step (full control)
    from mrv.pipeline import load_data, compute_factors, fit_labels, validate, report

    cfg = load("config.yaml")
    prices = load_data(cfg, "rep")                       # {asset: Series}
    factors = compute_factors(prices, cfg, factor_sets)   # {asset: {set_label: DataFrame}}
    labels = fit_labels(factors, model="gmm")             # {asset: {set_label: ndarray}}
    result = validate(cfg, "rep", labels=labels)          # run metrics + save
    report(result["json_path"], cfg=cfg)                  # PDF
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from mrv.utils.config import load
from mrv.utils.log import setup
from mrv.utils.download import download as _ib_download
from mrv.data.reader import load_ohlcv
from mrv.data.factors import build_factors, resolve_name
from mrv.data.normalize import normalize
from mrv.models import fit as fit_model
from mrv.validator.rep import RepValidator
from mrv.validator.res import ResValidator
from mrv.validator.report import generate_report as _generate_report

logger = logging.getLogger(__name__)

_VALIDATORS: Dict[str, type] = {"rep": RepValidator, "res": ResValidator}


# ---------------------------------------------------------------------------
# Step 1: Download (optional, user may have own data)
# ---------------------------------------------------------------------------

def download(config: Optional[str | Path] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Download data from IB. Returns loaded config."""
    if cfg is None:
        cfg = load(config)
    setup(cfg)
    logger.info("=== Download ===")
    _ib_download(cfg=cfg)
    return cfg


# ---------------------------------------------------------------------------
# Step 2: Load data → {asset_name: price_series}
# ---------------------------------------------------------------------------

def load_data(
    cfg: Dict[str, Any],
    validator: str = "rep",
) -> "Dict[str, pd.Series] | Dict[str, pd.DataFrame]":
    """
    Load price data for a validator's assets.

    For ``rep`` validator, returns ``{asset_name: price_series}``.
    For ``res`` validator, returns ``{asset_name: ohlcv_5m_dataframe}``.
    Replace this function to use your own data source.
    """
    v_cfg = cfg.get("validator", {}).get(validator, {})
    assets_map = v_cfg.get("assets", {})
    start = v_cfg.get("start")
    end = v_cfg.get("end")

    prices = {}
    for name, path_val in assets_map.items():
        # res validator: paths is a list → use first (5m) path
        if isinstance(path_val, list):
            path = Path(path_val[0])
        else:
            path = Path(path_val)

        if not path.exists():
            logger.warning("Skip %s: %s not found", name, path)
            continue

        df = load_ohlcv(path)

        if start:
            df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]

        if len(df) < 50:
            logger.warning("Skip %s: too few data (%d)", name, len(df))
            continue

        # For res validator, return full OHLCV DataFrame
        if validator == "res":
            prices[name] = df
        else:
            price = df["Close"] if "Close" in df.columns else df["close"]
            prices[name] = price

    return prices


# ---------------------------------------------------------------------------
# Step 3: Price → factors (per asset, per factor set)
# ---------------------------------------------------------------------------

def compute_factors(
    prices: Dict[str, pd.Series],
    cfg: Dict[str, Any],
    factor_sets: List[List[str]],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Compute normalized factor matrices.

    Returns ``{asset: {set_label: normalized_df}}``.
    """
    result = {}
    for asset, price in prices.items():
        asset_factors = {}
        for fs in factor_sets:
            resolved = [resolve_name(f) for f in fs]
            label = ", ".join(fs)
            raw = build_factors(price, factors=resolved, cfg=cfg)
            normed = normalize(raw, cfg=cfg).dropna()
            if len(normed) >= 50:
                asset_factors[label] = normed
        result[asset] = asset_factors
    return result


# ---------------------------------------------------------------------------
# Step 4: Factors → regime labels
# ---------------------------------------------------------------------------

def fit_labels(
    factors: Dict[str, Dict[str, pd.DataFrame]],
    model: str = "gmm",
    n_states: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit regime model on each factor set.

    Returns ``{asset: {set_label: ndarray}}``.
    """
    result = {}
    for asset, sets in factors.items():
        asset_labels = {}
        for label, df in sets.items():
            labels = fit_model(df, model=model, n_states=n_states)
            if labels is not None:
                asset_labels[label] = labels
        result[asset] = asset_labels
    return result


# ---------------------------------------------------------------------------
# Step 5: Validate (metrics + output)
# ---------------------------------------------------------------------------

def validate(
    cfg: Dict[str, Any],
    name: str = "rep",
    prices: Optional[Dict[str, pd.Series]] = None,
    labels: Optional[Dict[str, Dict[str, Any]]] = None,
    impact_fn=None,
) -> Dict[str, Any]:
    """
    Run a validator. If prices/labels are provided, they're passed through
    (skip internal data loading / model fitting).

    Parameters
    ----------
    impact_fn : callable, optional
        ``(labels: ndarray, prices: Series) -> float``.
        When provided, computes a business impact matrix across representations.
    """
    cls = _VALIDATORS.get(name)
    if cls is None:
        raise ValueError(f"Unknown validator '{name}'. Available: {list(_VALIDATORS.keys())}")
    logger.info("=== Validate: %s ===", name)
    v = cls(cfg, impact_fn=impact_fn)
    return v.validate(prices=prices, labels=labels)


# ---------------------------------------------------------------------------
# Step 6: Report
# ---------------------------------------------------------------------------

def report(
    json_path: str | Path,
    template: Optional[str | Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Generate PDF report from JSON."""
    logger.info("=== Report ===")
    return _generate_report(json_path, template=template, cfg=cfg)


def sr11_7_report(
    json_path: str | Path,
    template: Optional[str | Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    overrides: Optional[str | Path] = None,
) -> Optional[Path]:
    """Generate SR 11-7 compliant PDF report from JSON."""
    from mrv.validator.report import generate_sr11_7_report
    logger.info("=== SR 11-7 Report ===")
    return generate_sr11_7_report(json_path, template=template, cfg=cfg, overrides=overrides)


# ---------------------------------------------------------------------------
# Convenience: run = validate + report
# ---------------------------------------------------------------------------

def run(
    config: Optional[str | Path] = None,
    validator: str = "rep",
    cfg: Optional[Dict[str, Any]] = None,
    impact_fn=None,
) -> Optional[Path]:
    """Validate → report. Returns PDF path or None."""
    if cfg is None:
        cfg = load(config)
    setup(cfg)
    result = validate(cfg, validator, impact_fn=impact_fn)
    json_path = result.get("json_path")
    if json_path:
        return report(json_path, cfg=cfg)
    return None


def monitor(
    config: Optional[str | Path] = None,
    validator: str = "rep",
    mode: str = "incremental",
    cfg: Optional[Dict[str, Any]] = None,
    impact_fn=None,
) -> Dict[str, Any]:
    """Run monitoring cycle: validate → history → alerts.

    Parameters
    ----------
    mode : str
        ``"init"`` for baseline, ``"incremental"`` for daily append.
    """
    from mrv.validator.monitor import monitor as _monitor
    return _monitor(config=config, validator=validator, mode=mode, cfg=cfg, impact_fn=impact_fn)


def register_validator(name: str, cls: type) -> None:
    """Register a custom validator class."""
    _VALIDATORS[name] = cls
