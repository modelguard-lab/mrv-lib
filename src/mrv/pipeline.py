"""
mrv.pipeline -- INTERNAL labels-first backend for the ``mrv`` CLI.

This module is not the recommended public Python API. It powers the config-file
CLI workflow (``mrv init`` / ``mrv download`` / ``mrv run``) and exposes
dict-returning labels-first helpers. For programmatic use, prefer the top-level
typed validators::

    import mrv

    mrv.rep_invariance_validator(model_fn, admissible_class, ...)   # Paper 1
    mrv.res_invariance_validator(model_fn, resolution_set, ...)     # Paper 2

Those return typed results (``RepInvarianceResult`` / ``ResInvarianceResult``)
and are the stable public contract. The helpers below remain importable as the
CLI backend but may change.

Labels-first helpers (no model fitting -- callers supply labels)::

    from mrv.pipeline import validate_rep, validate_res

    # Representation invariance: caller provides labels from their own models
    result = validate_rep(labels={
        "SPY": {"rep_a": labels_a, "rep_b": labels_b, "rep_c": labels_c}
    })

    # Resolution invariance: caller provides labels at each frequency
    result = validate_res(labels={
        "SPY": {"5m": labels_5m, "15m": labels_15m, "1h": labels_1h, "1d": labels_1d}
    })

Convenience run (model fitting included -- requires ``pip install scikit-learn``)::

    from mrv.pipeline import run

    run("config.yaml", "rep")   # data -> factors -> model -> validate -> report
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mrv.exceptions import MrvConfigError
from mrv.validator.rep import RepValidator
from mrv.validator.report import generate_report as _generate_report
from mrv.validator.res import ResValidator

logger = logging.getLogger(__name__)

__all__ = [
    "validate_rep",
    "validate_res",
    "report",
    "download",
    "run",
    "validate",
]

_VALIDATORS: Dict[str, type] = {"rep": RepValidator, "res": ResValidator}


# ---------------------------------------------------------------------------
# Core API: labels-first (no model fitting)
# ---------------------------------------------------------------------------

def validate_rep(
    labels: Dict[str, Dict[str, np.ndarray]],
    risk_proxy: Optional[Dict[str, np.ndarray]] = None,
    prices: Optional[Dict[str, pd.Series]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    impact_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Representation invariance test -- labels in, metrics out.

    Parameters
    ----------
    labels : dict
        ``{asset: {spec_label: ndarray}}``.  At least 2 specs per asset.
    risk_proxy : dict, optional
        ``{asset: risk_array}`` for ordering consistency (Spearman).
    prices : dict, optional
        ``{asset: price_series}`` for business impact computation.
    cfg : dict, optional
        Config for report paths / thresholds.
    impact_fn : callable, optional
        ``(labels, prices) -> float`` for business impact.
    """
    v = RepValidator(cfg=cfg, impact_fn=impact_fn)
    return v.validate(labels=labels, risk_proxy=risk_proxy, prices=prices)


def validate_res(
    labels: Dict[str, Dict[str, pd.Series]],
    event_window: Optional[Any] = None,
    calm_window: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
    impact_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Resolution invariance test -- labels in, metrics out.

    Parameters
    ----------
    labels : dict
        ``{asset: {freq: pd.Series}}``.  At least 2 frequencies per asset.
    event_window : tuple, optional
        ``(start_date, end_date)`` for event-period analysis.
    calm_window : tuple, optional
        ``(start_date, end_date)`` for calm-period analysis.
    cfg : dict, optional
        Config for report paths / thresholds.
    """
    v = ResValidator(cfg=cfg, impact_fn=impact_fn)
    return v.validate(labels=labels, event_window=event_window, calm_window=calm_window)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def report(
    json_path: str | Path,
    template: Optional[str | Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Generate PDF report from JSON."""
    logger.info("=== Report ===")
    return _generate_report(json_path, template=template, cfg=cfg)


# ---------------------------------------------------------------------------
# Convenience API: config -> data -> model -> validate -> report
# (requires scikit-learn: pip install scikit-learn)
# ---------------------------------------------------------------------------

def _load_config(config=None, cfg=None):
    """Load config from path or use provided dict."""
    if cfg is not None:
        return cfg
    from mrv.utils.config import load
    return load(config)


def download(config=None, cfg=None) -> Dict[str, Any]:
    """Download data from Yahoo Finance or IB Gateway (based on config).

    The ``download.source`` field in config.yaml controls the data source:
    ``yahoo`` (default, free) or ``ib`` (requires IB Gateway running).
    """
    cfg = _load_config(config, cfg)
    from mrv.utils.log import setup
    setup(cfg)

    source = cfg.get("download", {}).get("source", "yahoo").lower()
    logger.info("=== Download (source: %s) ===", source)

    if source == "ib":
        from mrv.utils.download_ib import download as _ib_download
        _ib_download(cfg=cfg)
    elif source == "yahoo":
        from mrv.data.download_yahoo import download as _yahoo_download
        _yahoo_download(cfg=cfg)
    else:
        raise MrvConfigError(
            f"Unknown download source: '{source}'. Use 'yahoo' or 'ib'."
        )

    return cfg


def run(
    config=None,
    validator: str = "rep",
    cfg: Optional[Dict[str, Any]] = None,
    impact_fn=None,
) -> Optional[Path]:
    """Convenience: config -> data -> model -> validate -> report.

    This is the full pipeline for users who want mrv-lib to handle
    everything including model fitting.  Requires ``pip install scikit-learn``.

    For the labels-first API (recommended), use ``validate_rep()`` or
    ``validate_res()`` directly.
    """
    cfg = _load_config(config, cfg)
    from mrv.utils.log import setup
    setup(cfg)

    if validator == "rep":
        result = _run_rep_convenience(cfg, impact_fn)
    elif validator == "res":
        result = _run_res_convenience(cfg, impact_fn)
    else:
        raise MrvConfigError(f"Unknown validator '{validator}'. Available: rep, res")

    json_path = result.get("json_path")
    if json_path:
        return report(json_path, cfg=cfg)
    return None


def _run_rep_convenience(cfg, impact_fn=None):
    """Run representation invariance with internal model fitting."""
    from mrv.data.factors import build_factors, resolve_name
    from mrv.data.normalize import normalize
    from mrv.data.reader import load_ohlcv
    from mrv.models import fit as fit_model

    rep_cfg = cfg.get("validator", {}).get("rep", {})
    factor_sets_raw = rep_cfg.get("factors", [])
    if len(factor_sets_raw) < 2:
        raise MrvConfigError("Need >= 2 factor sets in config for rep validator")

    model_name = rep_cfg.get("model", "gmm")
    n_states = rep_cfg.get("n_states", 3)
    assets_map = rep_cfg.get("assets", {})

    labels: Dict[str, Dict[str, np.ndarray]] = {}
    risk_proxies: Dict[str, np.ndarray] = {}
    prices_dict: Dict[str, pd.Series] = {}

    for name, path_val in assets_map.items():
        path = Path(path_val) if isinstance(path_val, str) else Path(path_val[0])
        if not path.exists():
            logger.warning("Skip %s: %s not found", name, path)
            continue
        df = load_ohlcv(path)
        price = df["Close"] if "Close" in df.columns else df["close"]
        prices_dict[name] = price

        from mrv.data.factors import log_returns, volatility
        vol = volatility(log_returns(price), window=20, annualize=False)

        asset_labels = {}
        asset_label_index = None
        for fs in factor_sets_raw:
            resolved = [resolve_name(f) for f in fs]
            label = ", ".join(fs)
            raw = build_factors(price, factors=resolved, cfg=cfg)
            normed = normalize(raw, cfg=cfg).dropna()
            if len(normed) < 50:
                continue
            result = fit_model(normed, model=model_name, n_states=n_states)
            if result is not None:
                asset_labels[label] = result
                if asset_label_index is None:
                    asset_label_index = normed.index
        labels[name] = asset_labels

        # Align the risk proxy to the label index. ordering_consistency
        # compares proxy and labels positionally, so both must start on the
        # same date; reindexing the volatility onto the post-dropna feature
        # index removes the normalization-window offset that would otherwise
        # make the reported Spearman meaningless.
        if asset_label_index is not None:
            risk_proxies[name] = vol.reindex(asset_label_index).values
        else:
            risk_proxies[name] = vol.dropna().values

    return validate_rep(
        labels=labels, risk_proxy=risk_proxies, prices=prices_dict,
        cfg=cfg, impact_fn=impact_fn,
    )


def _run_res_convenience(cfg, impact_fn=None):
    """Resolution invariance has no model-fitting convenience path.

    Resolution invariance is labels-first by design: the caller must fit their
    own regime models at each frequency. There is no config-driven convenience
    or monitoring path for it. This always raises so the failure is loud and
    immediate rather than a silent no-op.
    """
    raise NotImplementedError(
        "Resolution invariance has no convenience/monitoring pipeline "
        "(validator='res' is not supported by run(), validate(), or monitor()). "
        "It is labels-first: fit your own regime models at each frequency and "
        "call mrv.pipeline.validate_res(labels=...) directly."
    )


def validate(
    config=None,
    validator: str = "rep",
    cfg: Optional[Dict[str, Any]] = None,
    impact_fn=None,
) -> Dict[str, Any]:
    """Dispatch a rep convenience validation run by name.

    Wraps ``_run_rep_convenience`` and returns the result dict. This is the
    backend used by ``mrv.validator.monitor.monitor()``, which only supports
    representation invariance. ``validator="res"`` is rejected: resolution
    invariance is labels-first and has no convenience/monitoring path (call
    ``validate_res(labels=...)`` directly).

    Parameters
    ----------
    config    : path to config.yaml (or None if ``cfg`` is provided).
    validator : ``"rep"`` only. ``"res"`` raises ``NotImplementedError``.
    cfg       : pre-loaded config dict (overrides ``config``).
    impact_fn : optional business-impact callback.
    """
    cfg = _load_config(config, cfg)
    from mrv.utils.log import setup
    setup(cfg)

    if validator == "rep":
        return _run_rep_convenience(cfg, impact_fn=impact_fn)
    elif validator == "res":
        return _run_res_convenience(cfg, impact_fn=impact_fn)
    else:
        raise MrvConfigError(f"Unknown validator '{validator}'. Available: rep, res")


def _register_validator(name: str, cls: type) -> None:
    """Register a custom validator class (internal)."""
    _VALIDATORS[name] = cls
