"""
mrv.validator.monitor -- Continuous monitoring with alerting.

Provides ``monitor()`` for incremental daily validation with persistent
history tracking and configurable alerts (file + webhook).

Usage::

    from mrv.validator.monitor import monitor

    monitor("config.yaml", "rep", mode="init")          # baseline
    monitor("config.yaml", "rep", mode="incremental")    # daily
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

import pandas as pd

from mrv.exceptions import MrvConfigError
from mrv.utils.config import load
from mrv.utils.log import setup

logger = logging.getLogger(__name__)

__all__ = ["monitor"]

HISTORY_COLUMNS = [
    "date", "asset", "validator", "mean_ari", "mean_ari_7d_avg",
    "delta_vs_baseline", "alert_fired",
]


def monitor(
    config: "Optional[str | Path]" = None,
    validator: str = "rep",
    mode: str = "incremental",
    cfg: Optional[Dict[str, Any]] = None,
    impact_fn=None,
) -> Dict[str, Any]:
    """Run monitoring cycle: validate -> update history -> check alerts.

    Parameters
    ----------
    config : path to config.yaml, or None when ``cfg`` is provided.
    validator : str
        Only ``"rep"`` (representation invariance) is supported for continuous
        monitoring. Resolution invariance requires the caller to supply
        pre-fitted labels at each frequency; use ``validate_res()`` directly
        and build your own history loop.
    mode : str
        ``"init"`` for first-run baseline, ``"incremental"`` for daily append.
    cfg : dict, optional
        Pre-loaded config dict. Overrides ``config`` when supplied.
    impact_fn : callable, optional
        Optional business-impact callback passed through to the validator.

    Raises
    ------
    ValueError
        If ``validator`` is not ``"rep"``.
    """
    if validator != "rep":
        raise MrvConfigError(
            f"monitor() only supports validator='rep'. "
            f"Got '{validator}'. For resolution invariance monitoring, "
            f"call validate_res(labels=...) directly and maintain your own "
            f"history loop."
        )
    if cfg is None:
        cfg = load(config)
    setup(cfg)

    v_cfg = cfg.get("validator", {})
    test_cfg = v_cfg.get(validator, {})
    mon_cfg = test_cfg.get("monitoring", {})
    report_dir = Path(v_cfg.get("report_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    history_path = report_dir / f"monitoring_history_{validator}.csv"
    alerts_path = report_dir / "alerts.json"

    today = datetime.now().strftime("%Y-%m-%d")

    # Load existing history
    history = _load_history(history_path)

    # Idempotency: skip if already run today
    if mode == "incremental" and _is_already_run(history, today, validator):
        logger.info("Monitor: already run for %s on %s, skipping", validator, today)
        return {"status": "skipped", "reason": "already_run_today"}

    # Run validation
    logger.info("=== Monitor (%s): %s ===", mode, validator)
    from mrv.pipeline import validate
    result = validate(cfg, validator, impact_fn=impact_fn)

    # Extract per-asset metrics
    new_rows = _extract_metrics(result, validator, today)
    if not new_rows:
        logger.warning("Monitor: no metrics extracted")
        return {"status": "no_data"}

    # Compute 7-day moving averages and deltas
    new_rows_df = pd.DataFrame(new_rows)
    history = _append_history(history, new_rows_df)

    # Compute rolling averages (mutates history in-place; written once below)
    _compute_rolling_stats(history)

    # Atomic write: write to temp file, then rename to avoid partial-state CSV.
    _write_history_atomic(history, history_path)

    # Check alerts
    alerts = _check_alerts(new_rows_df, history, mon_cfg)
    if alerts:
        _fire_alerts(alerts, alerts_path, mon_cfg)

    logger.info("Monitor: %d assets, %d alerts", len(new_rows), len(alerts))
    return {
        "status": "ok",
        "mode": mode,
        "date": today,
        "n_assets": len(new_rows),
        "alerts": alerts,
        "history_path": str(history_path),
    }


# -- History management -------------------------------------------------------

def _load_history(path: Path) -> pd.DataFrame:
    """Load or create empty history DataFrame."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.warning("Could not read history %s: %s", path, e)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _is_already_run(history: pd.DataFrame, date: str, validator: str) -> bool:
    if history.empty:
        return False
    mask = (history["date"] == date) & (history["validator"] == validator)
    return mask.any()


def _extract_metrics(
    result: Dict[str, Any],
    validator: str,
    date: str,
) -> List[Dict[str, Any]]:
    """Extract per-asset mean_ari from validation result."""
    rows = []
    assets = result.get("assets", {})
    for asset_name, asset_result in assets.items():
        # rep validator uses "mean_ari", res uses "overall_mean_ari"
        mean_ari = asset_result.get("mean_ari") or asset_result.get("overall_mean_ari")
        if mean_ari is None:
            continue
        rows.append({
            "date": date,
            "asset": asset_name,
            "validator": validator,
            "mean_ari": round(float(mean_ari), 6),
            "mean_ari_7d_avg": None,
            "delta_vs_baseline": None,
            "alert_fired": False,
        })
    return rows


def _append_history(
    history: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Append new rows; caller is responsible for writing back atomically."""
    return pd.concat([history, new_rows], ignore_index=True)


def _write_history_atomic(history: pd.DataFrame, path: Path) -> None:
    """Write history CSV atomically via a temp file + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            history.to_csv(f, index=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _compute_rolling_stats(history: pd.DataFrame) -> None:
    """Compute 7-day rolling average and delta vs first row (baseline), in-place."""
    if history.empty:
        return

    for _key, grp in history.groupby(["asset", "validator"]):
        idx = grp.index
        ari_values = grp["mean_ari"].astype(float)

        # 7-day rolling average
        rolling = ari_values.rolling(window=7, min_periods=1).mean()
        history.loc[idx, "mean_ari_7d_avg"] = rolling.round(6)

        # Delta vs baseline (first entry for this asset+validator)
        baseline = float(ari_values.iloc[0])
        history.loc[idx, "delta_vs_baseline"] = (ari_values - baseline).round(6)


# -- Alerting -----------------------------------------------------------------

def _check_alerts(
    new_rows: pd.DataFrame,
    history: pd.DataFrame,
    mon_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check new rows against alert thresholds."""
    alerts = []
    threshold_below = mon_cfg.get("alert_ari_below")
    threshold_delta = mon_cfg.get("alert_ari_delta")

    for _, row in new_rows.iterrows():
        reasons = []

        if threshold_below is not None and row["mean_ari"] < threshold_below:
            reasons.append(f"ARI={row['mean_ari']:.3f} < {threshold_below}")

        if threshold_delta is not None:
            # Compare vs 7-day avg from previous runs
            asset_hist = history[
                (history["asset"] == row["asset"])
                & (history["validator"] == row["validator"])
                & (history["date"] < row["date"])
            ]
            if not asset_hist.empty:
                prev_avg = asset_hist["mean_ari"].astype(float).tail(7).mean()
                delta = row["mean_ari"] - prev_avg
                if delta < threshold_delta:
                    reasons.append(f"delta={delta:+.3f} < {threshold_delta}")

        if reasons:
            alerts.append({
                "date": row["date"],
                "asset": row["asset"],
                "validator": row["validator"],
                "mean_ari": row["mean_ari"],
                "reasons": reasons,
                "timestamp": datetime.now().isoformat(),
            })

    # Mark alert_fired in new_rows (use .loc on a copy-safe reference)
    alerted_assets = {a["asset"] for a in alerts}
    mask = new_rows["asset"].isin(alerted_assets)
    if mask.any():
        new_rows.loc[mask, "alert_fired"] = True

    return alerts


def _fire_alerts(
    alerts: List[Dict[str, Any]],
    alerts_path: Path,
    mon_cfg: Dict[str, Any],
) -> None:
    """Write alerts to file and optionally POST to webhook."""
    # File alert (always)
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alerts_path, "a", encoding="utf-8") as f:
        for alert in alerts:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")
    logger.warning("Alerts written to %s: %d alerts", alerts_path, len(alerts))

    # Webhook alerts (if configured)
    channels = mon_cfg.get("alert_channels", [])
    for channel in channels:
        if channel.get("type") == "webhook":
            url = os.path.expandvars(channel.get("url", ""))
            if url and url.startswith("http"):
                _post_webhook(url, alerts)


def _post_webhook(url: str, alerts: List[Dict[str, Any]]) -> None:
    """POST alerts to a webhook URL (fire-and-forget)."""
    payload = json.dumps({"alerts": alerts}, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            logger.info("Webhook POST %s -> %d", url, resp.status)
    except Exception as e:
        logger.warning("Webhook POST failed: %s", e)
