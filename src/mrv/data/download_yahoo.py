"""
mrv.data.download_yahoo -- Download OHLCV data via Yahoo Finance (yfinance).

Free, no account needed. Supports daily and intraday (1h) data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mrv.exceptions import MrvConfigError
from mrv.utils.config import _normalize_freq

logger = logging.getLogger(__name__)


def _require_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError(
            "Yahoo download requires yfinance. Install with: pip install yfinance"
        ) from None


def download(cfg: Optional[Dict[str, Any]] = None) -> None:
    """Download all assets x freqs from Yahoo Finance."""
    cfg = cfg or {}
    dl_cfg = cfg.get("download", {})
    yf = _require_yfinance()

    symbols = dl_cfg.get("symbols", [])
    if not symbols:
        raise MrvConfigError("No symbols defined in config (download.symbols)")

    data_dir = Path(dl_cfg.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    freqs = _normalize_freq(dl_cfg.get("freq", ["1d"]))
    start = dl_cfg.get("start", "2020-01-01")
    end = dl_cfg.get("end") or datetime.now().strftime("%Y-%m-%d")

    # Yahoo freq mapping
    yf_interval = {
        "1d": "1d",
        "1h": "1h",
        "5m": "5m",
        "15m": "15m",
    }

    # Yahoo max period per interval
    yf_max_period = {
        "1d": None,      # unlimited with start/end
        "1h": "730d",     # ~2 years
        "15m": "60d",
        "5m": "60d",
    }

    for symbol in symbols:
        stem = str(symbol).replace("=F", "").replace("=X", "").replace("^", "").replace("/", "_")
        ticker = yf.Ticker(str(symbol))

        for freq in freqs:
            interval = yf_interval.get(freq)
            if interval is None:
                logger.warning("Unsupported Yahoo freq: %s, skipping", freq)
                continue

            out_path = data_dir / f"{stem}_{freq}.csv"

            logger.info("Downloading %s %s from Yahoo Finance", stem, freq)
            try:
                if freq == "1d":
                    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
                else:
                    max_p = yf_max_period.get(freq, "60d")
                    df = ticker.history(period=max_p, interval=interval, auto_adjust=False)

                if df.empty:
                    logger.warning("No data returned for %s/%s", stem, freq)
                    continue

                # Normalize columns
                col_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if "open" in cl:
                        col_map[c] = "Open"
                    elif "high" in cl:
                        col_map[c] = "High"
                    elif "low" in cl:
                        col_map[c] = "Low"
                    elif "close" in cl and "adj" not in cl:
                        col_map[c] = "Close"
                    elif "volume" in cl:
                        col_map[c] = "Volume"
                df = df.rename(columns=col_map)
                missing = [c for c in ("Open", "High", "Low", "Close") if c not in df.columns]
                if missing:
                    logger.warning("%s/%s: missing %s column(s), skipping",
                                   stem, freq, ", ".join(missing))
                    continue

                df.index.name = "Date"
                save_cols = ["Open", "High", "Low", "Close"]
                if "Volume" in df.columns:
                    save_cols.append("Volume")
                df[save_cols].to_csv(out_path)
                logger.info("Saved %s/%s (%d rows) -> %s", stem, freq, len(df), out_path)

            except Exception as e:
                logger.error("Yahoo download failed for %s/%s: %s", stem, freq, e)
