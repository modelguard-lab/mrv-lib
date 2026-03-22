"""
mrv.data.download — Download OHLCV data via Interactive Brokers (minute-level).

Adapted from Paper 2 (Resolution-Invariant) ``data_ib.py``.
Requires: IB Gateway or TWS running locally.
Optional dependency: ``ib_insync`` (lazy-imported so the rest of mrv works without it).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from mrv.utils.config import get_assets, _normalize_freq
from mrv.data.reader import load_ohlcv

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}


# ---------------------------------------------------------------------------
# Lazy ib_insync import
# ---------------------------------------------------------------------------

def _require_ib() -> tuple:
    """Import ``ib_insync`` lazily so users without IB can still use other modules."""
    try:
        from ib_insync import IB, ContFuture, Forex, Future, Index, Stock
        return IB, Forex, Stock, Future, Index, ContFuture
    except ImportError:
        raise ImportError(
            "IB download requires ib_insync.  Install with: pip install ib_insync"
        ) from None


# ---------------------------------------------------------------------------
# Contract builders
# ---------------------------------------------------------------------------

def canonical_stem(symbol: str) -> str:
    """Map ticker to canonical file stem (e.g. ``^GSPC`` -> ``GSPC``, ``CL=F`` -> ``CL``)."""
    s = str(symbol).strip().upper()
    s = s.replace("^", "").replace("/", "_").replace("=F", "").replace("=X", "").replace(".", "_")
    if s == "CLF":
        return "CL"
    if s == "USDJPYX":
        return "USDJPY"
    return s


def build_contract(
    symbol: str,
    kind: Optional[str] = None,
    exchange: Optional[str] = None,
    future_expiry: Optional[str] = None,
) -> Any:
    """
    Build an IB Contract for *symbol*.

    Parameters
    ----------
    symbol : str
        Ticker, e.g. ``"SPY"``, ``"CL=F"``, ``"USDJPY"``.
    kind : str, optional
        ``"stock"`` | ``"forex"`` | ``"future"`` | ``"index"``.
        Auto-detected if *None*.
    exchange : str, optional
        IB exchange override.
    future_expiry : str, optional
        Expiry code (e.g. ``"202612"``) or ``"CONTFUT"`` for continuous front-month.
    """
    IB, Forex, Stock, Future, Index, ContFuture = _require_ib()
    sym = canonical_stem(symbol)

    # ^GSPC -> GSPC -> map to SPX (IB's symbol for S&P 500 index)
    _INDEX_MAP = {"GSPC": "SPX"}
    if sym in _INDEX_MAP:
        sym = _INDEX_MAP[sym]

    if kind is None:
        if sym in ("SPX",):
            kind = "index"
        elif sym in ("USDJPY", "EURUSD", "GBPUSD"):
            kind = "forex"
        elif sym in ("CL", "ES", "GC"):
            kind = "future"
        else:
            kind = "stock"

    if kind == "index":
        return Index(sym, exchange or "CBOE", "USD")
    if kind == "stock":
        return Stock(sym, exchange or "SMART", "USD")
    if kind == "forex":
        raw = symbol.upper().replace("=X", "").replace("/", "").replace(".", "")
        return Forex(raw, exchange or "IDEALPRO")
    if kind == "future":
        if _is_contfut(future_expiry):
            return ContFuture(sym, exchange or "NYMEX", "USD")
        exp = future_expiry
        if exp is None:
            y = datetime.now().year
            exp = f"{y}12" if datetime.now().month < 12 else f"{y + 1}12"
        return Future(sym, exchange=exchange or "NYMEX", currency="USD",
                      lastTradeDateOrContractMonth=exp)
    raise ValueError(f"Unknown contract kind: {kind}")


def _is_contfut(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().upper() in {"CONTFUT", "CONT_FUT", "CONTINUOUS", "CONT"}


# ---------------------------------------------------------------------------
# IB datetime helpers
# ---------------------------------------------------------------------------

def _parse_dt(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        return s
    return pd.to_datetime(s).to_pydatetime()


def _to_ib_end_datetime(dt: datetime | pd.Timestamp, tz: str = "America/New_York") -> str:
    ts = pd.Timestamp(dt)
    ts = ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    utc = ts.tz_convert("UTC")
    return utc.strftime("%Y%m%d-%H:%M:%S")


def _ib_duration_for_days(days: int) -> str:
    days = max(1, int(days))
    if days <= 365:
        return f"{days} D"
    return f"{(days + 364) // 365} Y"


# ---------------------------------------------------------------------------
# Core fetch (single connection, called by public API)
# ---------------------------------------------------------------------------

# Map mrv freq string -> IB barSizeSetting
_IB_BAR_SIZE = {
    "5m": "5 mins",
    "15m": "15 mins",
    "1h": "1 hour",
    "1d": "1 day",
}

# IB limits lookback differently per bar size; use appropriate chunk sizes
_IB_CHUNK = {
    "5m": "1 W",
    "15m": "2 W",
    "1h": "1 M",
    "1d": "1 Y",
}


def _fetch_bars(
    ib: Any,
    contract: Any,
    start_date: str | datetime,
    end_date: str | datetime,
    freq: str = "5m",
    duration_chunk: Optional[str] = None,
    use_rth: bool = False,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Request historical bars at the given frequency using an already-connected IB."""
    bar_size = _IB_BAR_SIZE.get(freq)
    if bar_size is None:
        raise ValueError(f"Unsupported frequency for IB download: {freq}")
    if duration_chunk is None:
        duration_chunk = _IB_CHUNK.get(freq, "1 W")

    start_dt = _parse_dt(start_date)
    end_dt = _parse_dt(end_date)

    start_ts = pd.Timestamp(start_dt)
    start_ts = start_ts.tz_localize(tz) if start_ts.tzinfo is None else start_ts.tz_convert(tz)
    end_ts = pd.Timestamp(end_dt)
    end_ts = end_ts.tz_localize(tz) if end_ts.tzinfo is None else end_ts.tz_convert(tz)
    if end_ts.hour == 0 and end_ts.minute == 0:
        end_ts = end_ts.replace(hour=16)

    is_forex = getattr(contract, "secType", None) == "CASH" or contract.__class__.__name__ == "Forex"
    is_contfut = getattr(contract, "secType", None) == "CONTFUT" or contract.__class__.__name__ == "ContFuture"
    what_to_show = "MIDPOINT" if is_forex else "TRADES"
    use_rth_param = 0 if is_forex else (1 if use_rth else 0)

    all_bars: list = []

    if is_contfut:
        dur_days = max(7, (end_ts.normalize() - start_ts.normalize()).days + 10)
        dur_str = _ib_duration_for_days(dur_days)
        try:
            bars = ib.reqHistoricalData(
                contract, endDateTime="", durationStr=dur_str,
                barSizeSetting=bar_size, whatToShow=what_to_show,
                useRTH=use_rth_param, formatDate=1, timeout=180,
            )
        except Exception as e:
            logger.warning("reqHistoricalData failed for continuous future: %s", e)
            bars = []
        all_bars.extend(bars or [])
    else:
        current_end = end_ts.to_pydatetime()
        last_oldest: Optional[pd.Timestamp] = None
        while current_end > start_ts.to_pydatetime():
            end_str = _to_ib_end_datetime(current_end, tz)
            try:
                bars = ib.reqHistoricalData(
                    contract, endDateTime=end_str, durationStr=duration_chunk,
                    barSizeSetting=bar_size, whatToShow=what_to_show,
                    useRTH=use_rth_param, formatDate=1, timeout=120,
                )
            except Exception as e:
                logger.warning("reqHistoricalData failed for end=%s: %s", end_str, e)
                break
            if not bars:
                break
            all_bars.extend(bars)
            bar_times = []
            for b in bars:
                t = b.date
                ts = pd.Timestamp(datetime.fromtimestamp(t.timestamp())) if hasattr(t, "timestamp") else pd.Timestamp(t)
                ts = ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
                bar_times.append(ts)
            oldest = min(bar_times)
            if last_oldest is not None and oldest >= last_oldest:
                break
            last_oldest = oldest
            current_end = (oldest - pd.Timedelta(minutes=1)).to_pydatetime()
            time.sleep(0.5)

    if not all_bars:
        return pd.DataFrame()

    rows = []
    for b in all_bars:
        t = b.date
        ts = pd.Timestamp(datetime.fromtimestamp(t.timestamp())) if hasattr(t, "timestamp") else pd.Timestamp(t)
        rows.append({"Date": ts, "Open": b.open, "High": b.high,
                      "Low": b.low, "Close": b.close, "Volume": getattr(b, "volume", 0)})
    df = pd.DataFrame(rows).set_index("Date")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index = df.index.tz_localize("UTC").tz_convert(tz) if df.index.tz is None else df.index.tz_convert(tz)

    # Trim to requested range
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download(cfg: Optional[Dict[str, Any]] = None) -> None:
    """Download all assets × freqs from IB."""
    cfg = cfg or {}
    dl_cfg = cfg.get("download", {})
    assets = get_assets(cfg)
    if not assets:
        raise ValueError("No assets defined in config (download.symbols)")

    data_dir = Path(dl_cfg.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    IB_cls = _require_ib()[0]
    ib_cfg = dl_cfg.get("ib", {})
    host = str(ib_cfg.get("host", "127.0.0.1"))
    port = int(ib_cfg.get("port", 4002))
    if not 1 <= port <= 65535:
        raise ValueError(f"IB port out of range: {port}")
    if host not in _ALLOWED_HOSTS and not host.startswith("192.168."):
        logger.warning("IB host '%s' is not localhost — ensure this is intentional", host)
    client_id = ib_cfg.get("client_id")
    if client_id is None:
        client_id = int.from_bytes(os.urandom(2), "big") % 9900 + 100
    tz = ib_cfg.get("tz", "America/New_York")
    use_rth = bool(ib_cfg.get("use_rth", False))
    future_expiry_map = ib_cfg.get("future_expiry", {})

    ib = IB_cls()
    try:
        ib.connect(host, port, clientId=int(client_id), readonly=True)
        logger.info("IB connected host=%s port=%s clientId=%s", host, port, client_id)
    except Exception as e:
        logger.error("IB connect failed: %s (host=%s port=%s). Is Gateway/TWS running?", e, host, port)
        raise

    try:
        for entry in assets:
            ticker = entry["symbol"]
            freqs = _normalize_freq(entry.get("freq"))
            start = entry.get("start", "2026-01-01")
            end = entry.get("end") or datetime.now().strftime("%Y-%m-%d")
            stem = canonical_stem(ticker)
            future_expiry = future_expiry_map.get(stem) or future_expiry_map.get(ticker)
            contract = build_contract(ticker, future_expiry=future_expiry)

            for freq in freqs:
                out_path = data_dir / f"{stem}_{freq}.csv"

                effective_start = start
                existing_df: Optional[pd.DataFrame] = None
                if out_path.exists():
                    try:
                        existing_df = load_ohlcv(out_path, tz=tz)
                        if not existing_df.empty:
                            last_ts = existing_df.index.max()
                            step = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}.get(freq, 5)
                            next_ts = pd.Timestamp(last_ts) + pd.Timedelta(minutes=step)
                            end_ts = pd.Timestamp(end).tz_localize(tz) + pd.Timedelta(days=1)
                            if next_ts >= end_ts:
                                logger.info("Skip %s/%s: up to date", stem, freq)
                                continue
                            effective_start = next_ts.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        logger.warning("Could not read %s, full download: %s", out_path, e)

                logger.info("Downloading %s %s from %s to %s", stem, freq, effective_start, end)
                try:
                    df = _fetch_bars(ib, contract, effective_start, end,
                                     freq=freq, use_rth=use_rth, tz=tz)
                except Exception as e:
                    logger.error("IB fetch failed for %s/%s: %s", stem, freq, e)
                    continue

                if df.empty:
                    logger.info("No new bars for %s/%s", stem, freq)
                    continue

                if existing_df is not None and not existing_df.empty:
                    df = pd.concat([existing_df, df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()

                df.index.name = "Date"
                df.to_csv(out_path)
                logger.info("Saved %s/%s (%d rows) -> %s", stem, freq, len(df), out_path)
                time.sleep(0.5)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
