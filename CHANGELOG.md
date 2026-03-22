# Changelog

## 0.2.0 (2026-03-20)

### Breaking changes

- Removed `ReportGenerator` class — use `generate_report()` function instead
- Removed `build_intraday_features()` from factors (dead code)
- Moved `download.py` from `data/` to `utils/`
- Moved `factors/` into `data/factors.py`
- Moved `report/` into `validator/report.py`
- Removed `cli.py` and `__main__.py` — use `run.py` directly

### Improvements

- Added `pipeline.py` — standard 6-step flow (load → factors → fit → validate → report), each step replaceable
- Thresholds (ARI 0.65, Spearman 0.85) consolidated to single source in `metrics.py`
- Template conditionals (`%% IF_xxx / %% ELSE / %% ENDIF`) — all report text lives in LaTeX, Python only fills data
- Two-layer verdict: partition stability (ARI) + ordering stability (Spearman rank correlation)
- Report now reads thresholds from JSON, not hardcoded
- Division-by-zero guard in `rolling_zscore`
- Config comments updated to match actual behavior (each freq downloaded separately)
- Removed BTC-USD from default symbols (not available on IB)
- Aligned validator asset lists with download symbols

### Fixes

- Fixed undefined `model` variable in `rep.py` (was `model_name`)
- Fixed `pyproject.toml` entry point (was pointing to non-package `run.py`)
- Added missing `scipy` and `matplotlib` to optional dependencies

## 0.1.0 (2026-03-19)

First public release.

### Features

- **Data infrastructure**: IB download for all frequencies (5m/15m/1h/1d), incremental updates, OHLCV validation and resampling
- **Factor engine**: 7 built-in risk factors (volatility, drawdown, max drawdown, VaR, CVaR, realized skew, stability) with user-extensible registry
- **Model registry**: GMM and HMM built-in, user-extensible via `register_model()`
- **Normalization**: Rolling z-score and min-max
- **Validator: Representation Invariance (rep)**: Pairwise ARI across multiple factor sets, per-asset heatmaps, JSON output
- **Report generation**: JSON to LaTeX to PDF pipeline, professional template with cover page, dashboard, sign-off block, and remediation plan
- **CLI**: `python run.py run`, `python run.py download`, `python run.py report`
- **Configuration**: Single `config.yaml` with download, logging, factors, normalization, and validator sections
- **Logging**: Console + optional timestamped file logging
- **Security**: IB host/port validation, non-positive price detection
