# Changelog

## 0.2.1 (2026-03-30)

- Fix CI lint errors (unused imports, lambda assignment)
- Upgrade GitHub Actions to Node.js 22 (checkout v5, setup-python v6)
- Trim shipped releases from ROADMAP

## 0.2.0 (2026-03-30)

- **Validator: Resolution Invariance (res)**: Multi-frequency regime analysis (5m/15m/1h/1d), cross-frequency ARI/AMI/VI matrices, permutation p-values, event/calm window analysis, TOD seasonality, calendar-window robustness, robustness sweeps (K, window scale), timeline and rolling ARI visualizations, HMM dual-model comparison
- **Business Impact Function (`impact_fn`)**: User-defined callback `(labels, prices) -> float` on both RepValidator and ResValidator; computes pairwise impact delta matrix across representations/frequencies
- **Disagreement Attribution**: Leave-one-out factor attribution (rep), frequency-pair decomposition (res), temporal hotspot detection with per-day ARI timeline
- **Continuous Monitoring**: `monitor()` function with `init`/`incremental` modes, persistent `monitoring_history.csv`, configurable alert thresholds (`alert_ari_below`, `alert_ari_delta`), file-based alerts (`alerts.json`), webhook support
- **SR 11-7 Report**: New LaTeX template compliant with SR 11-7 / OCC 2011-12; auto-generated findings with severity classification (Critical/High/Medium/Low/Informational); user-fillable overrides via YAML; executive summary with risk rating; effective challenge evidence section
- **Findings Engine**: `findings.py` with `Finding` dataclass, `classify_severity()`, `generate_findings()`, `overall_risk_rating()`, YAML override merging

## 0.1.0 (2026-03-22) — First public release

- **Data infrastructure**: IB download for all frequencies (5m/15m/1h/1d), incremental updates, OHLCV validation and resampling
- **Factor engine**: 7 built-in risk factors (volatility, drawdown, max drawdown, VaR, CVaR, realized skew, stability) with user-extensible registry
- **Model registry**: GMM and HMM built-in, user-extensible via `register_model()`
- **Normalization**: Rolling z-score and min-max
- **Validator: Representation Invariance (rep)**: Pairwise ARI across multiple factor sets, per-asset heatmaps, JSON output
- **Pipeline**: Standard 6-step flow (load → factors → fit → validate → report), each step replaceable
- **Two-layer verdict**: Partition stability (ARI ≥ 0.65) + ordering stability (Spearman ≥ 0.85)
- **Report generation**: JSON → LaTeX → PDF pipeline with template conditionals, dashboard, heatmaps, and remediation plan
- **CLI**: `python run.py run`, `python run.py download`, `python run.py report`
- **Configuration**: Single `config.yaml` with download, logging, factors, normalization, and validator sections
- **Logging**: Console + optional timestamped file logging
- **Security**: IB host/port validation, non-positive price detection
