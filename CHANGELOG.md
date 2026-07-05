# Changelog

All notable changes to mrv-lib are documented here.
The project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Semantic Versioning](https://semver.org/).

## [Unreleased]

(no changes yet)

## [0.6.1] - 2026-07-06

Patch release: report packaging, resolution-path wiring, lint, and docs.

### Fixed

- **Report template now ships in the wheel.** The LaTeX template moved from the
  repo-root `templates/` directory into the package at
  `src/mrv/templates/template.tex` and is declared in
  `[tool.setuptools.package-data]`. `report()` / `generate_report()` resolve it
  via `importlib.resources`, so it works for pip-installed users (previously the
  template was not packaged and report generation failed).
- **`report()` now renders resolution (Paper 2) result JSON.** `_render` /
  `_expand_assets` branch on `data["test"]`; the resolution case is rendered from
  `overall_mean_ari` with the Spearman / factor-set sections omitted. Previously
  a resolution `result.json` raised `KeyError`.
- `RepValidator._build_json`: an exact overall ARI / Spearman of `0.0` now
  serialises as `0.0` instead of `null` (truthiness check replaced with an
  explicit `is not None`).
- `RepInvarianceResult.summary()` now reports the mean of the finite per-pair
  Spearman values (falling back to `n/a` only when none are finite) instead of a
  hardcoded `n/a`.
- `mrv.data.download_yahoo.download` skips a symbol/frequency when a required
  OHLC column is missing (instead of raising `KeyError`) and includes `Volume`
  in the saved CSV when present.

### Removed

- **Retired the SR 26-2 / SR 11-7 governance-report symbols.** `sr26_2_report`,
  `generate_sr26_2_report`, and the deprecated `sr11_7_report` /
  `generate_sr11_7_report` aliases have been removed; they referenced a template
  that no longer ships. Use the generic `report()` / `generate_report()`.
- Removed the `res` subcommand from the CLI. Resolution invariance is
  labels-first: fit your own regime models at each frequency and call
  `validate_res()` from the Python API. The CLI no longer exposes an
  unimplemented convenience path.

### Changed

- `docs/conf.py` now single-sources `version` / `release` from
  `mrv.__version__`.
- Zeroed all `ruff` (E/F/W/B/I) lint errors across `src/` and `tests/`.

## [0.6.0] - 2026-06-22

Minor release consolidating the representation-invariance (Paper 1) and
resolution-invariance (Paper 2) validators behind a stable public API.

### Added

- `mrv.invariance.rep_invariance_validator(model_fn, admissible_class, returns, K)`:
  functional Paper 1 representation-invariance check returning a typed
  `RepInvarianceResult` (per-pair ARI, per-pair Spearman ordering, `1/K` null,
  partition and ordering pass flags).
- `mrv.invariance.res_invariance_validator(model_fn, resolution_set, ...)`:
  functional Paper 2 resolution-invariance check returning a typed
  `ResInvarianceResult` (cross-frequency ARI matrix, AMI matrix,
  within-intraday excess, permutation p-value and null CI).
- `ResolutionSpec` dataclass with the Paper 2 four-frequency panel
  (`5m`, `15m`, `1h`, `1d`) and intraday subset (`5m`, `15m`, `1h`);
  constants `PAPER2_FREQS` and `PAPER2_INTRADAY_FREQS`.
- `PUBLIC_API.md`: semantic-versioning contract and full public-symbol listing.
- `__all__` declarations across the public modules; Google-style docstrings and
  type hints on the public surface.
- Example notebook section demonstrating `rep_invariance_validator` alongside the
  labels-first `RepValidator` API.

### Fixed

- `src/mrv/__init__.py`: `__version__` set to `"0.6.0"`.
- `tests/test_version.py`: version assertion updated to `"0.6.0"`.
- `CITATION.cff`: `version` set to `"0.6.0"`; Paper 1 and Paper 2 statuses updated.

## [0.5.1] - 2026-06-01

Patch release. Backward-compatible additions and stricter input validation.

### Fixed

- Rolling-window index off-by-one corrected; window-boundary values shift by one
  observation step. Callers replaying historical series should re-run.

## [0.5.0] - 2026-05-25

### Changed

- SR 11-7 references migrated to SR 26-2 / OCC Bulletin 2026-13 throughout the
  codebase, templates, examples, and documentation. SR 26-2 (Federal Reserve
  letter) and OCC Bulletin 2026-13, jointly issued 2026-04-17 by the OCC, Federal
  Reserve, and FDIC, supersede SR 11-7 (Federal Reserve, 2011) as the primary US
  model risk management guidance. See
  [OCC Bulletin 2026-13](https://www.occ.treas.gov/news-issuances/bulletins/2026/bulletin-2026-13.html).
- `sr11_7_report` (in `mrv.pipeline`) aliased to the new `sr26_2_report` with a
  `DeprecationWarning` when called.
- `generate_sr11_7_report` (in `mrv.validator.report`) aliased to the new
  `generate_sr26_2_report` with a `DeprecationWarning` when called.
- Version bumped to 0.5.0.

### Added

- `mrv.pipeline.sr26_2_report`: canonical replacement for `sr11_7_report`.
- `mrv.validator.report.generate_sr26_2_report`: canonical replacement for
  `generate_sr11_7_report`.

## [0.4.2] - 2026-05-17

### Changed

- **Relicensed from MIT to AGPL-3.0-or-later with a commercial exception.** The
  open-source license is now GNU AGPL-3.0 (full text in `LICENSE`): free for
  academic, teaching, and personal use, with AGPL copyleft and network-source
  obligations. Organizations needing proprietary or closed-source use that cannot
  meet the AGPL must obtain a separate commercial license; see
  `COMMERCIAL-LICENSE.md`. `pyproject.toml`, `CITATION.cff`, and `README.md`
  updated accordingly.

## [0.2.1] - 2026-03-30

### Fixed

- CI lint errors (unused imports, lambda assignment).
- GitHub Actions upgraded to Node.js 22 (checkout v5, setup-python v6).

## [0.2.0] - 2026-03-30

### Added

- **Resolution Invariance validator (res)**: multi-frequency regime analysis
  (5m/15m/1h/1d), cross-frequency ARI/AMI/VI matrices, permutation p-values,
  event/calm window analysis, time-of-day seasonality, calendar-window
  robustness, robustness sweeps (K, window scale), and timeline / rolling-ARI
  visualizations, with an HMM dual-model comparison.
- **Business impact function (`impact_fn`)**: user-defined callback
  `(labels, prices) -> float` on both `RepValidator` and `ResValidator`;
  computes a pairwise impact-delta matrix across representations/frequencies.
- **Disagreement attribution**: leave-one-out factor attribution (rep),
  frequency-pair decomposition (res), and temporal hotspot detection with a
  per-day ARI timeline.
- **Continuous monitoring**: `monitor()` with `init`/`incremental` modes,
  persistent `monitoring_history.csv`, configurable alert thresholds, file-based
  alerts, and webhook support.
- **Model validation report**: LaTeX template with auto-generated findings and
  severity classification (Critical/High/Medium/Low/Informational),
  user-fillable YAML overrides, executive summary with risk rating, and an
  effective-challenge evidence section.
- **Findings engine**: `findings.py` with a `Finding` dataclass,
  `classify_severity()`, `generate_findings()`, `overall_risk_rating()`, and
  YAML override merging.

## [0.1.0] - 2026-03-22 -- First public release

### Added

- **Data infrastructure**: IB download for all frequencies (5m/15m/1h/1d),
  incremental updates, OHLCV validation and resampling.
- **Factor engine**: seven built-in risk factors (volatility, drawdown, max
  drawdown, VaR, CVaR, realized skew, stability) with a user-extensible registry.
- **Model registry**: GMM and HMM built in, user-extensible via
  `register_model()`.
- **Normalization**: rolling z-score and min-max.
- **Representation Invariance validator (rep)**: pairwise ARI across multiple
  factor sets, per-asset heatmaps, JSON output.
- **Pipeline**: standard load -> factors -> fit -> validate -> report flow, each
  step replaceable.
- **Two-layer verdict**: partition stability (ARI >= 0.65) and ordering
  stability (Spearman >= 0.85).
- **Report generation**: JSON -> LaTeX -> PDF pipeline with template
  conditionals, dashboard, heatmaps, and remediation plan.
- **CLI**: `run`, `download`, `report` subcommands.
- **Configuration**: single `config.yaml` with download, logging, factors,
  normalization, and validator sections.
- **Logging**: console plus optional timestamped file logging.
- **Security**: IB host/port validation, non-positive price detection.
