# mrv-lib Public API Contract

Version: 0.7.0 (unreleased; last PyPI release 0.6.1)
Stability: **Stable** -- every symbol listed here follows the Semantic Versioning
contract below. 0.7.0 carries one breaking change: the
`ResInvarianceResult.within_intraday_excess` attribute is renamed to
`intraday_overall_ari_gap` (see the Removed / renamed table and the
`[0.7.0]` section of `CHANGELOG.md`).

---

## Semantic Versioning Contract

| Change type | Version bump |
|---|---|
| Add a new public function or class (kw-only params only) | MINOR |
| Add a new optional keyword-only parameter to an existing function | MINOR |
| Rename, remove, or reorder a public function or parameter | MAJOR |
| Change a required parameter to optional or vice versa | MAJOR |
| Change a return type's shape (e.g. dict key removed or type changed) | MAJOR |
| Bug fix that does not change the public signature | PATCH |
| Documentation, type hints, docstrings only | PATCH |

The library follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Every code-touching release adds an entry under `## [Unreleased]` in `CHANGELOG.md`
before the version tag is cut.

---

## The public Python API: top-level namespace (`mrv`)

The recommended public Python contract is the top-level functional API. Import
`mrv` and call the invariance validators directly; they return typed result
objects.

```python
import mrv
```

| Symbol | Kind | Since |
|---|---|---|
| `mrv.__version__` | `str` | 0.1.0 |
| `mrv.rep_invariance_validator` | function | 0.6.0 |
| `mrv.res_invariance_validator` | function | 0.6.0 |
| `mrv.report` | function | 0.2.0 |
| `mrv.ResolutionSpec` | dataclass | 0.6.0 |
| `mrv.RepInvarianceResult` | dataclass | 0.6.0 |
| `mrv.ResInvarianceResult` | dataclass | 0.6.0 |
| `mrv.PAPER2_FREQS` | `tuple[str, ...]` | 0.6.0 |
| `mrv.PAPER2_INTRADAY_FREQS` | `tuple[str, ...]` | 0.6.0 |
| `mrv.MrvError` | exception (base for all mrv errors) | 0.7.0 |
| `mrv.MrvValidationError` | exception (invalid validator input; also `ValueError`) | 0.7.0 |
| `mrv.MrvConfigError` | exception (invalid configuration; also `ValueError`) | 0.7.0 |

These same symbols are also re-exported from `mrv.invariance`.

### Exception hierarchy

`MrvError` is the base class for all library-raised errors, so downstream code
can `except mrv.MrvError` to catch them in one place. The concrete subclasses
also inherit from the builtin they replace (`MrvValidationError` and
`MrvConfigError` are both `ValueError`), so existing `except ValueError` code
keeps working -- adding the hierarchy is backward-compatible.

### `rep_invariance_validator(model_fn, admissible_class, ...) -> RepInvarianceResult`

Paper 1 representation-invariance check.

Parameters (stable):
- `model_fn: Callable[[np.ndarray], np.ndarray]`
- `admissible_class: dict[str, np.ndarray]` -- at least 2 specifications.
- `returns: np.ndarray = None`
- `K: int = 2`

To pass pre-computed labels directly, use a passthrough model:
`model_fn=lambda x: x` with the labels as `admissible_class` values.

### `res_invariance_validator(model_fn, resolution_set, ...) -> ResInvarianceResult`

Paper 2 resolution-invariance check.

Parameters (stable):
- `model_fn: Callable[[pd.Series], pd.Series]`
- `resolution_set: dict[str, dict[str, pd.Series]]`
- `spec: ResolutionSpec = None`
- `run_permutation: bool = True`
- `n_perm: int = 500`
- `seed: int = 42`

To pass pre-computed labels directly, use a passthrough model:
`model_fn=lambda s: s` with pre-labelled Series as `resolution_set` values.

### `report(json_path, template=None, cfg=None) -> Path | None`

Renders a representation or resolution result JSON to a report. The `.tex` is
always written; the PDF is compiled only when `pdflatex` is available on `PATH`.

### `ResolutionSpec` dataclass

- `freqs: tuple[str, ...] = PAPER2_FREQS`
- `intraday_freqs: tuple[str, ...] = None` (defaults to all non-"1d" freqs)

### `RepInvarianceResult` dataclass

Typed representation-invariance result. Per-asset attributes are dicts keyed by
asset name. Full public attribute contract:

- `ari_per_pair: dict[str, dict[tuple[str, str], float]]`
- `ordering_per_pair: dict[str, dict[tuple[str, str], float]]`
- `mean_ari: dict[str, float]`
- `min_ari: dict[str, float]`
- `null_1_over_K: float`
- `K: int`
- `ari_threshold: float`
- `spearman_threshold: float`
- `passes_partition: dict[str, bool]`
- `passes_ordering: dict[str, bool]`
- `summary() -> str`

### `ResInvarianceResult` dataclass

Typed resolution-invariance result. Per-asset attributes are dicts keyed by
asset name. Full public attribute contract:

- `ari_matrix: dict[str, pd.DataFrame]`
- `ami_matrix: dict[str, pd.DataFrame]`
- `overall_mean_ari: dict[str, float | None]`
- `intraday_mean_ari: dict[str, float | None]`
- `intraday_overall_ari_gap: dict[str, float | None]`
- `passes_partition: dict[str, bool]`
- `ari_threshold: float`
- `freqs: tuple[str, ...]`
- `intraday_freqs: tuple[str, ...]`
- `perm_pvalue: dict[str, float | None]`
- `perm_null_ci: dict[str, tuple[float, float] | None]`
- `summary() -> str`

---

## Internal / CLI backend (not the recommended user API; may change)

The symbols below remain importable and power the `mrv run` config-file
workflow, but they are **not** the recommended public Python API. Prefer the
top-level validators above. These labels-first entry points and validator
classes are documented here for completeness and may change without a MAJOR
bump.

### `mrv.pipeline` -- pipeline convenience API (CLI backend)

```python
from mrv.pipeline import validate_rep, validate_res, report, run, download, validate
```

- `validate_rep(labels, risk_proxy=None, prices=None, cfg=None, impact_fn=None) -> dict`
- `validate_res(labels, event_window=None, calm_window=None, cfg=None, impact_fn=None) -> dict`
- `report(json_path, template=None, cfg=None) -> Path | None`
- `download(config=None, cfg=None) -> dict`
- `run(config=None, validator="rep", cfg=None, impact_fn=None) -> Path | None`
- `validate(config=None, validator="rep", cfg=None, impact_fn=None) -> dict` -- dispatches a `rep` convenience run; the backend for `monitor()`. `validator="res"` raises: resolution invariance is labels-first and has no convenience/monitoring path (call `validate_res(labels=...)` directly).

### `mrv.validator` -- validator classes (CLI backend)

```python
from mrv.validator import BaseValidator, RepValidator, ResValidator
from mrv.validator import generate_report
```

- `RepValidator.validate(labels, risk_proxy=None, prices=None) -> dict`
- `ResValidator.validate(labels, event_window=None, calm_window=None) -> dict`
- `generate_report(json_path, template=None, cfg=None) -> Path | None` -- renders both representation and resolution result JSON (backs `mrv.report`).

---

## `mrv.data` -- Data Loading and Factors

```python
from mrv.data import load_daily, load_ohlcv, resample_ohlc, validate_ohlcv
from mrv.data import normalize, rolling_zscore, minmax
from mrv.data import build_factors, register_factor, log_returns
```

### Data loading

- `load_daily(path, price_col=None) -> pd.Series`
- `load_ohlcv(path, tz="America/New_York") -> pd.DataFrame`
- `validate_ohlcv(df, symbol="") -> list[str]`
- `resample_ohlc(df, freq, tz="America/New_York") -> pd.DataFrame`

### Normalisation

- `rolling_zscore(df, window=120) -> pd.DataFrame`
- `minmax(df, window=120) -> pd.DataFrame`
- `normalize(df, mode=None, window=None, cfg=None) -> pd.DataFrame`

### Factor builder

- `log_returns(price) -> pd.Series`
- `build_factors(price, factors=None, windows=None, cfg=None) -> pd.DataFrame`
- `register_factor(name, fn) -> None`

Built-in factor names: `"volatility"`, `"drawdown"`, `"max_drawdown_window"`,
`"var"`, `"cvar"`, `"realized_skew"`, `"stability"`.
Aliases: `"vol"`, `"maxdd"`, `"real_skew"`, `"vol_stab"`.

---

## `mrv.models` -- Regime Model Registry

```python
from mrv.models import fit, register_model
```

- `fit(features, model="gmm", n_states=3, **kwargs) -> np.ndarray | None`
- `register_model(name, fn) -> None`

Built-in models: `"gmm"`, `"hmm"`.

---

## `mrv.utils` -- Utilities

```python
from mrv.utils import load, get_data_dir, get_assets, setup_logging
```

- `load(path=None) -> dict`
- `get_data_dir(cfg, base=None) -> Path`
- `get_assets(cfg, freq=None) -> list[dict]`
- `setup_logging(cfg=None) -> None`

---

## Removed / renamed symbols

| Symbol | Removed / renamed in | Use instead |
|---|---|---|
| `ResInvarianceResult.within_intraday_excess` (attribute) | renamed 0.7.0 | `ResInvarianceResult.intraday_overall_ari_gap` |
| `mrv.sr26_2_report`, `mrv.pipeline.sr26_2_report` | removed 0.6.1 | `report` / `generate_report` |
| `mrv.validator.generate_sr26_2_report` | removed 0.6.1 | `generate_report` |
| `mrv.sr11_7_report`, `mrv.pipeline.sr11_7_report` | removed 0.6.1 | `report` / `generate_report` |
| `mrv.validator.generate_sr11_7_report` | removed 0.6.1 | `generate_report` |

---

## Non-public surface

The following are **internal implementation details** and carry no stability guarantee.
They may change or be removed in any release:

- `mrv.validator.base._compute_impact_matrix`
- `mrv.validator.rep._write_text_report`
- `mrv.validator.res._plot_timeline`, `_resolve_index_subset`, `_aligned_pair_values`
- `mrv.data.factors._REGISTRY`, `_ALIASES`, `DEFAULT_FACTORS`
- All `_`-prefixed functions and variables in any module.
