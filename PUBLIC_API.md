# mrv-lib Public API Contract

Version: 0.6.1
Stability: **Stable** -- all symbols listed here follow the Semantic Versioning contract below.

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

## Top-level namespace (`mrv`)

```python
import mrv
```

| Symbol | Kind | Since |
|---|---|---|
| `mrv.__version__` | `str` | 0.1.0 |
| `mrv.report` | function | 0.2.0 |
| `mrv.RepInvarianceResult` | dataclass | 0.6.0 |
| `mrv.rep_invariance_validator` | function | 0.6.0 |
| `mrv.ResInvarianceResult` | dataclass | 0.6.0 |
| `mrv.ResolutionSpec` | dataclass | 0.6.0 |
| `mrv.res_invariance_validator` | function | 0.6.0 |
| `mrv.PAPER2_FREQS` | `tuple[str, ...]` | 0.6.0 |
| `mrv.PAPER2_INTRADAY_FREQS` | `tuple[str, ...]` | 0.6.0 |

---

## `mrv.invariance` -- Invariance API

```python
from mrv.invariance import (
    rep_invariance_validator, RepInvarianceResult,
    res_invariance_validator, ResInvarianceResult, ResolutionSpec,
    PAPER2_FREQS, PAPER2_INTRADAY_FREQS,
)
```

### `rep_invariance_validator(model_fn, admissible_class, ...) -> RepInvarianceResult`

Paper 1 representation-invariance check.

Parameters (stable):
- `model_fn: Callable[[np.ndarray], np.ndarray]`
- `admissible_class: dict[str, np.ndarray]` -- at least 2 specifications.
- `returns: np.ndarray = None`
- `K: int = 2`

### `res_invariance_validator(model_fn, resolution_set, ...) -> ResInvarianceResult`

Paper 2 resolution-invariance check.

Parameters (stable):
- `model_fn: Callable[[pd.Series], pd.Series]`
- `resolution_set: dict[str, dict[str, pd.Series]]`
- `spec: ResolutionSpec = None`
- `run_permutation: bool = True`
- `n_perm: int = 500`
- `seed: int = 42`

### `ResolutionSpec` dataclass

- `freqs: tuple[str, ...] = PAPER2_FREQS`
- `intraday_freqs: tuple[str, ...] = None` (defaults to all non-"1d" freqs)

---

## `mrv.validator` -- Validators

```python
from mrv.validator import BaseValidator, RepValidator, ResValidator
from mrv.validator import generate_report
```

### `RepValidator.validate(labels, risk_proxy=None, prices=None) -> dict`

### `ResValidator.validate(labels, event_window=None, calm_window=None) -> dict`

### `generate_report(json_path, template=None, cfg=None) -> Path | None`

Renders both representation and resolution result JSON. The `.tex` is always
written; the PDF is compiled only when `pdflatex` is available on `PATH`.

---

## `mrv.pipeline` -- Pipeline Convenience API

```python
from mrv.pipeline import validate_rep, validate_res, report, run, download, validate
```

### `validate_rep(labels, risk_proxy=None, prices=None, cfg=None, impact_fn=None) -> dict`

### `validate_res(labels, event_window=None, calm_window=None, cfg=None, impact_fn=None) -> dict`

### `report(json_path, template=None, cfg=None) -> Path | None`

### `download(config=None, cfg=None) -> dict`

### `run(config=None, validator="rep", cfg=None, impact_fn=None) -> Path | None`

### `validate(config=None, validator="rep", cfg=None, impact_fn=None) -> dict`

Dispatches a convenience validation run by name; used internally by `monitor()`.

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

## Removed symbols

| Symbol | Removed in | Use instead |
|---|---|---|
| `mrv.sr26_2_report`, `mrv.pipeline.sr26_2_report` | 0.6.1 | `report` / `generate_report` |
| `mrv.validator.generate_sr26_2_report` | 0.6.1 | `generate_report` |
| `mrv.sr11_7_report`, `mrv.pipeline.sr11_7_report` | 0.6.1 | `report` / `generate_report` |
| `mrv.validator.generate_sr11_7_report` | 0.6.1 | `generate_report` |

---

## Non-public surface

The following are **internal implementation details** and carry no stability guarantee.
They may change or be removed in any release:

- `mrv.validator.base._compute_impact_matrix`
- `mrv.validator.rep._write_text_report`
- `mrv.validator.res._plot_timeline`, `_resolve_index_subset`, `_aligned_pair_values`
- `mrv.data.factors._REGISTRY`, `_ALIASES`, `DEFAULT_FACTORS`
- All `_`-prefixed functions and variables in any module.
