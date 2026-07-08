# mrv-lib: Model Risk Validator

[![CI](https://github.com/modelguard-lab/mrv-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/modelguard-lab/mrv-lib/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mrv-lib)](https://pypi.org/project/mrv-lib/)
[![Python](https://img.shields.io/pypi/pyversions/mrv-lib)](https://pypi.org/project/mrv-lib/)

**Your model might be producing different outputs depending on which features you feed it, which seed you use, or how you bin the data: and your current validation doesn't catch this.** mrv-lib tests whether your model outputs are stable across admissible specification choices, or silently depend on arbitrary modelling decisions.

mrv is a **pure validation library**: you supply labels from your own models, mrv measures how stable they are. Bank model risk management (OCC Bulletin 2026-13 -- the 2026-04-17 Revised Model Risk Management Guidance that supersedes SR 11-7) is the anchor application; the same tests deploy equally to **production ML monitoring** (route to fallback or human-in-the-loop when labels are unstable, regardless of domain).

## What it does

| Test | Question | Status |
| ---- | -------- | ------ |
| **Representation Invariance** | Do labels change when you use different feature representations? | v0.1.0 |
| **Resolution Invariance** | Do labels agree across 5m / 15m / 1h / 1d frequencies? | v0.2.1 |

Also includes: a business impact function (`impact_fn`), disagreement attribution (LOO / frequency-pair / temporal), and a specification-invariance report generator (`report()`: result JSON to LaTeX to PDF) covering both the representation and resolution tests.

## Install

```bash
pip install mrv-lib
```

## Quick start

The recommended public Python API lives at the top level of the `mrv` package.
You supply a `model_fn` (features to integer labels) plus the admissible set of
specifications, and mrv returns a typed result. mrv only measures agreement; it
never fits a model itself.

Representation invariance (Paper 1) across feature representations:

```python
import numpy as np
import mrv

rng = np.random.default_rng(42)
n = 200
base = rng.integers(0, 3, n)
labels_a = base.copy()
labels_b = base.copy()
flip = rng.random(n) < 0.10
labels_b[flip] = rng.integers(0, 3, flip.sum())   # small perturbation
returns = rng.standard_normal(n) * 0.01

result = mrv.rep_invariance_validator(
    model_fn=lambda x: x,   # passthrough: supply pre-computed labels directly
    admissible_class={"vol+dd+var": labels_a, "vol+var+cvar": labels_b},
    returns=returns,        # optional: enables the Spearman ordering check
    K=3,                    # number of regime states
)
print(result.summary())
print("mean ARI:", result.mean_ari["asset"])
print("partition passes:", result.passes_partition["asset"])
```

Already fit your own regime model? Wrap the labels with the passthrough
`model_fn=lambda x: x` as above, or pass a real callable that maps a feature
matrix to integer labels.

Resolution invariance (Paper 2) across frequencies:

```python
import pandas as pd
import numpy as np
import mrv

rng = np.random.default_rng(0)
idx = pd.date_range("2026-01-05 09:30", periods=480, freq="5min",
                    tz="America/New_York")
labels_5m  = pd.Series(rng.integers(0, 2, 480), index=idx, dtype=int)
labels_15m = labels_5m.iloc[::3].copy()

result = mrv.res_invariance_validator(
    model_fn=lambda s: s,   # passthrough: supply pre-computed labels per frequency
    resolution_set={"SPY": {"5m": labels_5m, "15m": labels_15m}},
    spec=mrv.ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "15m")),
    run_permutation=False,
)
print(result.summary())
print(result.ari_matrix["SPY"].round(3))
print("overall mean ARI:", result.overall_mean_ari["SPY"])
```

The typed results (`RepInvarianceResult` / `ResInvarianceResult`) expose
`.summary()` plus attributes such as `.ari_matrix`, `.overall_mean_ari`,
`.passes_partition`, and `.intraday_overall_ari_gap`. To feed a real model, pass
a `model_fn` that fits your regime model and returns integer labels. See
`examples/paper1_representation_invariance.ipynb` and
`examples/paper2_resolution_invariance.ipynb` for end-to-end walkthroughs.

## Logging

mrv-lib uses Python's standard `logging` module with hierarchical names
(`mrv.validator.rep`, `mrv.validator.res`, etc.). By default nothing is emitted.

```python
import logging

# Show all mrv INFO+ messages
logging.basicConfig(level=logging.INFO)

# Show DEBUG for the representation validator only
logging.getLogger("mrv.validator.rep").setLevel(logging.DEBUG)

# Route mrv logs to a file
handler = logging.FileHandler("mrv_run.log")
logging.getLogger("mrv").addHandler(handler)
```

See `src/mrv/utils/log.py` and `src/mrv/default_config.yaml` for the YAML-based
logging configuration used by the convenience pipeline.

## Project layout

```text
mrv-lib/
|-- config.yaml              # Configuration (for convenience pipeline)
|-- examples/
|   |-- quickstart.ipynb
|   |-- paper1_representation_invariance.ipynb
|   |-- paper2_resolution_invariance.ipynb
|   `-- example_california_housing.ipynb
|-- src/mrv/
|   |-- invariance/          # Recommended public Python API + typed results (rep, res)
|   |-- pipeline.py          # Internal labels-first backend behind the `mrv` CLI
|   |-- data/                # Data modules (optional)
|   |   |-- reader.py        # CSV / OHLCV loading
|   |   |-- factors.py       # Factor / feature engineering
|   |   |-- normalize.py     # Normalization (rolling z-score, minmax)
|   |   `-- download_yahoo.py # Yahoo Finance data download
|   |-- models/              # GMM/HMM fitting
|   |-- templates/
|   |   `-- template.tex     # Specification-invariance report template (rep + res)
|   |-- validator/
|   |   |-- base.py          # BaseValidator (subclass for custom tests)
|   |   |-- rep.py           # Representation Invariance (Paper 1)
|   |   |-- res.py           # Resolution Invariance (Paper 2)
|   |   |-- metrics.py       # ARI, AMI, NMI, Spearman, VI
|   |   |-- attribution.py   # LOO, frequency-pair, temporal hotspots
|   |   `-- report.py        # JSON -> LaTeX -> PDF
|   `-- utils/
|       |-- config.py        # YAML config loading
|       |-- download_ib.py   # IB data download
|       `-- log.py           # Logging setup
|-- reports/                  # Output (gitignored)
`-- tests/
```

## Output

Each run creates a timestamped directory under `reports/`:

- **result.json** -- Complete data (reusable for report regeneration)
- **report.pdf** -- Report with cover page, dashboard, heatmaps, and remediation plan
- **summary.txt** -- Plain text quick view
- **{asset}_ari_heatmap.png** -- ARI heatmap per asset
- **{asset}_timeline.png** -- Regime timeline (res validator)
- **pipeline_summary.csv** -- Summary metrics per asset

## Research

Based on the following PhD research:

- Zheng, Low & Wang (2026). *Regime Labels Are Not Representation-Invariant* (Paper 1). Finance Research Letters.
- Zheng, Low & Wang (2026). *Regime Labels Are Not Resolution-Invariant* (Paper 2). Finance Research Letters.

## License

Dual-licensed.

- **Open source:** GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See [LICENSE](LICENSE). Free for academic research, teaching, and personal use. Note that the AGPL's network-use clause requires any modified version offered over a network to also offer its complete source.
- **Commercial:** Organizations that wish to use mrv-lib in proprietary or closed-source systems, or otherwise cannot meet the AGPL obligations, require a separate commercial license. See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

## Maintainers

[ModelGuard Lab](https://github.com/modelguard-lab) -- Author: Kai Zheng.
