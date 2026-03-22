# mrv-lib: Model Risk Validator

[![CI](https://github.com/modelguard-lab/mrv-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/modelguard-lab/mrv-lib/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mrv-lib)](https://pypi.org/project/mrv-lib/)
[![Python](https://img.shields.io/pypi/pyversions/mrv-lib)](https://pypi.org/project/mrv-lib/)

**Your regime model might be producing different risk labels depending on which features you feed it — and your current validation doesn't catch this.** Most regime models (GMM, HMM) are validated on in-sample fit, but nobody checks whether the labels survive a change in input representation. If they don't, downstream risk decisions are sitting on an arbitrary modelling choice. mrv-lib tests for this.

mrv tests whether your market regime model produces **stable, reliable labels** — or whether they silently depend on undisclosed modelling choices like feature selection, temporal resolution, or rolling window parameters. Built for SR 11-7 / Basel IV model risk governance.

## What it does

| Test | Question | Status |
| ---- | -------- | ------ |
| **Representation Invariance** | Do regime labels change when you use different risk factors? | v0.1.0 |
| **Resolution Invariance** | Do labels agree across 5m / 1h / 1d frequencies? | v0.2.0 |
| **Temporal Stability** | Do labels persist across rolling windows? | v0.3.0 |

## Install

```bash
pip install mrv-lib

# Optional: IB data download
pip install mrv-lib[ib]

# Optional: regime models (GMM/HMM)
pip install mrv-lib[validator]

# Everything
pip install mrv-lib[all]
```

## Quick start

> **Notebook:** See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a step-by-step walkthrough with synthetic data — no IB connection needed.

```bash
# 1. Download data (requires IB Gateway running)
python run.py download config.yaml

# 2. Run representation invariance test + generate PDF
python run.py run config.yaml rep

# 3. Regenerate PDF from existing results
python run.py report
```

Or from Python:

```python
from mrv.pipeline import run, download

download("config.yaml")             # fetch data from IB
run("config.yaml", "rep")           # validate + PDF

# Step by step (full control)
from mrv.pipeline import load_data, compute_factors, fit_labels, validate, report

cfg = load("config.yaml")
prices = load_data(cfg, "rep")
# ... user can replace any step
```

## Configuration

All settings in one `config.yaml`:

```yaml
download:
  data_dir: data
  symbols: [SPY, USDJPY, CL=F, IEF, GLD]
  freq: [5m, 15m, 1h, 1d]
  start: "2026-01-01"
  ib:
    host: 127.0.0.1
    port: 4002

validator:
  rep:
    assets:
      SPY: data/SPY_5m.csv
      CL:  data/CL_5m.csv
    model: gmm
    factors:
      - [vol, drawdown, maxdd, var, cvar]
      - [vol, drawdown, var, cvar]
      - [real_skew, vol_stab, var, cvar]
```

## Custom factors and models

```python
from mrv.data.factors import register_factor

def momentum(returns, price, windows):
    return price.pct_change(windows.get("mom_window", 20)).rename("momentum")

register_factor("momentum", momentum)
```

```python
from mrv.models import register_model

def my_kmeans(features, n_states, **kwargs):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_states).fit_predict(features.values)

register_model("kmeans", my_kmeans)
```

## Project layout

```text
mrv-lib/
├── config.yaml              # Configuration
├── run.py                   # CLI entry point
├── templates/
│   └── template.tex         # LaTeX report template
├── src/mrv/
│   ├── pipeline.py          # data → factors → model → validate → report
│   ├── data/
│   │   ├── reader.py        # Load, validate, resample OHLCV
│   │   ├── factors.py       # Factor registry + built-in risk factors
│   │   └── normalize.py     # Rolling z-score, min-max
│   ├── models/
│   │   ├── __init__.py      # Model registry + fit()
│   │   ├── gmm.py           # Gaussian Mixture Model
│   │   └── hmm.py           # Hidden Markov Model
│   ├── validator/
│   │   ├── base.py          # BaseValidator (subclass for custom tests)
│   │   ├── rep.py           # Representation Invariance test
│   │   ├── metrics.py       # ARI, AMI, NMI, Spearman, VI
│   │   └── report.py        # JSON → LaTeX → PDF
│   └── utils/
│       ├── config.py        # YAML config loading
│       ├── download.py      # IB data download
│       └── log.py           # Logging setup
├── reports/                  # Output (gitignored)
│   └── mrv_report_YYYYMMDD_rep/
│       ├── result.json
│       ├── report.pdf
│       └── {asset}.png
└── tests/
```

## Output

Each run creates a timestamped directory under `reports/`:

- **result.json** — Complete data (reusable for report regeneration)
- **report.pdf** — Professional report with cover page, dashboard, heatmaps, and remediation plan
- **summary.txt** — Plain text quick view
- **{asset}.png** — ARI heatmap per asset

## Research

Based on the following PhD research:

- Zheng, Low & Wang (2026). *Regime Labels Are Not Representation-Invariant*. Submitted.
- Zheng, Low & Wang (2026). *Regime Labels Are Not Resolution-Invariant*. In preparation.

## License

MIT. See [LICENSE](LICENSE).

## Maintainers

[ModelGuard Lab](https://github.com/modelguard-lab) — Author: Kai Zheng.
