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
| **Resolution Invariance** | Do labels agree across 5m / 15m / 1h / 1d frequencies? | v0.2.1 |
| **MRI — Model Risk Index** | A single score combining rep + res into actionable governance signal | v0.3.0 |

v0.2.1 also includes: business impact function (`impact_fn`), continuous monitoring with alerts, disagreement attribution (LOO / frequency-pair / temporal), SR 11-7 compliant report with auto-generated findings, and a findings engine with severity classification.

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

> **Notebooks:** See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a quick overview, or [`examples/paper1_representation_invariance.ipynb`](examples/paper1_representation_invariance.ipynb) and [`examples/paper2_resolution_invariance.ipynb`](examples/paper2_resolution_invariance.ipynb) for step-by-step walkthroughs.

```bash
# 1. Download data (requires IB Gateway running)
python run.py download config.yaml

# 2. Run representation invariance test + generate PDF
python run.py run config.yaml rep

# 3. Run resolution invariance test + generate PDF
python run.py run config.yaml res

# 4. Regenerate PDF from existing results
python run.py report
```

Or from Python:

```python
from mrv.pipeline import run, download, validate, report

download("config.yaml")             # fetch data from IB
run("config.yaml", "rep")           # validate + PDF
run("config.yaml", "res")           # validate + PDF

# Step by step (full control)
from mrv.pipeline import load_data, compute_factors, fit_labels, validate, report
from mrv.utils.config import load

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

factors:
  vol_window: 20
  drawdown_window: 60
  tail_window: 60
  tail_alpha: 0.05

validator:
  rep:
    assets:
      SPY: data/SPY_1d.csv
    model: gmm
    n_states: 3
    factors:
      - [vol, drawdown, maxdd, var, cvar]
      - [vol, drawdown, var, cvar]
      - [real_skew, vol_stab, var, cvar]

  res:
    assets:
      SPY: [data/SPY_5m.csv]
    model: gmm
    n_states: 2
    episode: 2026_iran
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
│   ├── template.tex         # Academic report template
│   └── sr11_7_template.tex  # SR 11-7 regulatory report template
├── examples/
│   ├── quickstart.ipynb
│   ├── paper1_representation_invariance.ipynb
│   └── paper2_resolution_invariance.ipynb
├── src/mrv/
│   ├── pipeline.py          # data -> factors -> model -> validate -> report
│   ├── data/
│   │   ├── reader.py        # Load, validate, resample OHLCV
│   │   ├── factors.py       # Factor registry + 7 built-in risk factors
│   │   └── normalize.py     # Rolling z-score, min-max
│   ├── models/
│   │   ├── __init__.py      # Model registry + fit()
│   │   ├── gmm.py           # Gaussian Mixture Model
│   │   └── hmm.py           # Hidden Markov Model
│   ├── validator/
│   │   ├── base.py          # BaseValidator (subclass for custom tests)
│   │   ├── rep.py           # Representation Invariance test (Paper 1)
│   │   ├── res.py           # Resolution Invariance test (Paper 2)
│   │   ├── metrics.py       # ARI, AMI, NMI, Spearman, VI
│   │   ├── attribution.py   # LOO, frequency-pair, temporal hotspots
│   │   ├── findings.py      # SR 11-7 findings engine
│   │   ├── monitor.py       # Continuous monitoring + alerts
│   │   └── report.py        # JSON -> LaTeX -> PDF
│   └── utils/
│       ├── config.py        # YAML config loading
│       ├── download.py      # IB data download
│       └── log.py           # Logging setup
├── reports/                  # Output (gitignored)
└── tests/                    # 123 tests
```

## Output

Each run creates a timestamped directory under `reports/`:

- **result.json** -- Complete data (reusable for report regeneration)
- **report.pdf** -- Professional report with cover page, dashboard, heatmaps, and remediation plan
- **summary.txt** -- Plain text quick view
- **{asset}_ari_heatmap.png** -- ARI heatmap per asset
- **{asset}_timeline.png** -- Regime timeline (res validator)
- **pipeline_summary.csv** -- Summary metrics per asset

## Research

Based on the following PhD research:

- Zheng, Low & Wang (2026). *Regime Labels Are Not Representation-Invariant*. Submitted.
- Zheng, Low & Wang (2026). *Regime Labels Are Not Resolution-Invariant*. Submitted to Finance Research Letters.

## License

MIT. See [LICENSE](LICENSE).

## Maintainers

[ModelGuard Lab](https://github.com/modelguard-lab) -- Author: Kai Zheng.
