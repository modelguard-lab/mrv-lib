# mrv-lib: Market Regime Validity Library

**The Gold Standard for Model Risk Diagnostics in Non-Stationary Markets.**

mrv-lib is an open-source Python library designed to quantify and diagnose the stability of market regime identification models. Built upon the theoretical framework of **Inference Collapse** and **Ordinal Robustness**, it provides financial institutions with a rigorous toolset to meet Basel IV and SR 11-7 model risk governance requirements.

## Why mrv-lib?

Traditional market regime models often suffer from **"Stability Illusions."** A model may appear robust at daily resolutions but fail to capture structural shifts during high-frequency intraday stress events. mrv-lib exposes these vulnerabilities by measuring:

- **Representation Sensitivity:** How sensitive are your regime labels to feature engineering and preprocessing?
- **Resolution Dissonance:** Does your model's daily output contradict its high-frequency signals?
- **Identifiability Boundaries:** Is the market currently in a "Zone of Collapse" where absolute labels are mathematically unreliable?

## Key Features

### 1. `mrv_lib.scan` — Sensitivity Diagnostic

Automated stress-testing of regime labels across multiple feature sets (Representation) and temporal scales (Resolution). It calculates the **RSS (Representation Stability Score)** to quantify model robustness.

### 2. `mrv_lib.boundary` — Identifiability Index

Calculates the **Identifiability Index** (\(\mathcal{I}\)) based on structural drift and regime separation. It identifies the "Phase Boundaries" where model inference begins to collapse.

### 3. `mrv_lib.metrics` — Ordinal Robustness

When absolute labels (ARI) collapse, mrv-lib measures the **Ordinal Consistency** (Spearman's Rho) to determine if the risk ranking remains valid for fail-safe hedging.

## Installation

```bash
pip install mrv-lib
```

## Quick Start

```python
import mrv_lib as mrv
import pandas as pd

# Load your market data (OHLCV)
data = pd.read_csv("market_data.csv")

# Initialize the diagnostic scanner
scanner = mrv.Scanner(resolution=['5m', '1h', '1d'])

# Run representation stability test
results = scanner.run_representation_test(data, model="HMM")

# Get the RSS (Representation Stability Score)
print(f"Model RSS: {results.rss_score}")

# Detect Identifiability Boundaries
boundary = mrv.detect_boundary(data)
if boundary.is_collapsed:
    print(f"Warning: Entering Inference Collapse Zone. Identifiability Index: {boundary.index}")
```

## Theoretical Foundation

The methodology of mrv-lib is documented in a series of peer-reviewed research papers:

- **Regime Labels Are Not Representation-Invariant:** Evidence of instability across feature sets.
- **Regime Labels Are Not Resolution-Invariant:** Documentation of the 14-hour lag in daily risk reporting.
- **Inference Collapse and Ordinal Robustness:** Defining the phase boundaries of market state identification.

For academic citations, please refer to our [Documentation](https://github.com/your-org/mrv-lib#documentation).

## Commercial Support & SaaS

For enterprise-grade features including real-time alerting, Basel IV Compliance Reporting, and the Fail-Safe Actuator engine, please visit [ModelGuard.co.nz](https://modelguard.co.nz).

- **MRV-Sentinel:** Real-time monitoring for institutional trading desks.
- **ModelGuard Advisory:** Professional consulting for RBNZ/APRA regulatory alignment.

## Maintainers

Maintained by **ModelGuard Lab**. Lead Architect: **Kai Zheng**.

## License

mrv-lib is released under the **MIT License**. See [LICENSE](LICENSE) for details.
