# mrv-lib Roadmap

## v0.3.0 — MRI Scoring Engine (Paper 3 Integration)

Paper 3: *The Model Risk Index* (Zheng, Low & Wang 2026).

MRI unifies representation invariance (Paper 1) and resolution invariance (Paper 2) into a single actionable score with theory-grounded phase boundaries that replace the empirical ARI threshold.

### Overview

```
                    MRI Pipeline
                    ============
Raw Data (5m OHLCV)
    |
    +-- rep branch: build K factor sets --> fit GMM (K=3) --> pairwise ARI --> RSS_rep
    |
    +-- res branch: resample to 15m/1h/1d --> fit GMM (K=2) per freq --> cross-freq ARI --> RSS_res
    |
    +-- spectral branch: fit HMM at 5m --> transition matrix --> lambda_2, gamma, kappa
    |
    v
MRI = (1 - RSS_rep) * (1 - RSS_res)          [Proposition 3.1: multiplicative]
    |
    +-- classify_zone(MRI, rho_S)  -->  Zone I / II / III
    |
    +-- ordinal_bound(rho_0, K)    -->  violation check (Theorem 3.8)
    |
    v
Governance signal: PASS / WARNING / BREACH
```

### 1. MRI Core Module (`mrv.mri`)

New module `src/mrv/mri/` with:

```
mri/
  __init__.py        # public API: compute_rss, classify_zone, MriResult
  score.py           # MRI computation and zone classification
  spectral.py        # HMM spectral gap, lambda_2, kappa estimation
  wasserstein.py     # sliced Wasserstein distance
  ordinal.py         # ordinal robustness bound g(rho_0, K)
```

#### 1a. MRI Score Computation (`score.py`)

**Formula** (Proposition 3.1 -- multiplicative decomposition):

```python
def compute_mri(rss_rep: float, rss_res: float, rss_cross: float = 0.0) -> float:
    """MRI = (1 - RSS_rep) * (1 - RSS_res) * (1 - RSS_cross), clamped to [0, 1].

    RSS_rep = 1 - mean(ARI across representation pairs)
    RSS_res = 1 - mean(ARI across frequency pairs)
    RSS_cross = 0 (reserved for future cross-term)
    """
```

**Zone classification** (two-dimensional phase space):

```python
def classify_zone(
    mri: float,
    rho_s: float,
    b12: float = 0.80,       # Zone I/II boundary (MRI axis)
    b23: float = 0.80,       # Zone II/III boundary (rho_S axis)
) -> str:
    """
    Zone I  (Stable):    MRI >= b12 AND rho_S >= b23  ->  labels trustworthy
    Zone II (Collapsed): MRI <  b12 AND rho_S >= b23  ->  ordering preserved, partition unreliable
    Zone III (Chaotic):  rho_S < b23                  ->  fully unreliable
    """
```

**Result dataclass:**

```python
@dataclass
class MriResult:
    mri: float                    # composite MRI score [0, 1]
    rss_rep: float                # representation component
    rss_res: float                # resolution component
    zone: str                     # "I", "II", "III"
    rho_s: float                  # Spearman ordering consistency
    signal: str                   # "pass", "warning", "breach"
    spectral: dict                # {lambda_2, gamma, kappa}
    rep_ari_matrix: pd.DataFrame  # cross-rep ARI
    res_ari_matrix: pd.DataFrame  # cross-freq ARI
    ordinal_bound: float          # g(rho_0, K)
    ordinal_violation: bool       # rho_s < ordinal_bound
```

#### 1b. Spectral Analysis (`spectral.py`)

**HMM spectral gap** (Theorem 3.5 -- resolution collapse driver):

```python
def fit_hmm_spectral(X: np.ndarray, K: int = 3, seed: int = 42) -> dict:
    """Fit HMM and extract spectral parameters from transition matrix.

    Returns:
        labels, transmat, lambda_2, gamma (= 1 - lambda_2), kappa, stationary
    """
```

**Resolution collapse bound** (Theorem 3.5):

```python
def resolution_collapse_bound(lambda_2: float, m: int, kappa: float, epsilon_0: float) -> float:
    """Upper bound on ARI loss: (1 - lambda_2^m) / (1 - lambda_2) * kappa * epsilon_0"""
```

**Optimal aggregation m*** (Corollary 3.7):

```python
def optimal_aggregation(lambda_2: float, kappa: float, epsilon_0: float,
                        ari_threshold: float = 0.30) -> int:
    """Find maximum aggregation ratio m* where ARI remains above threshold."""
```

#### 1c. Sliced Wasserstein Distance (`wasserstein.py`)

**Theorem 3.2 -- representation collapse driver:**

```python
def sliced_wasserstein(X: np.ndarray, Y: np.ndarray,
                       n_slices: int = 100, seed: int = 42) -> float:
    """D_SW(X, Y) = mean over random 1D projections of W_1(proj(X), proj(Y))"""

def representation_collapse_ratio(X: np.ndarray, Y: np.ndarray,
                                   local_var: float, n_slices: int = 100) -> float:
    """D^2 / sigma^2_local -- monotonically related to 1 - ARI (Theorem 3.2)."""
```

#### 1d. Ordinal Robustness (`ordinal.py`)

**Theorem 3.8 -- ordinal bound:**

```python
def ordinal_bound(rho_0: float, K: int) -> float:
    """g(rho_0, K) = 1 - (1/K) * sqrt(1 - rho_0^2)"""

def should_use_ordinal_fallback(zone: str, rho_s: float,
                                 rho_0: float, K: int) -> bool:
    """Returns True if Zone II and ordinal bound is satisfied."""
```

---

### 2. MriValidator (`mrv.validator.rss_validator`)

```python
class MriValidator(BaseValidator):
    """Unified MRI validator -- runs rep + res + spectral, computes MRI, classifies zone."""

    def validate(self, prices=None) -> dict:
        """
        For each asset:
        1. Cross-representation ARI (Paper 1)
        2. Cross-frequency ARI (Paper 2)
        3. HMM spectral parameters
        4. MRI = (1 - RSS_rep) * (1 - RSS_res)
        5. Zone classification + ordinal bound check
        6. Return MriResult per asset + aggregate
        """
```

**CLI:**

```bash
python run.py run config.yaml mri
```

---

### 3. Config Extension

```yaml
validator:
  mri:
    assets:
      SPY: data/SPY_5m.csv
      CL:  data/CL_5m.csv
      USDJPY: data/USDJPY_5m.csv
    model: gmm
    n_states: 3                           # K for representation
    n_states_res: 2                       # K for resolution
    seeds: [1, 2, 3]
    representations:
      - [vol, drawdown, maxdd, var, cvar]
      - [vol, drawdown, var, cvar]
      - [real_skew, vol_stab, var, cvar]
    zone_boundary_mri: 0.80               # B_12
    zone_boundary_rho: 0.80               # B_23
    local_variance_window: 63
    wasserstein_slices: 100
    monitoring:
      alert_zone_breach: true
      alert_zone_transition: true
      alert_mri_below: 0.40
      alert_mri_delta: -0.10
```

---

### 4. Findings Engine Update

Zone-based severity replaces empirical ARI threshold:

| Zone | Condition | Severity |
|------|-----------|----------|
| III | Chaotic | Critical |
| II + ordinal violation | Collapsed, ordering unreliable | High |
| II | Collapsed, ordering preserved | Medium |
| I + fallback triggered | Stable but GMM fallback used | Low |
| I | Stable, no issues | Informational |

---

### 5. Deliverables

| Deliverable | Acceptance Criteria |
|-------------|---------------------|
| `mrv.mri.compute_mri()` | Multiplicative formula, clamped [0,1], unit tested |
| `mrv.mri.classify_zone()` | Correct zone assignment for all boundary cases |
| `mrv.mri.fit_hmm_spectral()` | Returns lambda_2, gamma, kappa; agrees with hmmlearn |
| `mrv.mri.sliced_wasserstein()` | Monotonic with known divergence; n_slices configurable |
| `mrv.mri.ordinal_bound()` | g(rho_0, K) matches Paper 3 formula |
| `MriValidator` | Runs rep + res + spectral in one pass; returns MriResult |
| Config `validator.mri` | YAML-driven, all parameters documented |
| JSON/PDF output | MRI dashboard, phase diagram, zone governance actions |
| Findings engine | Zone-based severity replaces ARI threshold |
| Monitoring | Zone transition alerts, MRI-based thresholds |
| Tests | Unit + integration covering all MRI functions |
| Notebook | `examples/paper3_model_risk_index.ipynb` |

### 6. Implementation Dependency

| Theorem | What it provides | Module |
|---------|------------------|--------|
| Theorem 3.2 | D^2/sigma^2_local ~ (1-ARI) | `wasserstein.py` |
| Theorem 3.5 | lambda_2 and m control resolution collapse | `spectral.py` |
| Corollary 3.7 | Optimal aggregation m* | `spectral.py` |
| Theorem 3.8 | g(rho_0, K) ordinal robustness bound | `ordinal.py` |
| Proposition 3.1 | Multiplicative decomposition | `score.py` |

All formulas are finalized in Paper 3 source code.
