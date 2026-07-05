"""
mrv.validator.metrics -- Label comparison metrics for regime diagnostics.

Standard statistical measures -- not extensible by users.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
)

# Thresholds -- single source of truth for the entire library.
ARI_THRESHOLD: float = 0.65          # Steinley (2004): acceptable partition recovery
SPEARMAN_THRESHOLD: float = 0.85     # Ordinal risk ordering stability
MIN_SAMPLES: int = 10                # Minimum observations for meaningful comparison

__all__ = [
    "ARI_THRESHOLD",
    "SPEARMAN_THRESHOLD",
    "MIN_SAMPLES",
    "ari",
    "ami",
    "nmi",
    "ordering_consistency",
    "variation_of_information",
]


def ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand Index. Range [-1,1]; 1=perfect, ~0=random."""
    n = min(len(labels_a), len(labels_b))
    if n < MIN_SAMPLES:
        return float("nan")
    return float(adjusted_rand_score(labels_a[:n], labels_b[:n]))


def ami(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Mutual Information. Range [0,1]; 1=perfect."""
    n = min(len(labels_a), len(labels_b))
    if n < MIN_SAMPLES:
        return float("nan")
    return float(adjusted_mutual_info_score(labels_a[:n], labels_b[:n]))


def nmi(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Normalized Mutual Information. Range [0,1]; 1=perfect."""
    n = min(len(labels_a), len(labels_b))
    if n < MIN_SAMPLES:
        return float("nan")
    return float(normalized_mutual_info_score(labels_a[:n], labels_b[:n]))


def _state_risk_rank(labels: np.ndarray, risk: np.ndarray) -> Dict[Any, int]:
    """Map each state to its mean risk, then rank states."""
    states = np.unique(labels)
    mean_risk: Dict[Any, float] = {s: float(np.mean(risk[labels == s])) for s in states}
    sorted_states = sorted(mean_risk, key=lambda k: mean_risk[k])
    return {s: rank for rank, s in enumerate(sorted_states)}


def ordering_consistency(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    features: np.ndarray,
) -> float:
    """
    Ordinal ordering consistency between two label sets.

    Each representation's states are ranked by mean risk (mean feature value).
    Each observation is then mapped to its state's risk rank (0=lowest, K-1=highest).
    Returns Spearman correlation of these risk-rank sequences.

    This measures whether the two representations agree on the *relative risk
    ordering* of observations, even if the exact partition boundaries differ.

    Threshold: Spearman >= 0.85 indicates stable risk ordering.
    """
    from scipy.stats import spearmanr

    n = min(len(labels_a), len(labels_b), len(features))
    if n < MIN_SAMPLES:
        return float("nan")
    a, b, X = labels_a[:n], labels_b[:n], features[:n]

    # Risk proxy: mean across feature columns (higher = riskier)
    if X.ndim > 1:
        risk_proxy = np.mean(X, axis=1)
    else:
        risk_proxy = X.copy()

    rank_a = _state_risk_rank(a, risk_proxy)
    rank_b = _state_risk_rank(b, risk_proxy)

    # Map observations to their state's risk rank
    ordinal_a = np.array([rank_a[s] for s in a], dtype=float)
    ordinal_b = np.array([rank_b[s] for s in b], dtype=float)

    rho, _ = spearmanr(ordinal_a, ordinal_b)
    return float(rho) if not np.isnan(rho) else 0.0


def variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Variation of Information. Lower=more similar; 0=identical."""
    n = min(len(labels_a), len(labels_b))
    if n < MIN_SAMPLES:
        return float("nan")
    a, b = labels_a[:n], labels_b[:n]
    mi = float(mutual_info_score(a, b))
    _, ca = np.unique(a, return_counts=True)
    _, cb = np.unique(b, return_counts=True)
    pa = ca.astype(float) / float(ca.sum())
    pb = cb.astype(float) / float(cb.sum())
    ha = -float(np.dot(pa, np.log(pa + 1e-15)))
    hb = -float(np.dot(pb, np.log(pb + 1e-15)))
    return float(ha + hb - 2.0 * mi)
