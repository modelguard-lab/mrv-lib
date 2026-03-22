"""mrv.models.gmm — Gaussian Mixture Model."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_gmm(features: pd.DataFrame, n_states: int = 3, **kwargs) -> Optional[np.ndarray]:
    """Fit GMM and return hard labels, or None if insufficient data."""
    from sklearn.mixture import GaussianMixture
    X = features.dropna().values
    if len(X) < max(n_states * 5, 20):
        logger.warning("GMM: insufficient data (%d rows)", len(X))
        return None
    gmm = GaussianMixture(
        n_components=n_states,
        random_state=kwargs.get("random_state", 42),
        n_init=kwargs.get("n_init", 5),
    )
    return gmm.fit_predict(X)
