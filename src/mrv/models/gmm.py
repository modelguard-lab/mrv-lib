"""mrv.models.gmm -- Gaussian Mixture Model."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

# Default RNG seed for reproducible model fitting.
_DEFAULT_SEED = 1

logger = logging.getLogger(__name__)

__all__ = ["fit_gmm"]


def fit_gmm(features: pd.DataFrame, K: int = 3, **kwargs) -> Optional[np.ndarray]:
    """Fit a Gaussian Mixture Model and return hard regime labels.

    Parameters
    ----------
    features : pd.DataFrame
        Normalised feature matrix. Rows with NaN are dropped before fitting.
    K : int, default 3
        Number of mixture components (regime states).
    **kwargs
        ``random_state`` (default 1) and ``n_init`` (default 10) are forwarded
        to ``sklearn.mixture.GaussianMixture``.

    Returns
    -------
    np.ndarray or None
        Integer label array of shape ``(n_obs,)`` where ``n_obs = len(features.dropna())``,
        or ``None`` when the input is too short to fit (fewer than
        ``max(K * 5, 20)`` rows after NaN removal).
    """
    from sklearn.mixture import GaussianMixture
    X = features.dropna().values
    if len(X) < max(K * 5, 20):
        logger.warning("GMM: insufficient data (%d rows)", len(X))
        return None
    gmm = GaussianMixture(
        n_components=K,
        random_state=kwargs.get("random_state", _DEFAULT_SEED),
        n_init=kwargs.get("n_init", 10),
    )
    return gmm.fit_predict(X)
