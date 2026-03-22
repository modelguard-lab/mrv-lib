"""mrv.models.hmm — Gaussian Hidden Markov Model."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_hmm(features: pd.DataFrame, n_states: int = 3, **kwargs) -> Optional[np.ndarray]:
    """Fit Gaussian HMM and return Viterbi-decoded labels, or None if insufficient data."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError("HMM requires hmmlearn. Install with: pip install hmmlearn")
    X = features.dropna().values
    if len(X) < max(n_states * 5, 20):
        logger.warning("HMM: insufficient data (%d rows)", len(X))
        return None
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=kwargs.get("covariance_type", "full"),
        n_iter=kwargs.get("n_iter", 200),
        random_state=kwargs.get("random_state", 42),
    )
    hmm.fit(X)
    return hmm.predict(X)
