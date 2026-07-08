"""mrv.models.hmm -- Gaussian Hidden Markov Model."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

# Default RNG seed for reproducible model fitting.
#
# NOTE ON SEED CONVENTION: model fitting defaults to seed 1, while the Paper 2
# resolution permutation test defaults to seed 42 (see
# ``mrv.invariance.res.res_invariance_validator``). This difference is
# intentional: the two are independent knobs. The model-fit seed controls
# sklearn/hmmlearn parameter initialisation; the permutation seed controls the
# null-distribution shuffle. Overriding one does not affect the other.
_DEFAULT_SEED = 1

logger = logging.getLogger(__name__)

__all__ = ["fit_hmm"]


def fit_hmm(features: pd.DataFrame, K: int = 3, **kwargs) -> Optional[np.ndarray]:
    """Fit a Gaussian Hidden Markov Model and return Viterbi-decoded regime labels.

    Parameters
    ----------
    features : pd.DataFrame
        Normalised feature matrix. Rows with NaN are dropped before fitting.
    K : int, default 3
        Number of hidden states (regime states).
    **kwargs
        ``covariance_type`` (default ``"full"``), ``n_iter`` (default 200),
        and ``random_state`` (default 1) are forwarded to
        ``hmmlearn.hmm.GaussianHMM``.

    Returns
    -------
    np.ndarray or None
        Integer label array of shape ``(n_obs,)`` where ``n_obs = len(features.dropna())``,
        or ``None`` when the input is too short to fit (fewer than
        ``max(K * 5, 20)`` rows after NaN removal).

    Raises
    ------
    ImportError
        If ``hmmlearn`` is not installed (``pip install hmmlearn``).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError(
            "HMM requires hmmlearn. Install with: pip install hmmlearn"
        ) from None
    X = features.dropna().values
    if len(X) < max(K * 5, 20):
        logger.warning("HMM: insufficient data (%d rows)", len(X))
        return None
    hmm = GaussianHMM(
        n_components=K,
        covariance_type=kwargs.get("covariance_type", "full"),
        n_iter=kwargs.get("n_iter", 200),
        random_state=kwargs.get("random_state", _DEFAULT_SEED),
    )
    hmm.fit(X)
    return hmm.predict(X)
