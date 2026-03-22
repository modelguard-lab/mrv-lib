"""
mrv.models — Regime model registry.

Built-in: gmm, hmm. Add custom: ``register_model("name", fn)``.

Model function signature: ``(features: DataFrame, n_states: int, **kwargs) -> ndarray | None``
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from mrv.models.gmm import fit_gmm
from mrv.models.hmm import fit_hmm

logger = logging.getLogger(__name__)

ModelFn = Callable[..., Optional[np.ndarray]]
_REGISTRY: Dict[str, ModelFn] = {}


def register_model(name: str, fn: ModelFn) -> None:
    """Register a model function."""
    _REGISTRY[name.lower()] = fn


def fit(features: pd.DataFrame, model: str = "gmm", n_states: int = 3, **kwargs) -> Optional[np.ndarray]:
    """Fit a regime model and return hard labels (or None on failure)."""
    fn = _REGISTRY.get(model.lower())
    if fn is None:
        raise ValueError(f"Unknown model '{model}'. Registered: {list(_REGISTRY.keys())}")
    return fn(features, n_states=n_states, **kwargs)


# Auto-register built-in
register_model("gmm", fit_gmm)
register_model("hmm", fit_hmm)

__all__ = ["fit", "register_model"]
