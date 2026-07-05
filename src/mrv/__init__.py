"""
mrv: Model Risk Validator.

A pure validation library for testing specification invariance of
financial models.  Users supply their own model outputs (labels, ranks,
grades) -- mrv measures how stable those outputs are across different
specifications.

Core API -- labels in, stability metrics out::

    from mrv.pipeline import validate_rep, validate_res

    # Representation invariance (Paper 1): labels from your own models
    result = validate_rep(labels={
        "SPY": {"rep_a": labels_a, "rep_b": labels_b, "rep_c": labels_c}
    })

    # Resolution invariance (Paper 2): labels at each frequency
    result = validate_res(labels={
        "SPY": {"5m": labels_5m, "15m": labels_15m, "1h": labels_1h, "1d": labels_1d}
    })

Subpackages
-----------
- mrv.invariance: Representation and resolution invariance tests (functional API)
- mrv.validator:  Validator classes, metrics, monitoring, and reporting
- mrv.data:       Data loading, factor computation, normalization (convenience)
- mrv.models:     Regime model fitting (optional, requires scikit-learn)
- mrv.utils:      Configuration (YAML) and logging setup
"""

__version__ = "0.6.1"

# Report generation
# Invariance API (Paper 1: representation; Paper 2: resolution)
from mrv.invariance import (
    PAPER2_FREQS,
    PAPER2_INTRADAY_FREQS,
    RepInvarianceResult,
    ResInvarianceResult,
    ResolutionSpec,
    rep_invariance_validator,
    res_invariance_validator,
)
from mrv.pipeline import report

__all__ = [
    "report",
    # Invariance
    "RepInvarianceResult",
    "rep_invariance_validator",
    "ResInvarianceResult",
    "ResolutionSpec",
    "res_invariance_validator",
    "PAPER2_FREQS",
    "PAPER2_INTRADAY_FREQS",
    "__version__",
]
