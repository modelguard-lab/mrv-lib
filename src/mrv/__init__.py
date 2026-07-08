"""
mrv: Model Risk Validator.

A pure validation library for testing specification invariance of
financial models.  Users supply their own model outputs (labels, ranks,
grades) -- mrv measures how stable those outputs are across different
specifications.

Recommended API -- functional validators that return typed results::

    import mrv

    # Representation invariance (Paper 1): supply your own model callable
    rep = mrv.rep_invariance_validator(model_fn, admissible_class, returns=..., K=3)
    rep.summary()      # short text verdict
    rep.ari_per_pair   # per-pair ARI; also mean_ari, ordering_per_pair, ...

    # Resolution invariance (Paper 2): validate labels across frequencies
    res = mrv.res_invariance_validator(model_fn, resolution_set, spec=mrv.ResolutionSpec())
    res.summary()      # short text verdict
    res.ari_matrix     # per-asset cross-frequency ARI matrix; also overall_mean_ari, ...

    # Report generation from a saved result.json
    mrv.report("result.json")

These top-level functions are the stable public contract. ``mrv.pipeline`` is
the INTERNAL labels-first backend for the ``mrv`` command-line interface
(``mrv init`` / ``mrv download`` / ``mrv run``); its dict-returning helpers may
change and are not the recommended programmatic entry point.

Subpackages
-----------
- mrv.invariance: Representation and resolution invariance validators (public API)
- mrv.validator:  Validator classes, metrics, monitoring, and reporting
- mrv.data:       Data loading, factor computation, normalization (convenience)
- mrv.models:     Regime model fitting (optional, requires scikit-learn)
- mrv.utils:      Configuration (YAML) and logging setup
"""

__version__ = "0.7.0"

# Exception hierarchy
from mrv.exceptions import MrvConfigError, MrvError, MrvValidationError

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

# Report generation -- bind the stable public symbol to its implementation
# module (mrv.validator.report) rather than the internal CLI backend
# (mrv.pipeline). Same signature/behavior; the CLI backend keeps its own
# mrv.pipeline.report for logging.
from mrv.validator.report import generate_report as report

__all__ = [
    "report",
    # Exceptions
    "MrvError",
    "MrvValidationError",
    "MrvConfigError",
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
