"""
mrv-lib: Market Regime Validity Library.

Model risk diagnostics in non-stationary markets — Representation Stability,
Identifiability Boundaries, and Ordinal Robustness.
"""

from mrv_lib.core import (
    BoundaryResult,
    RepresentationTestResult,
    Scanner,
    detect_boundary,
    ordinal_consistency,
    ari_score,
)

__all__ = [
    "Scanner",
    "RepresentationTestResult",
    "detect_boundary",
    "BoundaryResult",
    "ordinal_consistency",
    "ari_score",
]
__version__ = "0.0.2"
