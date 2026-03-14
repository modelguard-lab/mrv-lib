"""
mrv-lib: Market Regime Validity Library.

Model risk diagnostics in non-stationary markets — Representation Stability,
Identifiability Boundaries, and Ordinal Robustness.
"""

from mrv_lib.scan import Scanner, RepresentationTestResult
from mrv_lib.boundary import detect_boundary, BoundaryResult
from mrv_lib.metrics import ordinal_consistency

__all__ = [
    "Scanner",
    "RepresentationTestResult",
    "detect_boundary",
    "BoundaryResult",
    "ordinal_consistency",
]
__version__ = "0.1.0"
