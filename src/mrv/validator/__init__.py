"""
mrv.validator -- Regime diagnostics.

Validators
----------
- RepValidator: Representation Invariance (Paper 1)
- ResValidator: Resolution Invariance (Paper 2)

Base class: ``BaseValidator`` -- subclass to create custom tests.
"""

from mrv.validator import metrics
from mrv.validator.base import BaseValidator
from mrv.validator.rep import RepValidator
from mrv.validator.report import generate_report
from mrv.validator.res import ResValidator

__all__ = [
    "BaseValidator", "RepValidator", "ResValidator",
    "generate_report",
    "metrics",
]
