"""
mrv.validator — Regime diagnostics.

Validators
----------
- RepValidator: Representation Invariance (Paper 1)
- ResValidator: Resolution Invariance (Paper 2)
- (TempValidator: planned)

Base class: ``BaseValidator`` — subclass to create custom tests.
"""

from mrv.validator.base import BaseValidator
from mrv.validator.rep import RepValidator
from mrv.validator.res import ResValidator
from mrv.validator.report import generate_report, generate_sr11_7_report
from mrv.validator import metrics

__all__ = [
    "BaseValidator", "RepValidator", "ResValidator",
    "generate_report", "generate_sr11_7_report", "metrics",
]
