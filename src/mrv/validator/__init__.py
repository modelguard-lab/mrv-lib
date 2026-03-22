"""
mrv.validator — Regime diagnostics.

Validators
----------
- RepValidator: Representation Invariance (Paper 1)
- (ResValidator, TempValidator: planned)

Base class: ``BaseValidator`` — subclass to create custom tests.
"""

from mrv.validator.base import BaseValidator
from mrv.validator.rep import RepValidator
from mrv.validator.report import generate_report
from mrv.validator import metrics

__all__ = ["BaseValidator", "RepValidator", "generate_report", "metrics"]
