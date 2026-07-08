"""
mrv.exceptions -- Project exception hierarchy.

The typed hierarchy below covers input and configuration validation raised
along mrv's documented usage paths: the public invariance validators
(``mrv.rep_invariance_validator`` / ``mrv.res_invariance_validator`` and their
``RepValidator`` / ``ResValidator`` backends), the ``mrv`` CLI pipeline
(``mrv.pipeline`` and config loading), and the data and model convenience
helpers (factors, normalization, reader, model dispatch, Yahoo download). Every
such error derives from :class:`MrvError`, so downstream code can catch all of
them with a single ``except MrvError``.

The concrete subclasses additionally inherit from the builtin they replace
(``ValueError``), so existing code that catches the builtin keeps working::

    try:
        mrv.res_invariance_validator(model_fn, resolution_set={})
    except mrv.MrvError:        # library-specific handling
        ...
    except ValueError:          # still catches the same error (backward-compatible)
        ...

Not remapped, by design (these keep their standard builtin type): missing files
(``FileNotFoundError``), missing optional dependencies (``ImportError``),
unsupported-by-design code paths (``NotImplementedError``), and the optional
Interactive Brokers download helpers (``mrv.utils.download_ib``).
"""

from __future__ import annotations

__all__ = [
    "MrvError",
    "MrvValidationError",
    "MrvConfigError",
]


class MrvError(Exception):
    """Base class for all errors raised by the mrv library."""


class MrvValidationError(MrvError, ValueError):
    """Invalid user input to a validator.

    Multiply inherits from ``ValueError`` so existing ``except ValueError``
    code continues to catch it (backward-compatible).
    """


class MrvConfigError(MrvError, ValueError):
    """Invalid configuration.

    Multiply inherits from ``ValueError`` so existing ``except ValueError``
    code continues to catch it (backward-compatible).
    """
