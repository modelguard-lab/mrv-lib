"""Regression test: no em-dash (U+2014) in source, test, template, or example files.

R4-F4: rounds 1-4 each found em-dashes in a new surface class (src/, templates/,
examples/, tests/). This test scans all delivery surfaces so future rounds cannot
regress this pattern without CI catching it.

Permitted exceptions:
- This file itself is excluded from the scan (it must store the codepoint to test).
- Binary files (.pyc, images) are skipped by the glob filter.
- The LaTeX table-cell placeholder '---' is a three-char sequence, not U+2014; it
  is not caught by this scan.
- The actual em-dash codepoint is U+2014; en-dash '--' is U+2013 and is permitted
  (used for numeric ranges in CHANGELOG and ROADMAP).
"""

from __future__ import annotations

import pathlib

EM_DASH = "—"

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_THIS_FILE = pathlib.Path(__file__).resolve()

# Directories and glob patterns to scan.
_SCAN_GLOBS = [
    "src/**/*.py",
    "src/**/*.yaml",
    "src/**/*.tex",
    "tests/**/*.py",
    "templates/**/*.tex",
    "examples/**/*.py",
    "examples/**/*.ipynb",
    "*.md",
    "*.yaml",
    "*.toml",
    "*.cff",
]


def _scanned_files() -> set[pathlib.Path]:
    """Return the set of files the em-dash scan actually covers."""
    seen: set[pathlib.Path] = set()
    for pattern in _SCAN_GLOBS:
        for path in _REPO_ROOT.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            if path.resolve() == _THIS_FILE:
                continue
            seen.add(path)
    return seen


def _collect_hits() -> list[tuple[pathlib.Path, int, str]]:
    """Return (path, line_number, line_text) for every em-dash occurrence."""
    hits: list[tuple[pathlib.Path, int, str]] = []
    seen: set[pathlib.Path] = set()
    for pattern in _SCAN_GLOBS:
        for path in _REPO_ROOT.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            if path.resolve() == _THIS_FILE:
                # Exclude this file: it stores the codepoint by necessity.
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if EM_DASH in line:
                    hits.append((path, lineno, line.rstrip()))
    return hits


def test_no_em_dash_in_delivery_surfaces():
    """No file in src/, tests/, templates/, examples/, or root docs may contain U+2014."""
    hits = _collect_hits()
    if hits:
        lines = "\n".join(
            f"  {path.relative_to(_REPO_ROOT)}:{lineno}: {text}"
            for path, lineno, text in hits
        )
        raise AssertionError(
            f"Em-dash (U+2014) found in {len(hits)} location(s):\n{lines}\n\n"
            "Replace each em-dash with a colon, semicolon, comma, or sentence split "
            "(P11). Do not use en-dash '--' as a substitute for prose separators."
        )


def test_shipped_template_is_scanned():
    """The shipped LaTeX report template (src/mrv/templates/template.tex) must
    be inside the scanned set, so a future em-dash in it cannot slip past CI.
    """
    template = _REPO_ROOT / "src" / "mrv" / "templates" / "template.tex"
    assert template.is_file(), f"expected shipped template at {template}"
    assert template in _scanned_files()
