# Configuration file for Sphinx documentation builder.
# mrv-lib -- Model Risk Validator
# docs/conf.py

import sys
from pathlib import Path

# Make src/ importable so autodoc can import mrv
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "mrv-lib"
author = "Kai Zheng"
copyright = "2026, Kai Zheng"

# Single-source the version from the installed package.
from mrv import __version__ as _mrv_version  # noqa: E402

release = _mrv_version
version = ".".join(_mrv_version.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__, __dict__",
}

autodoc_typehints = "signature"
autodoc_typehints_format = "short"
always_document_param_types = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# MyST settings -- parse .md files for tutorials
myst_enable_extensions = ["colon_fence", "deflist"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Files that can be source documents
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ---------------------------------------------------------------------------
# HTML output options
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "mrv-lib"
html_short_title = "mrv-lib"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/modelguard-lab/mrv-lib",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/modelguard-lab/mrv-lib",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

html_static_path = []

# ---------------------------------------------------------------------------
# Suppress known harmless warnings
# ---------------------------------------------------------------------------
nitpicky = False

# Suppress warnings for cross-reference targets we do not control
nitpick_ignore = [
    ("py:class", "numpy.ndarray"),
    ("py:class", "pandas.Series"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "optional"),
]

# Suppress duplicate-object warnings from autosummary stubs that re-export
# symbols already documented in their home module
suppress_warnings = [
    "autosummary.import_cycle",
    "ref.duplicate",
    "autodoc",
    "py.duplicate_description",
]
