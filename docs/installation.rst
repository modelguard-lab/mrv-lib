Installation
============

Requirements
------------

Python 3.9 or later. mrv-lib is tested on Python 3.9, 3.10, and 3.12.

Standard install
----------------

.. code-block:: bash

    pip install mrv-lib

This installs the core library with all runtime dependencies
(numpy, pandas, scipy, scikit-learn, hmmlearn, matplotlib, pyyaml, yfinance).

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/modelguard-lab/mrv-lib.git
    cd mrv-lib
    pip install -e ".[dev]"

The ``dev`` extras add pytest, pytest-cov, black, ruff, and mypy.

Building documentation
----------------------

.. code-block:: bash

    pip install mrv-lib[docs]
    cd docs
    sphinx-build -W -b html . _build/html

Or from the repo root using the convenience script::

    sphinx-build -W -b html docs docs/_build/html
