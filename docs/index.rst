mrv-lib: Model Risk Validator
==============================

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/paper1_rep_invariance
   tutorials/paper2_res_invariance

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/mrv
   api/mrv.invariance
   api/mrv.validator
   api/mrv.pipeline
   api/mrv.models
   api/mrv.utils

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog

mrv-lib is a pure Model Risk Validator library: you supply labels from your own
models and mrv measures how stable they are across specification choices.

The recommended public Python API is the top-level functional interface
(:func:`mrv.rep_invariance_validator`, :func:`mrv.res_invariance_validator`,
:func:`mrv.report`, and the typed results). See :doc:`quickstart` to get
started. The config-file CLI (``mrv init`` / ``mrv download`` / ``mrv run``) is
backed internally by :doc:`api/mrv.pipeline`.

* **Representation invariance** (Paper 1): do regime labels change when you
  switch feature representations?
* **Resolution invariance** (Paper 2): do labels agree across 5m/15m/1h/1d
  frequencies?

The validation output supports OCC Bulletin 2026-13 / SR 26-2 (supersedes
SR 11-7 from 2026-04-17) model risk management workflows.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
