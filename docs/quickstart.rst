Quick start
===========

Recommended Python API
----------------------

The recommended public Python API lives at the top level of the ``mrv`` package.
You supply a ``model_fn`` (features to integer labels) and the admissible set of
specifications; mrv returns a typed result. mrv only measures agreement -- it
never fits a model itself.

Representation invariance (Paper 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import mrv

    rng = np.random.default_rng(42)
    n = 200
    base = rng.integers(0, 3, n)
    labels_a = base.copy()
    labels_b = base.copy()
    flip = rng.random(n) < 0.10
    labels_b[flip] = rng.integers(0, 3, flip.sum())   # small perturbation
    returns = rng.standard_normal(n) * 0.01

    result = mrv.rep_invariance_validator(
        model_fn=lambda x: x,   # passthrough: supply pre-computed labels directly
        admissible_class={"vol+dd+var": labels_a, "vol+var+cvar": labels_b},
        returns=returns,        # optional: enables the Spearman ordering check
        K=3,                    # number of regime states
    )
    print(result.summary())
    print("mean ARI:", result.mean_ari["asset"])
    print("partition passes:", result.passes_partition["asset"])

To feed a real model, replace the passthrough with a ``model_fn`` that fits your
regime model and returns integer labels.

Resolution invariance (Paper 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import numpy as np
    import mrv

    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-05 09:30", periods=480, freq="5min",
                        tz="America/New_York")
    labels_5m = pd.Series(rng.integers(0, 2, 480), index=idx, dtype=int)
    labels_15m = labels_5m.iloc[::3].copy()

    result = mrv.res_invariance_validator(
        model_fn=lambda s: s,   # passthrough: pre-computed labels per frequency
        resolution_set={"SPY": {"5m": labels_5m, "15m": labels_15m}},
        spec=mrv.ResolutionSpec(freqs=("5m", "15m"), intraday_freqs=("5m", "15m")),
        run_permutation=False,
    )
    print(result.summary())
    print(result.ari_matrix["SPY"].round(3))
    print("overall mean ARI:", result.overall_mean_ari["SPY"])

The typed results (``RepInvarianceResult`` / ``ResInvarianceResult``) expose
``.summary()`` plus attributes such as ``.ari_matrix``, ``.overall_mean_ari``,
``.passes_partition``, and ``.intraday_overall_ari_gap``.

Validation report
-----------------

Generate a specification-invariance report from a result JSON with the top-level
``mrv.report``. The ``.tex`` is always written; the PDF is compiled only when
``pdflatex`` is on ``PATH``:

.. code-block:: python

    import json, pathlib, tempfile
    import mrv

    result_json = {
        "test": "representation_invariance",
        "model": "GMM",
        "n_states": 3,
        "overall_mean_ari": 0.72,
        "overall_mean_spearman": 0.88,
        "partition_pass": True,
        "ordering_pass": True,
        "ari_threshold": 0.65,
        "spearman_threshold": 0.85,
        "assets": {},
    }
    tmp = pathlib.Path(tempfile.mkdtemp())
    p = tmp / "result.json"
    p.write_text(json.dumps(result_json))

    pdf_path = mrv.report(str(p))   # -> Path to the .pdf (or None if pdflatex absent)

Config-file workflow (CLI)
--------------------------

For the config-driven workflow (data download, model fitting, validation, and a
PDF report in one command), use the ``mrv`` CLI. It is backed internally by
``mrv.pipeline`` (see the API reference):

.. code-block:: bash

    mrv init                 # write a starter config.yaml
    mrv download config.yaml # fetch OHLCV data
    mrv run config.yaml      # run validators + generate a report

Next steps
----------

* :doc:`tutorials/paper1_rep_invariance` -- representation invariance in detail
* :doc:`tutorials/paper2_res_invariance` -- resolution invariance in detail
* :doc:`api/mrv` -- full API reference
