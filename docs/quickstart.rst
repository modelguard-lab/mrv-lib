Quick start
===========

Labels-first API
-----------------

Supply pre-computed labels from your own model. mrv only measures agreement --
it never fits a model itself:

.. code-block:: python

    from mrv.pipeline import validate_rep
    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    base = rng.integers(0, 3, n)
    labels_a = base.copy()
    labels_b = base.copy(); labels_b[rng.random(n) < 0.05] = rng.integers(0, 3, (rng.random(n) < 0.05).sum())

    result = validate_rep(labels={
        "SPY": {
            "vol+dd+var":   labels_a,
            "vol+var+cvar": labels_b,
        }
    })
    print(result["assets"]["SPY"]["mean_ari"])

Validation report
-----------------

Generate a specification-invariance report from a result JSON. The ``.tex`` is
always written; the PDF is compiled only when ``pdflatex`` is on ``PATH``:

.. code-block:: python

    import json, pathlib, tempfile
    from mrv.pipeline import report

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

    pdf_path = report(str(p))   # -> Path to the .pdf (or None if pdflatex absent)

Next steps
----------

* :doc:`tutorials/paper1_rep_invariance` -- representation invariance in detail
* :doc:`tutorials/paper2_res_invariance` -- resolution invariance in detail
* :doc:`api/mrv` -- full API reference
