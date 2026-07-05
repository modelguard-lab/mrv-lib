Tutorial: Paper 2 -- Resolution Invariance
==========================================

**Source paper:** Zheng, Low & Wang (2026), *Regime Labels Are Not
Resolution-Invariant: Evidence Across Five Asset Classes*.

This tutorial demonstrates :func:`mrv.invariance.res_invariance_validator`,
which tests whether a regime model produces consistent labels across
sampling frequencies (5m, 15m, 1h, 1d). The core question is: does a model
trained on daily bars agree with the same model trained on hourly bars?

Theory background
-----------------

Paper 2 defines *resolution invariance* as label stability under a
change of temporal aggregation. The key empirical finding (Paper 2 Table 2)
is that cross-frequency ARI drops significantly below the 0.65 threshold
for most asset-frequency pairs, with intraday pairs showing higher agreement
than daily-vs-intraday pairs.

:class:`~mrv.invariance.ResolutionSpec` encodes the Paper 2 panel:
four frequencies (``5m``, ``15m``, ``1h``, ``1d``) and the two
intraday-only frequencies for the within-intraday excess statistic.

Step 1: build per-frequency label dicts
-----------------------------------------

The validator expects a nested dict:
``{asset: {freq: label_array}}``.

.. code-block:: python

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    K = 3
    N_DAILY = 250

    def make_label_seq(n, seed):
        rng_l = np.random.default_rng(seed)
        labels = np.zeros(n, dtype=int)
        state = 0
        trans = {0: [0.88, 0.08, 0.04],
                 1: [0.10, 0.78, 0.12],
                 2: [0.06, 0.14, 0.80]}
        for i in range(1, n):
            state = rng_l.choice(K, p=trans[state])
            labels[i] = state
        return labels

    # 1d bars (250 observations)
    labels_1d = make_label_seq(N_DAILY, seed=0)

    # 1h bars (approx 8x more observations for equity session)
    labels_1h = make_label_seq(N_DAILY * 8, seed=1)

    # 15m bars
    labels_15m = make_label_seq(N_DAILY * 26, seed=2)

    # 5m bars
    labels_5m = make_label_seq(N_DAILY * 78, seed=3)

    resolution_set = {
        "SPY": {
            "5m":  labels_5m,
            "15m": labels_15m,
            "1h":  labels_1h,
            "1d":  labels_1d,
        }
    }

Step 2: run the validator
--------------------------

.. code-block:: python

    from mrv.invariance import res_invariance_validator, ResolutionSpec

    result = res_invariance_validator(
        model_fn=None,
        resolution_set=resolution_set,
        spec=ResolutionSpec(),    # default Paper 2 four-frequency panel
    )

    result.summary()
    print("ARI matrix (SPY):", result.ari_matrix["SPY"])
    print("AMI matrix (SPY):", result.ami_matrix["SPY"])

Within-intraday excess
-----------------------

Paper 2 shows that *within-intraday* ARI (e.g., 5m vs 15m) is consistently
higher than *daily-vs-intraday* ARI (1h or 15m vs 1d). The
``within_intraday_excess`` field reports the difference:

.. code-block:: python

    print("Within-intraday excess (SPY):",
          result.within_intraday_excess.get("SPY"))

A positive value means the model is more consistent among intraday frequencies
than it is when compared to the daily bar -- the typical finding from Paper 2.

Permutation p-values
---------------------

The validator computes permutation p-values for each frequency pair to guard
against inflated ARI from small sample sizes:

.. code-block:: python

    if result.perm_pvalue:
        for pair, pv in result.perm_pvalue.get("SPY", {}).items():
            print(f"  {pair}: p={pv:.4f}")

Using PAPER2_FREQS
-------------------

:data:`mrv.invariance.PAPER2_FREQS` and
:data:`mrv.invariance.PAPER2_INTRADAY_FREQS` are the canonical frequency
lists from Paper 2 and are used to construct :class:`ResolutionSpec`:

.. code-block:: python

    from mrv.invariance import PAPER2_FREQS, PAPER2_INTRADAY_FREQS

    print("Paper 2 frequencies:", PAPER2_FREQS)
    print("Intraday subset:    ", PAPER2_INTRADAY_FREQS)

Interpreting results
---------------------

+------------------------------+------------------------------------------+
| Field                        | Interpretation                           |
+==============================+==========================================+
| ``ari_matrix[asset]``        | DataFrame of cross-frequency ARI values. |
|                              | >= 0.65 is the Paper 2 threshold.        |
+------------------------------+------------------------------------------+
| ``ami_matrix[asset]``        | AMI robustness table (Paper 2 Table S1). |
+------------------------------+------------------------------------------+
| ``within_intraday_excess``   | Intraday mean ARI minus overall mean ARI |
|                              | (positive = within-intraday more stable).|
+------------------------------+------------------------------------------+
| ``perm_pvalue``              | Permutation p-values per pair per asset. |
+------------------------------+------------------------------------------+

See also
--------

* :func:`mrv.invariance.res_invariance_validator` -- full API reference
* :class:`mrv.invariance.ResInvarianceResult` -- result object fields
* :class:`mrv.invariance.ResolutionSpec` -- frequency-panel configuration
* :doc:`paper1_rep_invariance` -- representation invariance tutorial
