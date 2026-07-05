Tutorial: Paper 1 -- Representation Invariance
===============================================

**Source paper:** Zheng, Low & Wang (2026), *Regime Labels Are Not
Representation-Invariant*.

This tutorial shows how to use :func:`mrv.invariance.rep_invariance_validator`
to test whether a regime model's labels are stable across different feature
representations. The core question is: if a practitioner had chosen different
features (volume vs. drawdown, VaR vs. CVaR), would the model assign the same
regime labels to each day?

Theory background
-----------------

Paper 1 defines *representation invariance* as the property that the label
assignment :math:`\ell(\mathbf{x})` for an observation :math:`\mathbf{x}`
does not change when :math:`\mathbf{x}` is drawn from an admissible
alternative specification :math:`\phi \in \Phi`. The validator measures this
empirically with two layers: partition stability, quantified by the mean
off-diagonal Adjusted Rand Index (ARI) across specification pairs, and
risk-ordering stability, quantified by a rank-aligned Spearman correlation
against a risk proxy.

A model passes the partition layer when the mean ARI meets the library
threshold, and passes the ordering layer when the mean Spearman correlation
meets its threshold, even if the categorical labels themselves differ.

Step 1: prepare labels
-----------------------

The validator expects a dict mapping asset names to dicts of label arrays.
Labels are integers (regime identifiers); length must match across
specifications for a given asset.

.. code-block:: python

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 300
    K = 3

    # True latent regime (Markov chain)
    true_labels = np.zeros(n, dtype=int)
    state = 0
    trans = {0: [0.90, 0.07, 0.03],
             1: [0.10, 0.80, 0.10],
             2: [0.05, 0.15, 0.80]}
    for i in range(1, n):
        state = rng.choice(K, p=trans[state])
        true_labels[i] = state

    def perturb(labels, noise_p, seed):
        rng_l = np.random.default_rng(seed)
        out = labels.copy()
        mask = rng_l.random(len(out)) < noise_p
        out[mask] = rng_l.integers(0, K, mask.sum())
        return out

    # Three specifications with increasing noise
    labels_a = perturb(true_labels, 0.03, 1)   # vol + dd + var
    labels_b = perturb(true_labels, 0.07, 2)   # vol + var + CVaR
    labels_c = perturb(true_labels, 0.11, 3)   # vol + skew only

Step 2: run the validator
--------------------------

.. code-block:: python

    from mrv.invariance import rep_invariance_validator

    result = rep_invariance_validator(
        model_fn=None,               # not needed: labels are pre-computed
        admissible_class={
            "vol+dd+var":  labels_a,
            "vol+var+cvar": labels_b,
            "vol+skew":     labels_c,
        },
        returns=None,                # ordering check skipped when None
        K=K,
    )

    result.summary()
    print("ARI per pair:", result.ari_per_pair)
    print("Mean ARI:    ", result.mean_ari)
    print("Pass (ARI >= 0.65):", result.partition_pass)

The ``1/K`` null
-----------------

Paper 1 (Supplement, around Table 3) shows that a random relabelling
baseline achieves ARI approximately :math:`1/K`. The validator exposes this
null so you can report the margin above chance:

.. code-block:: python

    print("1/K null:", result.null_1_over_K)
    print("Margin above null:", result.mean_ari - result.null_1_over_K)

Ordering consistency
---------------------

When a risk proxy (e.g., rolling volatility) is available, the validator also
checks whether the ordinal risk ordering of regimes is consistent across
specifications. This corresponds to the *ordering null* reported in Paper 1
Table 3.

.. code-block:: python

    dates = pd.bdate_range("2023-01-02", periods=n)
    ret = rng.normal(0, 0.01, n)
    risk_proxy = pd.Series(ret, index=dates).rolling(20).std().bfill().values

    result2 = rep_invariance_validator(
        model_fn=None,
        admissible_class={
            "vol+dd+var":   labels_a,
            "vol+var+cvar": labels_b,
        },
        returns=risk_proxy,
        K=K,
    )
    print("Ordering pass:", result2.ordering_pass)
    print("Ordering per pair:", result2.ordering_per_pair)

Interpreting results
---------------------

+------------------------+----------------------------------------------+
| Field                  | Interpretation                               |
+========================+==============================================+
| ``mean_ari``           | Average ARI across all specification pairs.  |
|                        | >= 0.65 is the Paper 1 threshold.            |
+------------------------+----------------------------------------------+
| ``partition_pass``     | True if mean_ari >= ARI_THRESHOLD (0.65).    |
+------------------------+----------------------------------------------+
| ``ordering_pass``      | True if Spearman >= 0.85 for all pairs.      |
+------------------------+----------------------------------------------+
| ``null_1_over_K``      | Expected ARI under random relabelling        |
|                        | (baseline = 1/K).                            |
+------------------------+----------------------------------------------+

See also
--------

* :func:`mrv.invariance.rep_invariance_validator` -- full API reference
* :class:`mrv.invariance.RepInvarianceResult` -- result object fields
* :doc:`paper2_res_invariance` -- resolution invariance in detail
