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

The validator is *model-driven*: you supply a ``model_fn`` callable that maps a
feature matrix to integer regime labels, plus an ``admissible_class`` dict of
feature matrices (one per specification). The validator fits ``model_fn`` on
each specification and cross-compares the resulting labels.

Step 1: build the admissible feature representations
----------------------------------------------------

Each specification is a 2-D feature matrix of shape ``(n_obs, n_features)`` for
the *same* underlying observations. Here we synthesise three representations
that share a latent risk signal but differ in feature noise, mimicking the
"swap the features, keep the asset fixed" setup of Paper 1.

.. code-block:: python

    import numpy as np

    rng = np.random.default_rng(42)
    n = 300
    K = 3

    # A latent 2-D driver shared by every representation.
    base = rng.normal(0, 1, size=(n, 2))
    signal = base @ np.array([1.0, 0.5])

    def make_features(noise_scale, seed):
        r = np.random.default_rng(seed)
        extra = r.normal(0, noise_scale, size=(n, 2))
        third = signal + r.normal(0, noise_scale, n)
        return np.column_stack([base + extra, third])

    feat_a = make_features(0.10, 1)   # vol + dd + var
    feat_b = make_features(0.25, 2)   # vol + var + CVaR
    feat_c = make_features(0.60, 3)   # vol + skew only

    admissible_class = {
        "vol+dd+var":   feat_a,
        "vol+var+cvar": feat_b,
        "vol+skew":     feat_c,
    }

Step 2: run the validator
--------------------------

Pass a ``model_fn`` callable (features -> integer labels). Here we use a
K-state Gaussian mixture. The result object carries per-asset dicts keyed by
the internal asset name ``"asset"``.

.. code-block:: python

    from mrv.invariance import rep_invariance_validator
    from sklearn.mixture import GaussianMixture

    def gmm_model(features):
        gm = GaussianMixture(n_components=K, random_state=0, n_init=3)
        gm.fit(features)
        return gm.predict(features).astype(int)

    result = rep_invariance_validator(
        model_fn=gmm_model,
        admissible_class=admissible_class,
        returns=None,                # ordering check skipped when None
        K=K,
    )

    print(result.summary())
    print("ARI per pair:", result.ari_per_pair["asset"])
    print("Mean ARI:    ", result.mean_ari["asset"])
    print("Partition passes (ARI >= 0.65):", result.passes_partition["asset"])

The ``1/K`` null
-----------------

Paper 1 (Supplement, around Table 3) shows that a random relabelling
baseline achieves ARI approximately :math:`1/K`. The validator exposes this
null so you can report the margin above chance:

.. code-block:: python

    print("1/K null:", result.null_1_over_K)
    print("Margin above null:", result.mean_ari["asset"] - result.null_1_over_K)

Ordering consistency
---------------------

When a risk proxy (e.g., rolling volatility) is available, the validator also
checks whether the ordinal risk ordering of regimes is consistent across
specifications. This corresponds to the *ordering null* reported in Paper 1
Table 3. Pass the proxy as ``returns`` (a 1-D array aligned with the feature
rows).

.. code-block:: python

    import pandas as pd

    dates = pd.bdate_range("2023-01-02", periods=n)
    ret = rng.normal(0, 0.01, n)
    risk_proxy = pd.Series(ret, index=dates).rolling(20).std().bfill().values

    result2 = rep_invariance_validator(
        model_fn=gmm_model,
        admissible_class={
            "vol+dd+var":   feat_a,
            "vol+var+cvar": feat_b,
        },
        returns=risk_proxy,
        K=K,
    )
    print("Ordering passes:", result2.passes_ordering["asset"])
    print("Ordering per pair:", result2.ordering_per_pair["asset"])

Interpreting results
---------------------

All verdict fields are per-asset dicts keyed by the internal asset name
``"asset"`` (the functional wrapper validates a single asset at a time).

+-------------------------------+----------------------------------------------+
| Field                         | Interpretation                               |
+===============================+==============================================+
| ``mean_ari["asset"]``         | Average ARI across all specification pairs.  |
|                               | >= 0.65 is the Paper 1 threshold.            |
+-------------------------------+----------------------------------------------+
| ``passes_partition["asset"]`` | ``bool`` -- True if mean_ari >=              |
|                               | ARI_THRESHOLD (0.65).                        |
+-------------------------------+----------------------------------------------+
| ``passes_ordering["asset"]``  | ``bool`` -- True if mean Spearman >= 0.85.   |
+-------------------------------+----------------------------------------------+
| ``null_1_over_K``             | Expected ARI under random relabelling        |
|                               | (baseline = 1/K); a plain float.             |
+-------------------------------+----------------------------------------------+

See also
--------

* :func:`mrv.invariance.rep_invariance_validator` -- full API reference
* :class:`mrv.invariance.RepInvarianceResult` -- result object fields
* :doc:`paper2_res_invariance` -- resolution invariance in detail
