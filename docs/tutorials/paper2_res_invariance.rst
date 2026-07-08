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

The validator is *model-driven*: you supply a ``model_fn`` callable that maps a
price :class:`pandas.Series` (with a ``DatetimeIndex``) to integer regime
labels, plus a ``resolution_set`` of the shape ``{asset: {freq: price_series}}``.
The validator fits ``model_fn`` once per (asset, frequency), aligns the labels
onto the finest resolution, and cross-compares them.

Step 1: build the per-frequency price panel
---------------------------------------------

Each inner value is a price :class:`pandas.Series` with a ``DatetimeIndex`` at
that frequency (not a bare numpy array). Here we synthesise 5-minute bars and
resample up to the coarser frequencies.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from sklearn.mixture import GaussianMixture

    np.random.seed(42)
    bars_per_day = 78          # 09:30-16:00, 5-minute bars
    n_days = 15

    idx = None
    for d in pd.bdate_range("2026-01-05", periods=n_days):
        start = pd.Timestamp(f"{d.date()} 09:30", tz="America/New_York")
        times = pd.date_range(start, periods=bars_per_day, freq="5min")
        idx = times if idx is None else idx.append(times)

    regime = np.random.choice([0, 1], size=len(idx), p=[0.7, 0.3])
    returns = np.where(regime == 0,
                       np.random.randn(len(idx)) * 0.001,
                       np.random.randn(len(idx)) * 0.004)
    close_5m = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)

    RULE = {"5m": None, "15m": "15min", "1h": "60min", "1d": "1D"}
    price_map = {}
    for freq, rule in RULE.items():
        price_map[freq] = close_5m if rule is None else close_5m.resample(rule).last().dropna()

    resolution_set = {"SPY": price_map}

Step 2: run the validator
--------------------------

.. code-block:: python

    from mrv.invariance import res_invariance_validator, ResolutionSpec

    def label_model(prices):
        """Fit a 2-state vol regime and return int labels on the input index."""
        log_ret = np.log(prices / prices.shift(1))
        vol = log_ret.rolling(20, min_periods=2).std()
        log_vol = np.log(vol.replace(0, np.nan)).dropna()

        labels = pd.Series(0, index=prices.index, dtype=int)
        if len(log_vol) >= 4:
            X = log_vol.values.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X)
            crisis = int(np.argmax(gmm.means_.ravel()))
            labels.loc[log_vol.index] = (gmm.predict(X) == crisis).astype(int)
        return labels

    result = res_invariance_validator(
        model_fn=label_model,
        resolution_set=resolution_set,
        spec=ResolutionSpec(),    # default Paper 2 four-frequency panel
        run_permutation=True,
        n_perm=500,
        seed=42,
    )

    print(result.summary())
    print("ARI matrix (SPY):", result.ari_matrix["SPY"])
    print("AMI matrix (SPY):", result.ami_matrix["SPY"])

Intraday-vs-overall ARI gap
---------------------------

On your own labels, within-intraday ARI (e.g., 5m vs 15m) is often higher than
daily-vs-intraday ARI (1h or 15m vs 1d). The ``intraday_overall_ari_gap`` field
reports this difference (intraday mean ARI minus overall mean ARI):

.. code-block:: python

    print("Intraday-vs-overall ARI gap (SPY):",
          result.intraday_overall_ari_gap.get("SPY"))

A positive value means the model is more consistent among intraday frequencies
than it is when compared to the daily bar. This library field is a plain
agreement gap on your own labels; it is NOT Paper 2's headline "within-intraday
excess" (empirical intraday ARI minus a simulated MS-Gaussian baseline), which
this library does not compute.

Permutation p-value
-------------------

The validator computes one overall permutation p-value per asset (for the mean
off-diagonal ARI) to guard against inflated agreement from small samples.
``perm_pvalue[asset]`` is a single ``float`` (or ``None`` when the sample is too
small); ``perm_null_ci[asset]`` is the matching 95% null CI:

.. code-block:: python

    pval = result.perm_pvalue.get("SPY")
    ci = result.perm_null_ci.get("SPY")
    if pval is not None:
        print(f"SPY overall permutation p-value: {pval:.4f}")
        if ci is not None:
            print(f"Null 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

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
| ``intraday_overall_ari_gap`` | Intraday mean ARI minus overall mean ARI |
|   ``[asset]``                | (positive = within-intraday more stable).|
+------------------------------+------------------------------------------+
| ``perm_pvalue[asset]``       | Single overall permutation p-value       |
|                              | (``float`` or ``None``) for the mean     |
|                              | off-diagonal ARI.                        |
+------------------------------+------------------------------------------+

See also
--------

* :func:`mrv.invariance.res_invariance_validator` -- full API reference
* :class:`mrv.invariance.ResInvarianceResult` -- result object fields
* :class:`mrv.invariance.ResolutionSpec` -- frequency-panel configuration
* :doc:`paper1_rep_invariance` -- representation invariance tutorial
