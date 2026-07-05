"""Tests for mrv.invariance.rep_invariance_validator (Paper 1 wrapper).

Canonical fixture: S&P 500-style synthetic daily returns with three
feature representations (full tail-risk / no-maxdd / higher-order moments).

All tests operate on deterministic synthetic data; no network access is needed.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_spy_like_price(n: int = 800, seed: int = 42) -> np.ndarray:
    """S&P 500-style synthetic daily close prices.

    Two-regime DGP:
      calm   (60 %): mu=+0.03%, sigma=0.8%
      stress (40 %): mu=-0.05%, sigma=2.5%

    Returns
    -------
    np.ndarray
        1-D float array of shape (n,), starting at 100.
    """
    rng = np.random.RandomState(seed)
    n_calm = int(n * 0.6)
    n_stress = n - n_calm
    rets = np.concatenate([
        rng.normal(0.0003, 0.008, n_calm),
        rng.normal(-0.0005, 0.025, n_stress),
    ])
    rng.shuffle(rets)
    price = 100.0 * np.exp(np.cumsum(rets))
    return price


def _rolling_vol(price: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling realised volatility (std of log-returns)."""
    rets = np.diff(np.log(price))
    vol = np.full(len(price), np.nan)
    for i in range(window, len(price)):
        vol[i] = float(np.std(rets[i - window:i], ddof=1))
    return vol


def _rolling_drawdown(price: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling drawdown from window-high."""
    dd = np.full(len(price), 0.0)
    for i in range(window, len(price)):
        peak = float(np.max(price[i - window:i + 1]))
        dd[i] = (peak - price[i]) / max(peak, 1e-10)
    return dd


def _rolling_var(returns: np.ndarray, window: int = 60, alpha: float = 0.05) -> np.ndarray:
    """Rolling historical VaR (loss, positive = bad)."""
    var = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        var[i] = float(-np.percentile(returns[i - window:i], 100 * alpha))
    return var


def _rolling_cvar(returns: np.ndarray, window: int = 60, alpha: float = 0.05) -> np.ndarray:
    """Rolling CVaR (expected shortfall, positive = bad)."""
    cvar = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        chunk = returns[i - window:i]
        cutoff = float(np.percentile(chunk, 100 * alpha))
        tail = chunk[chunk <= cutoff]
        cvar[i] = float(-np.mean(tail)) if len(tail) > 0 else float(-cutoff)
    return cvar


def _rolling_skew(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling realised skewness."""
    from scipy.stats import skew as scipy_skew
    sk = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        sk[i] = float(scipy_skew(returns[i - window:i]))
    return sk


def _build_spy_admissible_class(
    n: int = 800,
    seed: int = 42,
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build three S&P 500-style feature matrices and a returns array.

    Returns
    -------
    admissible_class : dict
        ``{"rep_A": feat_A, "rep_B": feat_B, "rep_C": feat_C}``
        Each value is a 2-D float array, NaN-trimmed to a common length.
    returns : np.ndarray
        1-D log-return array aligned with the feature rows.
    price : np.ndarray
        Raw price series (length n).
    """
    price = _make_spy_like_price(n=n, seed=seed)
    rets = np.diff(np.log(price))  # length n-1
    full_rets = np.concatenate([[np.nan], rets])  # align with price (len n)

    vol = _rolling_vol(price, window=20)
    dd = _rolling_drawdown(price, window=60)
    var = _rolling_var(full_rets, window=60)
    cvar = _rolling_cvar(full_rets, window=60)
    skew = _rolling_skew(full_rets, window=60)

    # Rolling max-drawdown (window of 60)
    maxdd = np.full(len(price), np.nan)
    for i in range(60, len(price)):
        window_p = price[i - 60:i + 1]
        peaks = np.maximum.accumulate(window_p)
        dds = (peaks - window_p) / np.where(peaks > 0, peaks, 1)
        maxdd[i] = float(np.max(dds))

    # Vol-stability: rolling std of vol
    vol_stab = np.full(len(price), np.nan)
    for i in range(80, len(price)):
        vol_stab[i] = float(np.std(vol[i - 60:i], ddof=1))

    # Stack into feature matrices
    feat_A = np.column_stack([vol, dd, maxdd, var, cvar])
    feat_B = np.column_stack([vol, dd, var, cvar])
    feat_C = np.column_stack([skew, vol_stab, var, cvar])

    # Find common valid rows (no NaN in any matrix)
    valid = (
        np.all(np.isfinite(feat_A), axis=1)
        & np.all(np.isfinite(feat_B), axis=1)
        & np.all(np.isfinite(feat_C), axis=1)
        & np.isfinite(full_rets)
    )
    idx = np.where(valid)[0]

    return (
        {
            "rep_A": feat_A[idx],
            "rep_B": feat_B[idx],
            "rep_C": feat_C[idx],
        },
        full_rets[idx],
        price,
    )


def _gmm_model_fn(K: int = 3):
    """Return a callable ``(features) -> labels`` using K-state GMM."""
    from sklearn.mixture import GaussianMixture

    def model_fn(features: np.ndarray) -> np.ndarray:
        gm = GaussianMixture(n_components=K, random_state=0, n_init=3)
        gm.fit(features)
        return gm.predict(features).astype(int)

    return model_fn


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestRepValidatorImport:
    """Verify the function is reachable through all documented import paths."""

    def test_import_from_invariance_subpackage(self):
        from mrv.invariance import rep_invariance_validator
        assert callable(rep_invariance_validator)

    def test_import_from_mrv_top_level(self):
        import mrv
        assert callable(mrv.rep_invariance_validator)

    def test_result_class_importable(self):
        from mrv.invariance import RepInvarianceResult
        assert RepInvarianceResult is not None

    def test_result_class_importable_top_level(self):
        from mrv import RepInvarianceResult
        assert RepInvarianceResult is not None

    def test_all_exports_present(self):
        from mrv import invariance
        for name in ["rep_invariance_validator", "RepInvarianceResult"]:
            assert hasattr(invariance, name), f"missing from mrv.invariance: {name}"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestRepValidatorArgValidation:
    """Edge-case and error-path tests for rep_invariance_validator."""

    def _simple_model(self, features: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(0)
        return rng.randint(0, 2, len(features))

    def test_requires_at_least_two_specs(self):
        from mrv.invariance import rep_invariance_validator
        with pytest.raises(ValueError, match=">="):
            rep_invariance_validator(
                model_fn=self._simple_model,
                admissible_class={"only_one": np.random.randn(100, 2)},
                K=2,
            )

    def test_accepts_exactly_two_specs(self):
        from mrv.invariance import rep_invariance_validator
        rng = np.random.RandomState(1)
        ac = {
            "a": rng.randn(100, 3),
            "b": rng.randn(100, 3),
        }
        result = rep_invariance_validator(self._simple_model, ac, K=2)
        # C(2, 2) = 1 pair for the two specs {a, b}.
        assert len(result.ari_per_pair["asset"]) == 1
        assert np.isfinite(result.mean_ari["asset"])

    def test_null_1_over_K_formula(self):
        """null_1_over_K must equal 1/K for any K."""
        from mrv.invariance import rep_invariance_validator
        rng = np.random.RandomState(2)
        ac = {"a": rng.randn(50, 2), "b": rng.randn(50, 2)}
        for K in [2, 3, 5]:
            r = rep_invariance_validator(self._simple_model, ac, K=K)
            assert math.isclose(r.null_1_over_K, 1.0 / K, rel_tol=1e-9), (
                f"K={K}: expected null_1_over_K={1/K}, got {r.null_1_over_K}"
            )

    def test_K_stored_on_result(self):
        from mrv.invariance import rep_invariance_validator
        rng = np.random.RandomState(3)
        ac = {"a": rng.randn(80, 2), "b": rng.randn(80, 2)}
        r = rep_invariance_validator(self._simple_model, ac, K=4)
        assert r.K == 4

    def test_returns_none_produces_nan_ordering(self):
        """When returns is not supplied, ordering_per_pair values should be NaN."""
        from mrv.invariance import rep_invariance_validator
        rng = np.random.RandomState(4)
        ac = {"a": rng.randn(80, 2), "b": rng.randn(80, 2)}
        r = rep_invariance_validator(self._simple_model, ac, K=2)
        for pair_vals in r.ordering_per_pair.values():
            for v in pair_vals.values():
                assert math.isnan(v), f"Expected NaN ordering without returns, got {v}"

    def test_returns_provided_produces_finite_ordering(self):
        from mrv.invariance import rep_invariance_validator
        rng = np.random.RandomState(5)
        n = 100
        ac = {"a": rng.randn(n, 2), "b": rng.randn(n, 2)}
        returns = rng.randn(n) * 0.01
        r = rep_invariance_validator(self._simple_model, ac, returns=returns, K=2)
        for pair_vals in r.ordering_per_pair.values():
            for v in pair_vals.values():
                assert math.isfinite(v), f"Expected finite ordering with returns, got {v}"

    def test_ari_thresholds_exposed(self):
        from mrv.invariance import rep_invariance_validator
        from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD
        rng = np.random.RandomState(6)
        ac = {"a": rng.randn(80, 2), "b": rng.randn(80, 2)}
        r = rep_invariance_validator(self._simple_model, ac, K=2)
        assert r.ari_threshold == ARI_THRESHOLD
        assert r.spearman_threshold == SPEARMAN_THRESHOLD


# ---------------------------------------------------------------------------
# Core output structure
# ---------------------------------------------------------------------------


class TestRepValidatorOutputStructure:
    """Tests that validate the shape and content of RepInvarianceResult."""

    def _run(self, n_specs: int = 3, n_obs: int = 120, K: int = 3, seed: int = 0):
        from mrv.invariance import rep_invariance_validator
        rng = np.random.RandomState(seed)

        def model_fn(features: np.ndarray) -> np.ndarray:
            # Deterministic label based on first feature quintile
            q = np.percentile(features[:, 0], np.linspace(0, 100, K + 1))
            labels = np.digitize(features[:, 0], q[1:-1]).astype(int)
            return labels

        ac = {f"spec_{i}": rng.randn(n_obs, 4) for i in range(n_specs)}
        returns = rng.randn(n_obs) * 0.01
        return rep_invariance_validator(model_fn, ac, returns=returns, K=K)

    def test_ari_per_pair_has_all_pairs(self):
        r = self._run(n_specs=3)
        pairs = list(r.ari_per_pair["asset"].keys())
        # C(3,2) = 3 pairs
        assert len(pairs) == 3

    def test_ari_per_pair_values_in_range(self):
        r = self._run(n_specs=3)
        for v in r.ari_per_pair["asset"].values():
            assert -1.0 <= v <= 1.0, f"ARI out of [-1,1]: {v}"

    def test_ordering_per_pair_has_all_pairs(self):
        r = self._run(n_specs=3)
        pairs = list(r.ordering_per_pair["asset"].keys())
        assert len(pairs) == 3

    def test_ordering_per_pair_values_finite(self):
        r = self._run(n_specs=3)
        for v in r.ordering_per_pair["asset"].values():
            assert math.isfinite(v), f"Unexpected non-finite ordering: {v}"

    def test_mean_ari_key_present(self):
        r = self._run(n_specs=3)
        assert "asset" in r.mean_ari
        assert math.isfinite(r.mean_ari["asset"])

    def test_min_ari_key_present(self):
        r = self._run(n_specs=3)
        assert "asset" in r.min_ari
        assert math.isfinite(r.min_ari["asset"])

    def test_min_ari_leq_mean_ari(self):
        r = self._run(n_specs=3)
        assert r.min_ari["asset"] <= r.mean_ari["asset"] + 1e-9

    def test_passes_partition_type(self):
        r = self._run()
        assert isinstance(r.passes_partition["asset"], bool)

    def test_passes_ordering_type(self):
        r = self._run()
        assert isinstance(r.passes_ordering["asset"], bool)

    def test_null_1_over_K_is_one_over_K(self):
        r = self._run(K=3)
        assert math.isclose(r.null_1_over_K, 1 / 3, rel_tol=1e-9)

    def test_summary_method_runs(self):
        r = self._run()
        s = r.summary()
        assert isinstance(s, str)
        assert "RepInvarianceResult" in s

    def test_summary_contains_null_1_over_K(self):
        r = self._run(K=3)
        s = r.summary()
        assert "null_1/K" in s or "null" in s.lower()


# ---------------------------------------------------------------------------
# Semantic invariants
# ---------------------------------------------------------------------------


class TestRepValidatorSemantics:
    """Semantic guarantees: identical labels -> ARI=1; random -> ARI~0."""

    def test_identical_labels_gives_ari_one(self):
        """When model_fn always produces the same partition, ARI should be 1."""
        from mrv.invariance import rep_invariance_validator

        def constant_model(features: np.ndarray) -> np.ndarray:
            # Always label by first-column median split -- same result regardless of input
            med = np.median(features[:, 0])
            return (features[:, 0] >= med).astype(int)

        rng = np.random.RandomState(7)
        # Same feature matrix for both specs -- identical input -> identical labels
        feat = rng.randn(200, 2)
        ac = {"spec_a": feat.copy(), "spec_b": feat.copy()}
        r = rep_invariance_validator(constant_model, ac, K=2)
        assert abs(r.mean_ari["asset"] - 1.0) < 0.01, (
            f"Identical-input ARI expected ~1.0, got {r.mean_ari['asset']}"
        )

    def test_random_independent_labels_low_ari(self):
        """Independent random labels from different feature sets should have ARI near 0."""
        from mrv.invariance import rep_invariance_validator

        seeds = [10, 20]

        def model_fn(features: np.ndarray) -> np.ndarray:
            # Partition into 3 equal thirds by sorting first feature
            n = len(features)
            rank = np.argsort(np.argsort(features[:, 0]))
            return (rank // (n // 3 + 1)).astype(int)

        rng_a = np.random.RandomState(seeds[0])
        rng_b = np.random.RandomState(seeds[1])
        ac = {
            "rand_a": rng_a.randn(500, 3),
            "rand_b": rng_b.randn(500, 3),
        }
        r = rep_invariance_validator(model_fn, ac, K=3)
        assert abs(r.mean_ari["asset"]) < 0.25, (
            f"Independent random features should give low ARI, got {r.mean_ari['asset']}"
        )

    def test_ari_pair_tuple_keys(self):
        """ARI per pair keys must be 2-tuples of spec names."""
        from mrv.invariance import rep_invariance_validator
        rng = np.random.RandomState(11)
        ac = {"alpha": rng.randn(80, 2), "beta": rng.randn(80, 2), "gamma": rng.randn(80, 2)}

        def model_fn(f):
            return (f[:, 0] > 0).astype(int)

        r = rep_invariance_validator(model_fn, ac, K=2)
        for key in r.ari_per_pair["asset"]:
            assert isinstance(key, tuple) and len(key) == 2, (
                f"ari_per_pair key must be 2-tuple, got: {key!r}"
            )
            assert key[0] in ac and key[1] in ac

    def test_passes_partition_consistent_with_mean_ari(self):
        """passes_partition must agree with mean_ari >= ARI_THRESHOLD."""
        from mrv.invariance import rep_invariance_validator
        from mrv.validator.metrics import ARI_THRESHOLD
        rng = np.random.RandomState(12)
        feat = rng.randn(120, 3)
        ac = {"x": feat, "y": feat}

        def model_fn(f):
            return (f[:, 0] > 0).astype(int)

        r = rep_invariance_validator(model_fn, ac, K=2)
        expected = r.mean_ari["asset"] >= ARI_THRESHOLD
        assert r.passes_partition["asset"] == expected


# ---------------------------------------------------------------------------
# Canonical S&P 500-style fixture demo
# ---------------------------------------------------------------------------


class TestSPY500CanonicalFixture:
    """End-to-end demo on a synthetic S&P 500-style daily price series.

    Three feature representations:
      rep_A -- full tail-risk: vol + drawdown + max_drawdown + VaR + CVaR
      rep_B -- no max-drawdown: vol + drawdown + VaR + CVaR
      rep_C -- higher-order: skewness + vol-stability + VaR + CVaR

    Expected behaviour (Paper 1 main finding): partition ARI < threshold
    across representations while ordering Spearman is higher, illustrating
    that regime *boundaries* are sensitive to feature choice even when the
    relative risk *ordering* is more stable.
    """

    @pytest.fixture(scope="class")
    def spy_result(self):
        from mrv.invariance import rep_invariance_validator
        admissible_class, returns, _ = _build_spy_admissible_class(n=800, seed=42)
        model_fn = _gmm_model_fn(K=3)
        return rep_invariance_validator(
            model_fn=model_fn,
            admissible_class=admissible_class,
            returns=returns,
            K=3,
        )

    def test_result_not_none(self, spy_result):
        # Structural / value checks: finite mean ARI, min <= mean.
        mean_ari = spy_result.mean_ari["asset"]
        min_ari = spy_result.min_ari["asset"]
        assert np.isfinite(mean_ari)
        assert np.isfinite(min_ari)
        assert min_ari <= mean_ari + 1e-9

    def test_null_1_over_K_for_K3(self, spy_result):
        assert math.isclose(spy_result.null_1_over_K, 1 / 3, rel_tol=1e-9)

    def test_ari_per_pair_has_three_pairs(self, spy_result):
        # C(3,2) = 3 pairs for rep_A, rep_B, rep_C
        assert len(spy_result.ari_per_pair["asset"]) == 3

    def test_all_pair_keys_are_known_reps(self, spy_result):
        known = {"rep_A", "rep_B", "rep_C"}
        for a, b in spy_result.ari_per_pair["asset"]:
            assert a in known and b in known

    def test_ari_values_in_minus1_to_1(self, spy_result):
        for v in spy_result.ari_per_pair["asset"].values():
            assert -1.0 <= v <= 1.0

    def test_ordering_values_finite(self, spy_result):
        for v in spy_result.ordering_per_pair["asset"].values():
            assert math.isfinite(v), f"Ordering pair has non-finite value: {v}"

    def test_ari_a_b_higher_than_a_c(self, spy_result):
        """rep_A and rep_B differ only in max-drawdown feature: their ARI
        should exceed the ARI of rep_A vs rep_C (which uses entirely
        different factor families).  Paper 1 Table 2 pattern.
        """
        pairs = spy_result.ari_per_pair["asset"]
        # Find the (rep_A, rep_B) pair -- key order may vary
        ab = pairs.get(("rep_A", "rep_B")) or pairs.get(("rep_B", "rep_A"))
        ac = pairs.get(("rep_A", "rep_C")) or pairs.get(("rep_C", "rep_A"))
        assert ab is not None, "Pair (rep_A, rep_B) missing"
        assert ac is not None, "Pair (rep_A, rep_C) missing"
        assert ab > ac, (
            f"Expected ARI(A,B)={ab:.3f} > ARI(A,C)={ac:.3f}: "
            "similar factor sets should be more partition-consistent"
        )

    def test_summary_output(self, spy_result):
        s = spy_result.summary()
        assert "K=3" in s
        assert "null_1/K=0.333" in s

    def test_passes_partition_is_bool(self, spy_result):
        assert isinstance(spy_result.passes_partition["asset"], bool)

    def test_passes_ordering_is_bool(self, spy_result):
        assert isinstance(spy_result.passes_ordering["asset"], bool)

    def test_mean_ari_is_finite(self, spy_result):
        assert math.isfinite(spy_result.mean_ari["asset"])

    def test_min_ari_leq_mean_ari(self, spy_result):
        assert spy_result.min_ari["asset"] <= spy_result.mean_ari["asset"] + 1e-9

    def test_null_below_mean_ari_for_related_reps(self, spy_result):
        """Mean ARI across reps should exceed the 1/K random null."""
        assert spy_result.mean_ari["asset"] > spy_result.null_1_over_K, (
            f"Mean ARI {spy_result.mean_ari['asset']:.3f} not above "
            f"1/K null {spy_result.null_1_over_K:.3f}"
        )


# ---------------------------------------------------------------------------
# Multi-spec stress test
# ---------------------------------------------------------------------------


class TestRepValidatorMultiSpec:
    """Verify correct pair counting and structure with varying spec counts."""

    def _run_n_specs(self, n_specs: int, n_obs: int = 100, K: int = 2, seed: int = 0):
        from mrv.invariance import rep_invariance_validator
        rng = np.random.RandomState(seed)
        ac = {f"s{i}": rng.randn(n_obs, 3) for i in range(n_specs)}

        def model_fn(f):
            return (f[:, 0] > 0).astype(int)

        return rep_invariance_validator(model_fn, ac, K=K)

    def test_two_specs_one_pair(self):
        r = self._run_n_specs(2)
        assert len(r.ari_per_pair["asset"]) == 1

    def test_three_specs_three_pairs(self):
        r = self._run_n_specs(3)
        assert len(r.ari_per_pair["asset"]) == 3

    def test_four_specs_six_pairs(self):
        r = self._run_n_specs(4)
        assert len(r.ari_per_pair["asset"]) == 6

    def test_five_specs_ten_pairs(self):
        r = self._run_n_specs(5)
        assert len(r.ari_per_pair["asset"]) == 10

    def test_pair_count_formula(self):
        """len(pairs) == C(n_specs, 2) = n*(n-1)//2 for n in 2..6."""
        for n in range(2, 7):
            r = self._run_n_specs(n)
            expected = n * (n - 1) // 2
            actual = len(r.ari_per_pair["asset"])
            assert actual == expected, f"n_specs={n}: expected {expected} pairs, got {actual}"
