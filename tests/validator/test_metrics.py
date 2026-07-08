"""Tests for mrv.validator.metrics."""

import numpy as np
import pytest


class TestMetrics:
    def test_ari_identical(self):
        from mrv.validator.metrics import ari
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        assert ari(labels, labels) == pytest.approx(1.0)

    def test_ari_random(self):
        from mrv.validator.metrics import ari
        np.random.seed(42)
        assert -0.1 < ari(np.random.randint(0, 3, 200), np.random.randint(0, 3, 200)) < 0.2

    def test_ari_too_few_samples(self):
        from mrv.validator.metrics import ari
        assert np.isnan(ari(np.array([0, 1, 2]), np.array([0, 1, 2])))

    def test_ari_different_lengths(self):
        from mrv.validator.metrics import ari
        a = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        b = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        assert ari(a, b) == pytest.approx(1.0)

    def test_ami(self):
        from mrv.validator.metrics import ami
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
        assert ami(labels, labels) == pytest.approx(1.0)

    def test_nmi(self):
        from mrv.validator.metrics import nmi
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
        assert nmi(labels, labels) == pytest.approx(1.0)

    def test_variation_of_information_identical(self):
        from mrv.validator.metrics import variation_of_information
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        assert variation_of_information(labels, labels) == pytest.approx(0.0, abs=1e-10)

    def test_variation_of_information_different(self):
        from mrv.validator.metrics import variation_of_information
        a = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        b = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        assert variation_of_information(a, b) > 0

    def test_ordering_consistency_identical(self):
        from mrv.validator.metrics import ordering_consistency
        np.random.seed(42)
        features = np.random.randn(100)
        labels = (features > 0).astype(int)
        assert ordering_consistency(labels, labels, features) == pytest.approx(1.0)

    def test_ordering_consistency_too_few(self):
        from mrv.validator.metrics import ordering_consistency
        assert np.isnan(ordering_consistency(
            np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0])
        ))

    def test_ordering_consistency_all_nan_risk_returns_nan(self):
        """F4: an all-NaN risk proxy has no ordering to measure -> NaN."""
        from mrv.validator.metrics import ordering_consistency
        np.random.seed(0)
        labels_a = np.random.randint(0, 2, 100)
        labels_b = np.random.randint(0, 2, 100)
        features = np.full(100, np.nan)
        assert np.isnan(ordering_consistency(labels_a, labels_b, features))

    def test_ordering_consistency_masks_nan_rows(self):
        """F4: NaN risk rows are dropped before ranking; the result equals the
        clean-subset computation and is unaffected by injected NaN rows.
        """
        from mrv.validator.metrics import ordering_consistency
        np.random.seed(1)
        n = 120
        features = np.random.randn(n)
        labels = (features > 0).astype(int)
        clean = ordering_consistency(labels, labels, features)

        # Inject NaNs into the risk proxy at some rows; the dropped rows must
        # not change the ranking outcome (still perfectly consistent).
        feat_nan = features.copy()
        feat_nan[::10] = np.nan
        with_nan = ordering_consistency(labels, labels, feat_nan)
        assert np.isfinite(with_nan)
        assert with_nan == pytest.approx(clean, abs=0.05)
        # Identical labels remain perfectly ordering-consistent despite NaNs.
        assert with_nan == pytest.approx(1.0)

    def test_thresholds_exported(self):
        from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD
        assert ARI_THRESHOLD == 0.65
        assert SPEARMAN_THRESHOLD == 0.85
