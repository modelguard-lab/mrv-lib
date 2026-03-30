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
        assert np.isnan(ordering_consistency(np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0])))

    def test_thresholds_exported(self):
        from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD
        assert ARI_THRESHOLD == 0.65
        assert SPEARMAN_THRESHOLD == 0.85
