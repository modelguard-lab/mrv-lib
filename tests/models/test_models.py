"""Tests for mrv.models."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import has_hmmlearn


class TestModels:
    def test_fit_gmm(self):
        from mrv.models.gmm import fit_gmm
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
        labels = fit_gmm(X, n_states=3)
        assert labels is not None
        assert len(labels) == 200
        assert set(labels).issubset({0, 1, 2})

    def test_fit_gmm_insufficient_data(self):
        from mrv.models.gmm import fit_gmm
        assert fit_gmm(pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"]), n_states=3) is None

    @pytest.mark.skipif(not has_hmmlearn(), reason="hmmlearn not installed")
    def test_fit_hmm(self):
        from mrv.models.hmm import fit_hmm
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
        labels = fit_hmm(X, n_states=2)
        assert labels is not None
        assert len(labels) == 200
        assert set(labels).issubset({0, 1})

    @pytest.mark.skipif(not has_hmmlearn(), reason="hmmlearn not installed")
    def test_fit_hmm_insufficient_data(self):
        from mrv.models.hmm import fit_hmm
        assert fit_hmm(pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"]), n_states=3) is None

    def test_model_registry(self):
        from mrv.models import fit
        np.random.seed(42)
        labels = fit(pd.DataFrame(np.random.randn(100, 2), columns=["a", "b"]), model="gmm", n_states=2)
        assert labels is not None

    def test_model_registry_unknown(self):
        from mrv.models import fit
        with pytest.raises(ValueError, match="Unknown model"):
            fit(pd.DataFrame(np.random.randn(50, 2), columns=["a", "b"]), model="nonexistent")

    def test_register_custom_model(self):
        from mrv.models import fit, register_model
        register_model("dummy", lambda features, n_states=2, **kw: np.zeros(len(features), dtype=int))
        labels = fit(pd.DataFrame(np.random.randn(50, 2), columns=["a", "b"]), model="dummy", n_states=2)
        assert (labels == 0).all()
