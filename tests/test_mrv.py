"""Tests for mrv-lib: factors, normalize, reader, config, log, download, metrics, models, report."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_series(n: int = 300) -> pd.Series:
    """Synthetic daily price series for testing."""
    np.random.seed(42)
    returns = np.random.randn(n) * 0.01
    price = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(price, index=idx, name="close")


def _ohlcv_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 0.3
    low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 0.3
    vol = np.random.randint(1000, 10000, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------

class TestFactors:
    def test_log_returns(self):
        from mrv.data.factors import log_returns
        p = _price_series(50)
        r = log_returns(p)
        assert len(r) == 50
        assert pd.isna(r.iloc[0])
        assert not pd.isna(r.iloc[1])

    def test_log_returns_rejects_non_positive(self):
        from mrv.data.factors import log_returns
        p = pd.Series([100.0, 0.0, 50.0])
        with pytest.raises(ValueError, match="Non-positive"):
            log_returns(p)

    def test_volatility(self):
        from mrv.data.factors import log_returns, volatility
        p = _price_series(100)
        r = log_returns(p)
        v = volatility(r, window=20)
        assert v.name == "volatility"
        # First 20 values are NaN (rolling window)
        assert v.iloc[:20].isna().all()
        assert v.iloc[20:].notna().all()

    def test_drawdown(self):
        from mrv.data.factors import drawdown
        p = _price_series(100)
        dd = drawdown(p, window=20)
        assert dd.name == "drawdown"
        # Drawdown should be <= 0
        valid = dd.dropna()
        assert (valid <= 0).all()

    def test_var_cvar(self):
        from mrv.data.factors import log_returns, var, cvar
        p = _price_series(200)
        r = log_returns(p)
        v = var(r, window=60)
        c = cvar(r, window=60)
        assert v.name == "var"
        assert c.name == "cvar"
        # CVaR should be <= VaR (both are negative)
        valid = pd.concat([v, c], axis=1).dropna()
        assert (valid["cvar"] <= valid["var"]).all()

    def test_build_factors(self):
        from mrv.data.factors import build_factors
        p = _price_series(300)
        df = build_factors(p, factors=["volatility", "drawdown", "var", "cvar"])
        assert set(df.columns) == {"volatility", "drawdown", "var", "cvar"}
        assert len(df) == 300

    def test_build_factors_with_config(self):
        from mrv.data.factors import build_factors
        p = _price_series(300)
        cfg = {"factors": {"vol_window": 10, "tail_window": 30}}
        df = build_factors(p, factors=["volatility", "var"], cfg=cfg)
        assert "volatility" in df.columns
        # With window=10, should have values earlier
        assert df["volatility"].iloc[10:15].notna().all()


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_rolling_zscore(self):
        from mrv.data.normalize import rolling_zscore
        df = pd.DataFrame({"a": np.arange(200, dtype=float), "b": np.arange(200, dtype=float) * 2})
        result = rolling_zscore(df, window=50)
        assert result.shape == df.shape
        # First 49 rows NaN (min_periods=50 means index 49 is the first valid)
        assert result.iloc[:49].isna().all().all()
        # After warmup, should be normalized
        valid = result.iloc[49:]
        assert valid.notna().all().all()

    def test_minmax(self):
        from mrv.data.normalize import minmax
        df = pd.DataFrame({"a": np.arange(200, dtype=float)})
        result = minmax(df, window=50)
        valid = result.iloc[50:].dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()

    def test_normalize_dispatch(self):
        from mrv.data.normalize import normalize
        df = pd.DataFrame({"a": np.random.randn(200)})
        # "none" returns copy
        r1 = normalize(df, mode="none")
        pd.testing.assert_frame_equal(r1, df)
        # "rolling_zscore"
        r2 = normalize(df, mode="rolling_zscore", window=50)
        assert r2.iloc[50:].notna().all().all()


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

class TestReader:
    def test_validate_ohlcv_good(self):
        from mrv.data.reader import validate_ohlcv
        df = _ohlcv_df(100)
        issues = validate_ohlcv(df, "TEST")
        assert issues == []

    def test_validate_ohlcv_bad_hl(self):
        from mrv.data.reader import validate_ohlcv
        df = _ohlcv_df(100)
        df.loc[df.index[5], "High"] = df.loc[df.index[5], "Low"] - 1
        issues = validate_ohlcv(df, "TEST")
        assert any("High < Low" in i for i in issues)

    def test_resample_ohlc_passthrough(self):
        from mrv.data.reader import resample_ohlc
        df = _ohlcv_df(50)
        # Fake intraday index with tz
        df.index = pd.date_range("2020-01-01", periods=50, freq="5min", tz="America/New_York")
        result = resample_ohlc(df, "5m")
        assert len(result) == 50

    def test_load_ohlcv_csv(self, tmp_path):
        from mrv.data.reader import load_ohlcv
        df = _ohlcv_df(50)
        df.index.name = "Date"
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path)
        loaded = load_ohlcv(csv_path)
        assert len(loaded) == 50
        assert "Open" in loaded.columns


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_defaults(self):
        from mrv.utils.config import load
        cfg = load()
        assert "download" in cfg
        assert "symbols" in cfg["download"]
        assert "data_dir" in cfg["download"]
        assert "ib" in cfg["download"]
        assert "logging" in cfg
        assert "factors" in cfg
        assert "normalize" in cfg
        assert cfg["download"]["ib"]["port"] == 4002

    def test_get_assets_expands_symbols(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {
            "symbols": ["SPY", "IEF", "GLD"],
            "freq": ["5m", "1h", "1d"],
            "start": "2026-01-01",
            "end": None,
        }}
        all_assets = get_assets(cfg)
        assert len(all_assets) == 3
        assert all_assets[0]["symbol"] == "SPY"
        assert all_assets[0]["freq"] == ["5m", "1h", "1d"]
        assert all_assets[0]["start"] == "2026-01-01"

    def test_get_assets_filter_by_freq(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {
            "symbols": ["SPY", "IEF"],
            "freq": ["5m", "1d"],
            "start": "2026-01-01",
        }}
        intraday = get_assets(cfg, freq="5m")
        assert len(intraday) == 2
        daily = get_assets(cfg, freq="1d")
        assert len(daily) == 2
        none = get_assets(cfg, freq="15m")
        assert len(none) == 0

    def test_get_assets_normalizes_scalar_freq(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {
            "symbols": ["GLD"],
            "freq": "1d",
        }}
        assets = get_assets(cfg)
        assert assets[0]["freq"] == ["1d"]

    def test_load_missing_raises(self, tmp_path):
        from mrv.utils.config import load
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------

class TestLog:
    def test_setup_no_error(self):
        from mrv.utils.log import setup
        cfg = {"logging": {"level": "WARNING", "log_dir": None, "quiet": []}}
        setup(cfg)  # Should not raise

    def test_setup_with_file_logging(self, tmp_path):
        from mrv.utils.log import setup
        import logging
        log_dir = tmp_path / "test_logs"
        cfg = {"logging": {"level": "DEBUG", "log_dir": str(log_dir), "quiet": []}}
        # Reset root handlers for clean test
        root = logging.getLogger()
        root.handlers.clear()
        setup(cfg)
        assert log_dir.exists()
        log_files = list(log_dir.glob("mrv_*.log"))
        assert len(log_files) == 1
        # Cleanup
        root.handlers.clear()


# ---------------------------------------------------------------------------
# Additional factor tests
# ---------------------------------------------------------------------------

class TestFactorsExtended:
    def test_max_drawdown(self):
        from mrv.data.factors import max_drawdown
        p = _price_series(200)
        mdd = max_drawdown(p, window=60)
        assert mdd.name == "max_drawdown_window"
        valid = mdd.dropna()
        assert (valid <= 0).all()

    def test_realized_skew(self):
        from mrv.data.factors import log_returns, realized_skew
        p = _price_series(200)
        r = log_returns(p)
        sk = realized_skew(r, window=60)
        assert sk.name == "realized_skew"
        assert sk.dropna().shape[0] > 0

    def test_stability(self):
        from mrv.data.factors import log_returns, volatility, stability
        p = _price_series(200)
        r = log_returns(p)
        v = volatility(r, window=20)
        s = stability(v, window=60)
        assert s.name == "stability"
        valid = s.dropna()
        assert (valid >= 0).all()



# ---------------------------------------------------------------------------
# canonical_stem
# ---------------------------------------------------------------------------

class TestCanonicalStem:
    def test_basic(self):
        from mrv.utils.download import canonical_stem
        assert canonical_stem("SPY") == "SPY"
        assert canonical_stem("^GSPC") == "GSPC"
        assert canonical_stem("CL=F") == "CL"
        assert canonical_stem("USDJPY") == "USDJPY"


# ---------------------------------------------------------------------------
# reader: load_daily
# ---------------------------------------------------------------------------

class TestLoadDaily:
    def test_load_daily_csv(self, tmp_path):
        from mrv.data.reader import load_daily
        df = _ohlcv_df(50)
        df.index.name = "Date"
        csv_path = tmp_path / "test_daily.csv"
        df.to_csv(csv_path)
        price = load_daily(csv_path)
        assert len(price) == 50
        assert price.name == "test_daily"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_ari_identical(self):
        from mrv.validator.metrics import ari
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        assert ari(labels, labels) == pytest.approx(1.0)

    def test_ari_random(self):
        from mrv.validator.metrics import ari
        np.random.seed(42)
        a = np.random.randint(0, 3, 200)
        b = np.random.randint(0, 3, 200)
        result = ari(a, b)
        assert -0.1 < result < 0.2  # random labels ~ 0

    def test_ari_too_few_samples(self):
        from mrv.validator.metrics import ari
        a = np.array([0, 1, 2])
        b = np.array([0, 1, 2])
        assert np.isnan(ari(a, b))

    def test_ari_different_lengths(self):
        from mrv.validator.metrics import ari
        a = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        b = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        result = ari(a, b)
        assert result == pytest.approx(1.0)

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
        result = ordering_consistency(labels, labels, features)
        assert result == pytest.approx(1.0)

    def test_ordering_consistency_too_few(self):
        from mrv.validator.metrics import ordering_consistency
        a = np.array([0, 1, 2])
        b = np.array([0, 1, 2])
        f = np.array([1.0, 2.0, 3.0])
        assert np.isnan(ordering_consistency(a, b, f))

    def test_thresholds_exported(self):
        from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD
        assert ARI_THRESHOLD == 0.65
        assert SPEARMAN_THRESHOLD == 0.85


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

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
        X = pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"])
        assert fit_gmm(X, n_states=3) is None

    def test_fit_hmm(self):
        from mrv.models.hmm import fit_hmm
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
        labels = fit_hmm(X, n_states=2)
        assert labels is not None
        assert len(labels) == 200
        assert set(labels).issubset({0, 1})

    def test_fit_hmm_insufficient_data(self):
        from mrv.models.hmm import fit_hmm
        X = pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"])
        assert fit_hmm(X, n_states=3) is None

    def test_model_registry(self):
        from mrv.models import fit
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 2), columns=["a", "b"])
        # Built-in gmm should work
        labels = fit(X, model="gmm", n_states=2)
        assert labels is not None

    def test_model_registry_unknown(self):
        from mrv.models import fit
        X = pd.DataFrame(np.random.randn(50, 2), columns=["a", "b"])
        with pytest.raises(ValueError, match="Unknown model"):
            fit(X, model="nonexistent")

    def test_register_custom_model(self):
        from mrv.models import fit, register_model

        def dummy_model(features, n_states=2, **kwargs):
            return np.zeros(len(features), dtype=int)

        register_model("dummy", dummy_model)
        X = pd.DataFrame(np.random.randn(50, 2), columns=["a", "b"])
        labels = fit(X, model="dummy", n_states=2)
        assert (labels == 0).all()


# ---------------------------------------------------------------------------
# factor registry
# ---------------------------------------------------------------------------

class TestFactorRegistry:
    def test_resolve_aliases(self):
        from mrv.data.factors import resolve_name
        assert resolve_name("vol") == "volatility"
        assert resolve_name("maxdd") == "max_drawdown_window"
        assert resolve_name("real_skew") == "realized_skew"
        assert resolve_name("vol_stab") == "stability"
        assert resolve_name("unknown") == "unknown"

    def test_register_custom_factor(self):
        from mrv.data.factors import register_factor, build_factors
        def momentum(returns, price, windows):
            return price.pct_change(windows.get("mom_window", 20)).rename("momentum")
        register_factor("momentum", momentum)
        p = _price_series(100)
        df = build_factors(p, factors=["momentum"])
        assert "momentum" in df.columns

    def test_build_factors_unknown_skipped(self):
        from mrv.data.factors import build_factors
        p = _price_series(100)
        df = build_factors(p, factors=["volatility", "nonexistent_factor"])
        assert "volatility" in df.columns
        assert len(df.columns) == 1

    def test_build_factors_default(self):
        from mrv.data.factors import build_factors, DEFAULT_FACTORS
        p = _price_series(300)
        df = build_factors(p)
        assert len(df.columns) == len(DEFAULT_FACTORS)


# ---------------------------------------------------------------------------
# normalize extended
# ---------------------------------------------------------------------------

class TestNormalizeExtended:
    def test_normalize_with_config(self):
        from mrv.data.normalize import normalize
        df = pd.DataFrame({"a": np.random.randn(200)})
        cfg = {"normalize": {"mode": "minmax", "window": 50}}
        result = normalize(df, cfg=cfg)
        valid = result.iloc[50:].dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()

    def test_normalize_unknown_mode(self):
        from mrv.data.normalize import normalize
        df = pd.DataFrame({"a": np.random.randn(200)})
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            normalize(df, mode="invalid_mode")

    def test_rolling_zscore_constant_column(self):
        from mrv.data.normalize import rolling_zscore
        df = pd.DataFrame({"a": np.ones(200)})
        result = rolling_zscore(df, window=50)
        # Constant column has std=0 -> NaN (division by zero guard)
        assert result.iloc[50:].isna().all().all()


# ---------------------------------------------------------------------------
# reader extended
# ---------------------------------------------------------------------------

class TestReaderExtended:
    def test_validate_ohlcv_missing_column(self):
        from mrv.data.reader import validate_ohlcv
        df = pd.DataFrame({"Open": [1], "High": [2], "Low": [1]})
        issues = validate_ohlcv(df)
        assert any("Close" in i for i in issues)

    def test_validate_ohlcv_empty(self):
        from mrv.data.reader import validate_ohlcv
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        issues = validate_ohlcv(df)
        assert any("empty" in i for i in issues)

    def test_validate_ohlcv_duplicate_timestamps(self):
        from mrv.data.reader import validate_ohlcv
        df = _ohlcv_df(10)
        df.index = [df.index[0]] * 10  # all same timestamp
        issues = validate_ohlcv(df)
        assert any("duplicate" in i for i in issues)

    def test_resample_ohlc_15m(self):
        from mrv.data.reader import resample_ohlc
        df = _ohlcv_df(60)
        df.index = pd.date_range("2020-01-02 09:30", periods=60, freq="5min", tz="America/New_York")
        result = resample_ohlc(df, "15m")
        assert len(result) < 60
        assert len(result) > 0

    def test_resample_ohlc_1h(self):
        from mrv.data.reader import resample_ohlc
        df = _ohlcv_df(60)
        df.index = pd.date_range("2020-01-02 09:30", periods=60, freq="5min", tz="America/New_York")
        result = resample_ohlc(df, "1h")
        assert len(result) > 0
        assert len(result) < 60

    def test_resample_ohlc_invalid_freq(self):
        from mrv.data.reader import resample_ohlc
        df = _ohlcv_df(10)
        df.index = pd.date_range("2020-01-01", periods=10, freq="5min", tz="America/New_York")
        with pytest.raises(ValueError, match="Unsupported frequency"):
            resample_ohlc(df, "3m")

    def test_infer_price_column(self):
        from mrv.data.reader import _infer_price_column
        df = pd.DataFrame({"Adj Close": [1], "Open": [1]})
        assert _infer_price_column(df) == "Adj Close"
        df2 = pd.DataFrame({"close": [1], "open": [1]})
        assert _infer_price_column(df2) == "close"

    def test_infer_price_column_raises(self):
        from mrv.data.reader import _infer_price_column
        df = pd.DataFrame({"volume": [1], "open": [1]})
        with pytest.raises(ValueError, match="Could not infer"):
            _infer_price_column(df)


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------

class TestReportHelpers:
    def test_tex_escaping(self):
        from mrv.validator.report import _tex
        assert _tex("a & b") == "a \\& b"
        assert _tex("100%") == "100\\%"
        assert _tex("$x$") == "\\$x\\$"
        assert _tex("a_b") == "a\\_b"

    def test_ari_table(self):
        from mrv.validator.report import _ari_table
        labels = ["Set 0", "Set 1"]
        values = [[1.0, 0.5], [0.5, 1.0]]
        result = _ari_table(labels, values, threshold=0.65)
        assert "\\begin{tabular}" in result
        assert "\\bottomrule" in result
        assert "cellcolor" in result  # 0.5 < 0.65

    def test_ari_table_all_pass(self):
        from mrv.validator.report import _ari_table
        labels = ["Set 0", "Set 1"]
        values = [[1.0, 0.8], [0.8, 1.0]]
        result = _ari_table(labels, values, threshold=0.65)
        assert "cellcolor" not in result

    def test_eval_conditionals(self):
        from mrv.validator.report import _eval_conditionals
        text = "before\n%% IF_PASS\nyes\n%% ELSE\nno\n%% ENDIF\nafter"
        result = _eval_conditionals(text, {"PASS": True})
        assert "yes" in result
        assert "no" not in result
        result2 = _eval_conditionals(text, {"PASS": False})
        assert "no" in result2
        assert "yes" not in result2

    def test_eval_conditionals_elif(self):
        from mrv.validator.report import _eval_conditionals
        text = "start\n%% IF_A\nA\n%% ELIF_B\nB\n%% ELSE\nC\n%% ENDIF\nend"
        assert "A" in _eval_conditionals(text, {"A": True, "B": False})
        result_b = _eval_conditionals(text, {"A": False, "B": True})
        assert "B" in result_b
        assert "A" not in result_b
        result_c = _eval_conditionals(text, {"A": False, "B": False})
        assert "C" in result_c


# ---------------------------------------------------------------------------
# base validator
# ---------------------------------------------------------------------------

class TestBaseValidator:
    def test_make_run_dir(self, tmp_path):
        from mrv.validator.base import BaseValidator

        class DummyValidator(BaseValidator):
            name = "test"
            def validate(self, prices=None, labels=None):
                return {}

        cfg = {"validator": {"report_dir": str(tmp_path), "report_name": "test_{date}"}}
        v = DummyValidator(cfg)
        run_dir = v._make_run_dir()
        assert run_dir.exists()
        assert "test_" in run_dir.name
        assert "_test" in run_dir.name


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_consistency(self):
        import mrv
        assert mrv.__version__ == "0.1.0"
