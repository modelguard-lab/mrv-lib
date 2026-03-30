"""Tests for mrv.utils.config and mrv.utils.log."""

import logging

import pytest


class TestConfig:
    def test_load_defaults(self):
        from mrv.utils.config import load
        cfg = load()
        assert "download" in cfg
        assert "symbols" in cfg["download"]
        assert cfg["download"]["ib"]["port"] == 4002

    def test_get_assets_expands_symbols(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {"symbols": ["SPY", "IEF", "GLD"], "freq": ["5m", "1h", "1d"], "start": "2026-01-01", "end": None}}
        all_assets = get_assets(cfg)
        assert len(all_assets) == 3
        assert all_assets[0]["symbol"] == "SPY"

    def test_get_assets_filter_by_freq(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {"symbols": ["SPY", "IEF"], "freq": ["5m", "1d"], "start": "2026-01-01"}}
        assert len(get_assets(cfg, freq="5m")) == 2
        assert len(get_assets(cfg, freq="15m")) == 0

    def test_get_assets_normalizes_scalar_freq(self):
        from mrv.utils.config import get_assets
        cfg = {"download": {"symbols": ["GLD"], "freq": "1d"}}
        assert get_assets(cfg)[0]["freq"] == ["1d"]

    def test_load_missing_raises(self, tmp_path):
        from mrv.utils.config import load
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.yaml")


class TestLog:
    def test_setup_no_error(self):
        from mrv.utils.log import setup
        setup({"logging": {"level": "WARNING", "log_dir": None, "quiet": []}})

    def test_setup_with_file_logging(self, tmp_path):
        from mrv.utils.log import setup
        log_dir = tmp_path / "test_logs"
        root = logging.getLogger()
        root.handlers.clear()
        setup({"logging": {"level": "DEBUG", "log_dir": str(log_dir), "quiet": []}})
        assert log_dir.exists()
        assert len(list(log_dir.glob("mrv_*.log"))) == 1
        root.handlers.clear()


class TestCanonicalStem:
    def test_basic(self):
        from mrv.utils.download import canonical_stem
        assert canonical_stem("SPY") == "SPY"
        assert canonical_stem("^GSPC") == "GSPC"
        assert canonical_stem("CL=F") == "CL"
        assert canonical_stem("USDJPY") == "USDJPY"
