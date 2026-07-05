"""Tests for the ``mrv`` command-line interface (mrv.cli)."""

import numpy as np
import pandas as pd
import pytest


def _write_daily_csv(path, n=400, seed=0):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.standard_normal(n))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": rng.randint(100_000, 1_000_000, n),
        },
        index=idx,
    )
    df.index.name = "Date"
    df.to_csv(path)


class TestCliInit:
    def test_init_creates_config(self, tmp_path, monkeypatch):
        from mrv import cli
        monkeypatch.chdir(tmp_path)
        cli.main(["init"])
        assert (tmp_path / "config.yaml").exists()

    def test_init_no_overwrite_without_force(self, tmp_path, monkeypatch, capsys):
        from mrv import cli
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text("sentinel: 1\n", encoding="utf-8")
        cli.main(["init"])
        out = capsys.readouterr().out
        assert "already exists" in out
        # Existing file must be untouched.
        assert (tmp_path / "config.yaml").read_text(encoding="utf-8").startswith("sentinel")

    def test_init_force_overwrites(self, tmp_path, monkeypatch):
        from mrv import cli
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text("sentinel: 1\n", encoding="utf-8")
        cli.main(["init", "--force"])
        assert "sentinel" not in (tmp_path / "config.yaml").read_text(encoding="utf-8")


class TestCliRun:
    def test_run_rep_end_to_end(self, tmp_path):
        from mrv import cli
        data = tmp_path / "SPY.csv"
        _write_daily_csv(data)
        reports = tmp_path / "reports"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "validator:\n"
            f"  report_dir: {reports.as_posix()}\n"
            '  report_name: "r_{date}"\n'
            "  report_template: null\n"
            "  rep:\n"
            "    assets:\n"
            f"      SPY: {data.as_posix()}\n"
            "    model: gmm\n"
            "    n_states: 3\n"
            "    factors:\n"
            "      - [vol, drawdown, var, cvar]\n"
            "      - [real_skew, vol_stab, var, cvar]\n"
            "logging:\n"
            "  level: WARNING\n",
            encoding="utf-8",
        )
        # Must run without raising (no raw traceback reaches the caller).
        cli.main(["run", str(cfg), "rep"])
        produced = list(reports.glob("*/result.json"))
        assert produced, "rep run should produce a result.json"

    def test_run_res_not_a_choice(self, tmp_path):
        """`res` was removed from the CLI; argparse must reject it cleanly."""
        from mrv import cli
        cfg = tmp_path / "config.yaml"
        cfg.write_text("validator: {}\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            cli.main(["run", str(cfg), "res"])

    def test_run_unknown_validator_exits(self, tmp_path):
        from mrv import cli
        cfg = tmp_path / "config.yaml"
        cfg.write_text("validator: {}\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            cli.main(["run", str(cfg), "__bogus__"])

    def test_no_args_prints_help(self, capsys):
        from mrv import cli
        cli.main([])
        out = capsys.readouterr().out
        assert "usage" in out.lower()
