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

    def _write_rep_cfg(self, cfg_path, data_path, reports):
        cfg_path.write_text(
            "validator:\n"
            f"  report_dir: {reports.as_posix()}\n"
            '  report_name: "r_{date}"\n'
            "  report_template: null\n"
            "  rep:\n"
            "    assets:\n"
            f"      SPY: {data_path.as_posix()}\n"
            "    model: gmm\n"
            "    n_states: 3\n"
            "    factors:\n"
            "      - [vol, drawdown, var, cvar]\n"
            "      - [real_skew, vol_stab, var, cvar]\n"
            "logging:\n"
            "  level: WARNING\n",
            encoding="utf-8",
        )

    def test_run_report_selects_latest_and_runs(self, tmp_path):
        """`mrv run <cfg> report` must not raise and must pick the newest
        result.json under the report dir (guards _find_latest_json selection)."""
        import os
        import time

        from mrv import cli
        from mrv.cli import _find_latest_json
        from mrv.utils.config import load as load_cfg

        data = tmp_path / "SPY.csv"
        _write_daily_csv(data)
        reports = tmp_path / "reports"
        cfg_path = tmp_path / "config.yaml"
        self._write_rep_cfg(cfg_path, data, reports)

        # Produce a genuine, renderable result.json via a rep run.
        cli.main(["run", str(cfg_path), "rep"])
        produced = list(reports.glob("*/result.json"))
        assert produced, "rep run should produce a result.json"
        genuine = produced[0]

        # Add a second, strictly-newer run dir with a copy of that JSON.
        newer_dir = reports / "r_newer"
        newer_dir.mkdir()
        newer_json = newer_dir / "result.json"
        newer_json.write_text(genuine.read_text(encoding="utf-8"), encoding="utf-8")
        future = time.time() + 100
        os.utime(newer_json, (future, future))

        cfg = load_cfg(str(cfg_path))
        assert _find_latest_json(cfg) == str(newer_json)

        # The report subcommand runs without raising on the selected JSON.
        cli.main(["run", str(cfg_path), "report"])

    def test_run_returns_nonzero_when_step_fails(self, tmp_path):
        """`mrv run <cfg>` (all steps) must return a non-zero exit code when a
        step raises, so a broken run cannot show a false-green in CI."""
        from mrv import cli

        reports = tmp_path / "reports"
        cfg = tmp_path / "config.yaml"
        # Only one factor set -> _run_rep_convenience raises (needs >= 2), so the
        # rep step fails and the report step then finds no result.json.
        cfg.write_text(
            "validator:\n"
            f"  report_dir: {reports.as_posix()}\n"
            '  report_name: "r_{date}"\n'
            "  rep:\n"
            "    assets: {}\n"
            "    factors:\n"
            "      - [vol, drawdown]\n"
            "logging:\n"
            "  level: ERROR\n",
            encoding="utf-8",
        )
        rc = cli.main(["run", str(cfg)])
        assert rc not in (0, None), f"expected non-zero exit, got {rc!r}"

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
