"""Tests for mrv.validator.monitor."""

import json

import numpy as np
import pandas as pd
import pytest


class TestMonitoring:
    def test_load_history_empty(self, tmp_path):
        from mrv.validator.monitor import _load_history
        assert _load_history(tmp_path / "nonexistent.csv").empty

    def test_extract_metrics(self):
        from mrv.validator.monitor import _extract_metrics
        rows = _extract_metrics(
            {"assets": {"SPY": {"mean_ari": 0.72}, "CL": {"mean_ari": 0.25}}}, "rep", "2026-03-23")
        assert len(rows) == 2
        assert rows[0]["mean_ari"] == 0.72

    def test_check_alerts_below_threshold(self):
        from mrv.validator.monitor import _check_alerts
        new_rows = pd.DataFrame([{
            "date": "2026-03-23", "asset": "CL", "validator": "rep",
            "mean_ari": 0.2, "mean_ari_7d_avg": None, "delta_vs_baseline": None, "alert_fired": False,
        }])
        alerts = _check_alerts(new_rows, pd.DataFrame(columns=["date", "asset", "validator", "mean_ari"]),
                                {"alert_ari_below": 0.3})
        assert len(alerts) == 1
        assert "ARI=0.200 < 0.3" in alerts[0]["reasons"][0]

    def test_check_alerts_no_alert(self):
        from mrv.validator.monitor import _check_alerts
        new_rows = pd.DataFrame([{
            "date": "2026-03-23", "asset": "SPY", "validator": "rep",
            "mean_ari": 0.8, "mean_ari_7d_avg": None, "delta_vs_baseline": None, "alert_fired": False,
        }])
        assert len(_check_alerts(new_rows, pd.DataFrame(columns=["date", "asset", "validator", "mean_ari"]),
                                  {"alert_ari_below": 0.3})) == 0

    def test_idempotency(self):
        from mrv.validator.monitor import _is_already_run
        history = pd.DataFrame([{"date": "2026-03-23", "asset": "SPY", "validator": "rep", "mean_ari": 0.5}])
        assert _is_already_run(history, "2026-03-23", "rep")
        assert not _is_already_run(history, "2026-03-24", "rep")
        assert not _is_already_run(history, "2026-03-23", "res")

    def test_fire_alerts_file(self, tmp_path):
        from mrv.validator.monitor import _fire_alerts
        alerts_path = tmp_path / "alerts.json"
        _fire_alerts([{"date": "2026-03-23", "asset": "CL", "reasons": ["test"]}], alerts_path, {})
        assert alerts_path.exists()
        assert json.loads(alerts_path.read_text().strip())["asset"] == "CL"
