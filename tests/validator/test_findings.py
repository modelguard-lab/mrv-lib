"""Tests for mrv.validator.findings and SR 11-7 report."""

import numpy as np
import pytest


class TestFindings:
    def test_classify_severity(self):
        from mrv.validator.findings import classify_severity
        assert classify_severity(0.05) == "Critical"
        assert classify_severity(0.3) == "High"
        assert classify_severity(0.7, min_pairwise_ari=0.4) == "Medium"
        assert classify_severity(0.8, any_fallback=True) == "Low"
        assert classify_severity(0.8) == "Informational"
        assert classify_severity(None) == "High"

    def test_generate_findings_rep(self):
        from mrv.validator.findings import generate_findings
        findings = generate_findings(
            {"SPY": {"mean_ari": 0.3, "min_ari": 0.1, "mean_spearman": 0.9},
             "CL": {"mean_ari": 0.8, "min_ari": 0.7, "mean_spearman": 0.95}}, "rep")
        spy = [f for f in findings if "SPY" in f.title]
        assert len(spy) >= 1
        assert spy[0].severity in ("Critical", "High")

    def test_generate_findings_res(self):
        from mrv.validator.findings import generate_findings
        findings = generate_findings(
            {"SPY": {"overall_mean_ari": 0.05, "fallback_flags": {"5m": False, "1d": True}}}, "res")
        assert "Critical" in [f.severity for f in findings]

    def test_overall_risk_rating(self):
        from mrv.validator.findings import Finding, overall_risk_rating
        assert overall_risk_rating([Finding(id="1", severity="Medium", title="t", description="d")]) == "Medium"
        assert overall_risk_rating([Finding(id="1", severity="Critical", title="t", description="d")]) == "High"
        assert overall_risk_rating([Finding(id="1", severity="Informational", title="t", description="d")]) == "Low"

    def test_findings_summary(self):
        from mrv.validator.findings import Finding, findings_summary
        s = findings_summary([
            Finding(id="1", severity="High", title="t", description="d"),
            Finding(id="2", severity="High", title="t", description="d"),
            Finding(id="3", severity="Low", title="t", description="d"),
        ])
        assert s["High"] == 2 and s["Low"] == 1 and s["Critical"] == 0

    def test_yaml_overrides(self, tmp_path):
        from mrv.validator.findings import generate_findings
        override_file = tmp_path / "overrides.yaml"
        override_file.write_text(
            "MRV-2026-001:\n  remediation_owner: 'J. Smith'\n  deadline: '2026-06-30'\n", encoding="utf-8")
        findings = generate_findings(
            {"SPY": {"mean_ari": 0.3, "min_ari": 0.1, "mean_spearman": 0.9}}, "rep", overrides_path=override_file)
        assert findings[0].remediation_owner == "J. Smith"

    def test_sr11_7_report_function_exists(self):
        from mrv.validator.report import generate_sr11_7_report
        assert callable(generate_sr11_7_report)
