"""Tests pinning the top-level public API surface of ``mrv``.

The 0.7.0 headline change rebinds ``mrv.report`` to the report implementation
module (``mrv.validator.report.generate_report``) rather than the internal CLI
backend (``mrv.pipeline.report``). Nothing previously exercised the top-level
symbol; these tests lock both the identity of the rebind and its behavior.
"""

import json

import mrv
import mrv.validator.report


def test_report_is_generate_report():
    # The public symbol must be exactly the implementation function.
    assert mrv.report is mrv.validator.report.generate_report
    assert "report" in mrv.__all__


def _res_result_json():
    """Minimal resolution result JSON (mirrors tests/validator/test_report.py)."""
    return {
        "test": "resolution_invariance",
        "generated": "2026-07-06T00:00:00",
        "date_range": {"start": "2026-01-01", "end": "2026-01-07"},
        "ari_threshold": 0.65,
        "overall_mean_ari": 0.30,
        "partition_pass": False,
        "assets": {
            "SPY": {
                "frequencies": ["5m", "15m", "1h", "1d"],
                "overall_mean_ari": 0.30,
                "partition_pass": False,
                "pvalue_perm": 0.4,
                "null_ci": [0.1, 0.5],
                "event_mean_ari": None,
                "calm_mean_ari": None,
                "crisis_shares": {},
                "rolling_ari_median": None,
                "ari_matrix": {
                    "labels": ["5m", "15m", "1h", "1d"],
                    "values": [
                        [1.0, 0.30, 0.20, 0.10],
                        [0.30, 1.0, 0.25, 0.15],
                        [0.20, 0.25, 1.0, 0.20],
                        [0.10, 0.15, 0.20, 1.0],
                    ],
                },
            }
        },
    }


def test_top_level_report_writes_tex(tmp_path):
    # Drive a resolution result JSON through the TOP-LEVEL mrv.report symbol.
    run_dir = tmp_path / "res_run"
    run_dir.mkdir()
    jp = run_dir / "result.json"
    jp.write_text(json.dumps(_res_result_json()), encoding="utf-8")

    # Returns None when pdflatex is absent; the .tex file is always written.
    mrv.report(jp)

    tex = run_dir / f"{run_dir.name}.tex"
    assert tex.exists()
    content = tex.read_text(encoding="utf-8")
    assert "SPY" in content
    assert "0.300" in content
