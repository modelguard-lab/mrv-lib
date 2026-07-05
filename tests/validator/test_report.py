"""Tests for mrv.validator.report."""

import json


def _rep_result_json():
    return {
        "test": "representation_invariance",
        "generated": "2026-07-06T00:00:00",
        "model": "GMM",
        "n_states": 3,
        "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "ari_threshold": 0.65,
        "spearman_threshold": 0.85,
        "overall_mean_ari": 0.72,
        "overall_mean_spearman": 0.9,
        "partition_pass": True,
        "ordering_pass": True,
        "assets": {
            "SPY": {
                "n_specs": 2,
                "n_factor_sets": 2,
                "n_obs": 200,
                "mean_ari": 0.72,
                "min_ari": 0.6,
                "mean_spearman": 0.9,
                "partition_pass": True,
                "ordering_pass": True,
                "ari_matrix": {
                    "labels": ["set0", "set1"],
                    "values": [[1.0, 0.72], [0.72, 1.0]],
                },
                "heatmap_png": "SPY.png",
            }
        },
    }


def _res_result_json():
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


class TestReportRender:
    def test_generate_report_rep(self, tmp_path):
        from mrv.validator.report import generate_report

        run_dir = tmp_path / "rep_run"
        run_dir.mkdir()
        jp = run_dir / "result.json"
        jp.write_text(json.dumps(_rep_result_json()), encoding="utf-8")

        generate_report(jp)  # returns None when pdflatex is absent; .tex always written

        tex = run_dir / f"{run_dir.name}.tex"
        assert tex.exists()
        content = tex.read_text(encoding="utf-8")
        assert "SPY" in content
        assert "0.720" in content          # overall/mean ARI row
        assert "0.65" in content           # ARI threshold
        assert "0.85" in content           # Spearman threshold

    def test_generate_report_res(self, tmp_path):
        from mrv.validator.report import generate_report

        run_dir = tmp_path / "res_run"
        run_dir.mkdir()
        jp = run_dir / "result.json"
        jp.write_text(json.dumps(_res_result_json()), encoding="utf-8")

        # Must not raise KeyError on resolution JSON.
        generate_report(jp)

        tex = run_dir / f"{run_dir.name}.tex"
        assert tex.exists()
        content = tex.read_text(encoding="utf-8")
        assert "SPY" in content
        assert "0.300" in content                        # overall mean ARI
        assert "resolution stability dashboard" in content  # res-branch caption
        # No ordering layer for resolution: Spearman column renders as N/A.
        assert "N/A" in content

    def test_report_pipeline_wrapper_res(self, tmp_path):
        """The pipeline-level report() also handles a resolution result JSON."""
        from mrv.pipeline import report

        run_dir = tmp_path / "res_run2"
        run_dir.mkdir()
        jp = run_dir / "result.json"
        jp.write_text(json.dumps(_res_result_json()), encoding="utf-8")
        report(jp)
        assert (run_dir / f"{run_dir.name}.tex").exists()


class TestReportHelpers:
    def test_tex_escaping(self):
        from mrv.validator.report import _tex
        assert _tex("a & b") == "a \\& b"
        assert _tex("100%") == "100\\%"
        assert _tex("$x$") == "\\$x\\$"
        assert _tex("a_b") == "a\\_b"

    def test_ari_table(self):
        from mrv.validator.report import _ari_table
        result = _ari_table(["Set 0", "Set 1"], [[1.0, 0.5], [0.5, 1.0]], threshold=0.65)
        assert "\\begin{tabular}" in result
        assert "cellcolor" in result  # 0.5 < 0.65

    def test_ari_table_all_pass(self):
        from mrv.validator.report import _ari_table
        result = _ari_table(["Set 0", "Set 1"], [[1.0, 0.8], [0.8, 1.0]], threshold=0.65)
        assert "cellcolor" not in result

    def test_eval_conditionals(self):
        from mrv.validator.report import _eval_conditionals
        text = "before\n%% IF_PASS\nyes\n%% ELSE\nno\n%% ENDIF\nafter"
        assert "yes" in _eval_conditionals(text, {"PASS": True})
        assert "no" in _eval_conditionals(text, {"PASS": False})

    def test_eval_conditionals_elif(self):
        from mrv.validator.report import _eval_conditionals
        text = "start\n%% IF_A\nA\n%% ELIF_B\nB\n%% ELSE\nC\n%% ENDIF\nend"
        assert "A" in _eval_conditionals(text, {"A": True, "B": False})
        result_b = _eval_conditionals(text, {"A": False, "B": True})
        assert "B" in result_b and "A" not in result_b
