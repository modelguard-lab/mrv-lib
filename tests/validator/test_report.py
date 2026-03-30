"""Tests for mrv.validator.report."""


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
