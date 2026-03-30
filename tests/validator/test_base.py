"""Tests for mrv.validator.base."""

from mrv.validator.base import BaseValidator


class TestBaseValidator:
    def test_make_run_dir(self, tmp_path):
        class Dummy(BaseValidator):
            name = "test"
            def validate(self, prices=None, labels=None):
                return {}

        v = Dummy({"validator": {"report_dir": str(tmp_path), "report_name": "test_{date}"}})
        run_dir = v._make_run_dir()
        assert run_dir.exists()
        assert "test_" in run_dir.name
        assert "_test" in run_dir.name
