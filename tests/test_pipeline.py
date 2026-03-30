"""Tests for mrv.pipeline."""


class TestPipeline:
    def test_validators_registered(self):
        from mrv.pipeline import _VALIDATORS
        assert "rep" in _VALIDATORS
        assert "res" in _VALIDATORS

    def test_monitor_exposed(self):
        from mrv.pipeline import monitor
        assert callable(monitor)

    def test_sr11_7_report_exposed(self):
        from mrv.pipeline import sr11_7_report
        assert callable(sr11_7_report)
