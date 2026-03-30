"""Tests for mrv.validator.rep (Representation Invariance)."""


class TestRepValidator:
    def test_rep_imported_from_package(self):
        from mrv.validator import RepValidator
        assert RepValidator.name == "rep"
