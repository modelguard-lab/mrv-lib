"""Test version consistency."""


def test_version():
    import mrv
    assert mrv.__version__ == "0.2.0"
