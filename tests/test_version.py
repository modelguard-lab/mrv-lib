"""Test version consistency."""


def test_version():
    import mrv
    assert mrv.__version__ == "0.7.0"
