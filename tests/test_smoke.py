from meta_model import __version__


def test_version_exposed() -> None:
    assert __version__ == "0.1.0"
