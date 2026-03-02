"""Smoke tests to verify that tmswarp installs and imports correctly."""

import importlib


def test_import_tmswarp():
    """Verify the top-level package is importable."""
    import tmswarp

    assert tmswarp is not None


def test_version_string():
    """Verify __version__ is set and looks like a version string."""
    from tmswarp import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2, f"Expected semver-like version, got {__version__}"


def test_warp_importable():
    """Verify that the warp dependency is importable when installed."""
    pytest = importlib.import_module("pytest")
    warp = pytest.importorskip("warp")
    assert warp is not None


def test_numpy_fem_importable():
    """Verify that numpy FEM modules are importable."""
    from tmswarp import analytical, coil, conductor, solver, fields  # noqa: F401
