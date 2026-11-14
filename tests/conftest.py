"""Shared pytest configuration for the ``lrmlp`` test suite."""

# The environment must be fixed before anything can import TensorFlow:
# TF_USE_LEGACY_KERAS=1 would switch ``tf.keras`` to ``tf_keras``, but the
# tests always run against Keras 3. The variable is removed unconditionally,
# also for ``legacy_env`` tests: they must not load the legacy code in-process
# (``golden.legacy.load_legacy`` would raise) but start it in a subprocess with
# ``env={**os.environ, "TF_USE_LEGACY_KERAS": "1"}`` (e.g. the generators' --check).
import os

os.environ.pop("TF_USE_LEGACY_KERAS", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``legacy_env`` tests unless ``LRMLP_LEGACY_ENV=1`` is set."""
    if os.environ.get("LRMLP_LEGACY_ENV") == "1":
        return
    skip_legacy = pytest.mark.skip(reason="needs the legacy environment (set LRMLP_LEGACY_ENV=1)")
    for item in items:
        if "legacy_env" in item.keywords:
            item.add_marker(skip_legacy)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Root directory of the repository."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    """Directory with the committed test fixtures (``tests/fixtures``)."""
    return repo_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def golden_dir(fixtures_dir: Path) -> Path:
    """Directory with the golden reference outputs (``tests/fixtures/golden``)."""
    return fixtures_dir / "golden"
