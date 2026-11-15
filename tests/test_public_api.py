"""Contract of the public API ``lrmlp.__all__`` and of the CI workflow.

The public names are exported lazily: ``import lrmlp`` must not load
TensorFlow, every name in ``__all__`` resolves and is documented, unknown names
raise :class:`AttributeError`. The optional subpackages ``lrmlp.tuning`` and
``lrmlp.analysis`` are not part of ``__all__``.

The CI workflow cannot run on this machine, so its commands are checked as text
(PyYAML is not a dependency): the jobs lint, typecheck, test and slow, the coverage
threshold of the test job and the slow job without the "no tests collected" guard.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import lrmlp

EXPECTED_ALL = {
    "__version__",
    "ExperimentConfig",
    "GridConfig",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
    "compute_descriptors",
    "build_dataset",
    "load_dataset",
    "build_model",
    "train",
    "evaluate",
    "Potential",
}
COVERAGE_THRESHOLD = 85


# --------------------------------------------------------------------------- public API


def test_all_is_the_planned_api() -> None:
    assert set(lrmlp.__all__) == EXPECTED_ALL
    assert len(lrmlp.__all__) == len(EXPECTED_ALL)


@pytest.mark.parametrize("name", sorted(EXPECTED_ALL - {"__version__"}))
def test_public_name_resolves_and_is_documented(name: str) -> None:
    obj = getattr(lrmlp, name)
    assert inspect.isclass(obj) or inspect.isfunction(obj), name
    assert obj.__module__.startswith("lrmlp."), obj.__module__
    assert inspect.getdoc(obj), f"{name} has no docstring"
    assert obj is getattr(sys.modules[obj.__module__], name)
    assert name in dir(lrmlp)


def test_version_matches_distribution() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", lrmlp.__version__)
    assert lrmlp.__version__ == importlib.metadata.version("long-range-ml-potential")


@pytest.mark.parametrize("name", ["does_not_exist", "tuning_grid", "Prediction", "_LAZY"])
def test_unknown_name_raises_attribute_error(name: str) -> None:
    with pytest.raises(AttributeError, match=name):
        getattr(lrmlp, name)


@pytest.mark.parametrize("name", ["tuning", "analysis"])
def test_optional_subpackages_not_public(name: str) -> None:
    assert name not in lrmlp.__all__


def _run_python(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr


def test_import_does_not_load_tensorflow() -> None:
    _run_python(
        "import sys, lrmlp\n"
        "bad = sorted(m for m in ('tensorflow', 'keras') if m in sys.modules)\n"
        "assert not bad, bad\n"
    )


def test_numpy_only_names_do_not_load_tensorflow() -> None:
    """Configuration and descriptors are NumPy-only."""
    _run_python(
        "import sys, lrmlp\n"
        "lrmlp.load_config, lrmlp.ExperimentConfig, lrmlp.compute_descriptors\n"
        "bad = sorted(m for m in ('tensorflow', 'keras') if m in sys.modules)\n"
        "assert not bad, bad\n"
    )


# --------------------------------------------------------------------------- CI workflow


def _jobs(text: str) -> dict[str, str]:
    """Split the ``jobs:`` block of a workflow into ``{job name: job text}``."""
    jobs_block = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\s*$", jobs_block, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def _run_commands(job: str) -> list[str]:
    return [m.strip() for m in re.findall(r"^\s+- run: (.+)$", job, flags=re.MULTILINE)]


@pytest.fixture(scope="module")
def ci_jobs(repo_root: Path) -> dict[str, str]:
    text = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "\t" not in text, "YAML must not contain tabs"
    assert re.search(r"^on:\s*$", text, flags=re.MULTILINE)
    return _jobs(text)


def test_ci_has_planned_jobs(ci_jobs: dict[str, str]) -> None:
    assert set(ci_jobs) == {"lint", "typecheck", "test", "slow"}
    for name, job in ci_jobs.items():
        assert "actions/checkout@" in job, name
        assert 'python-version: "3.12"' in job or "python-version: ${{" in job, name
        assert "cache: pip" in job, name


def test_ci_lint_and_typecheck_commands(ci_jobs: dict[str, str]) -> None:
    lint = _run_commands(ci_jobs["lint"])
    assert "ruff check ." in lint
    assert "ruff format --check ." in lint
    assert "mypy" in _run_commands(ci_jobs["typecheck"])


def test_ci_test_job_enforces_coverage(ci_jobs: dict[str, str]) -> None:
    job = ci_jobs["test"]
    assert re.search(r"os: \[macos-14, ubuntu-latest\]", job)
    assert re.search(r'python-version: \["3\.12"\]', job)
    commands = _run_commands(job)
    assert 'python -m pip install -e ".[dev,plot,tune]"' in commands
    pytest_cmd = [c for c in commands if c.startswith("pytest")]
    assert len(pytest_cmd) == 1
    assert '-m "not slow and not legacy_env"' in pytest_cmd[0]
    assert "--cov=lrmlp" in pytest_cmd[0]
    assert f"--cov-fail-under={COVERAGE_THRESHOLD}" in pytest_cmd[0]


def test_ci_slow_job_runs_slow_tests_without_guard(ci_jobs: dict[str, str]) -> None:
    job = ci_jobs["slow"]
    assert "needs: test" in job
    assert "refs/heads/main" in job
    commands = _run_commands(job)
    assert 'pytest -m "slow and not legacy_env"' in commands
    assert not any("$?" in c or "||" in c for c in commands)


def test_pyproject_coverage_threshold(repo_root: Path) -> None:
    config = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["coverage"]["report"]["fail_under"] == COVERAGE_THRESHOLD
    assert config["tool"]["coverage"]["run"]["source"] == ["lrmlp"]
