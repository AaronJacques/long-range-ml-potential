"""Tests for the package scaffold: version, lazy exports and the CLI stub."""

import subprocess
import sys

import pytest

import lrmlp
from lrmlp import cli

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


def test_version() -> None:
    assert lrmlp.__version__ == "0.1.0"


def test_import_does_not_load_tensorflow() -> None:
    # A fresh interpreter, because other tests may already have imported TensorFlow.
    code = "import sys, lrmlp; sys.exit('tensorflow' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_all_contains_exactly_public_api() -> None:
    assert len(lrmlp.__all__) == len(EXPECTED_ALL)
    assert set(lrmlp.__all__) == EXPECTED_ALL


def test_dir_lists_public_api() -> None:
    assert set(dir(lrmlp)) >= EXPECTED_ALL


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
        _ = lrmlp.does_not_exist


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "lrmlp 0.1.0"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_cli_help(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([flag]) == 0
    assert "usage: lrmlp" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[], ["train"], ["--version", "extra"]])
def test_cli_other_arguments_not_implemented(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_python_m_lrmlp_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lrmlp", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "lrmlp 0.1.0"


def test_conftest_forces_keras3_environment() -> None:
    import os

    assert "TF_USE_LEGACY_KERAS" not in os.environ
    assert os.environ["KERAS_BACKEND"] == "tensorflow"
