"""Tests for the ``lrmlp`` command line interface (T11)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from lrmlp import __version__, cli

COMMANDS = (
    "prepare",
    "train",
    "evaluate",
    "predict",
    "tune",
    "plot-log",
    "plot-weights",
    "info",
)
LAZY_COMMANDS = {
    "tune": "lrmlp.tuning",
    "plot-log": "lrmlp.analysis.plots",
    "plot-weights": "lrmlp.analysis.weights",
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # parent package missing
        return False


# --------------------------------------------------------------------------- help, version


def test_help_lists_all_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: lrmlp")
    for command in COMMANDS:
        assert command in out
    assert "--log-level" in out
    assert "--version" in out


def test_help_lists_lazy_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["-h"]) == 0
    out = capsys.readouterr().out
    for command in LAZY_COMMANDS:
        assert f"    {command}" in out


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command, "--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: lrmlp {command}")
    assert "current working directory" in out


def test_help_mentions_thesis_and_path_rules(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["prepare", "--help"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "preprocessing is run once, cf. thesis Sec. 4.3.2" in out
    assert "relative to the config file" in out
    assert cli.main(["evaluate", "--help"]) == 0
    assert "Tab. 5.3" in " ".join(capsys.readouterr().out.split())


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"lrmlp {__version__}"


def test_python_m_lrmlp_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lrmlp", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"lrmlp {__version__}"


@pytest.mark.parametrize(
    "argv", [["--help"], ["--version"], ["train", "--help"], ["predict", "--help"]]
)
def test_help_does_not_import_tensorflow(argv: list[str]) -> None:
    # A fresh interpreter, because other tests may already have imported TensorFlow.
    code = (
        "import sys\n"
        "from lrmlp.cli import main\n"
        f"code = main({argv!r})\n"
        "assert code == 0, code\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('tensorflow', 'keras'))\n"
        "sys.exit('loaded: ' + ', '.join(loaded) if loaded else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- usage errors


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["bogus"],
        ["train"],
        ["prepare", "--workers", "-1", "--config", "c.toml"],
        ["train", "--config", "c.toml", "--set", "no-equals-sign"],
        ["evaluate"],
        ["evaluate", "--run", "r", "--split", "holdout"],
        ["evaluate", "--run", "r", "--epoch", "first"],
        ["predict", "--run", "r", "--input", "in.npz"],
        ["predict", "--run", "r", "--input", "i", "--output", "o", "--batch-size", "0"],
        ["tune", "--config", "c.toml"],
        ["plot-log"],
        ["plot-log", "--run", "r", "--log", "l.csv"],
        ["plot-weights", "--run-a", "a", "--run-b", "b"],
        ["info"],
        ["info", "--dataset", "d", "--run", "r"],
        ["--log-level", "LOUD", "info", "--run", "r"],
    ],
)
def test_usage_error_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert "usage: lrmlp" in err
    assert "error:" in err


def test_runtime_error_exit_1_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "no-such-run"
    assert cli.main(["info", "--run", str(missing)]) == 1
    err = capsys.readouterr().err
    assert "no-such-run" in err
    assert "ERROR" in err
    assert "Traceback" not in err


def test_runtime_error_traceback_with_debug(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--log-level", "debug", "info", "--run", str(tmp_path / "x")]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_missing_config_is_runtime_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["prepare", "--config", str(tmp_path / "missing.toml")])
    assert code == 1
    assert "missing.toml" in capsys.readouterr().err


@pytest.mark.parametrize("command", sorted(LAZY_COMMANDS))
def test_lazy_command_without_module_exit_2(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = LAZY_COMMANDS[command]
    if _module_available(module):
        pytest.skip(f"{module} is available; the working path is tested by T13")
    argv = {
        "tune": ["tune", "--config", "c.toml", "--space", "s.toml", "--dry-run"],
        "plot-log": ["plot-log", "--log", "log.csv"],
        "plot-weights": ["plot-weights", "--run-a", "a", "--run-b", "b", "--output", "w.pdf"],
    }[command]
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert f"requires {module} (not installed / not available)" in err


def test_lazy_reports_missing_function(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    monkeypatch.setitem(sys.modules, "lrmlp._cli_test_dummy", types.ModuleType("dummy"))
    with pytest.raises(cli.UsageError, match="requires lrmlp._cli_test_dummy"):
        cli._lazy("lrmlp._cli_test_dummy", "run_search")


def test_lazy_does_not_hide_other_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str) -> None:
        raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")

    monkeypatch.setattr(cli.importlib, "import_module", fail)
    with pytest.raises(ModuleNotFoundError, match="matplotlib"):
        cli._lazy("lrmlp.analysis.plots", "plot_training_log")


def test_lazy_command_calls_function_with_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import types

    calls: dict[str, tuple[object, ...]] = {}
    module = types.ModuleType("fake")
    module.run_search = lambda *a: calls.setdefault("tune", a)  # type: ignore[attr-defined]
    module.plot_training_log = lambda *a: calls.setdefault("log", a)  # type: ignore[attr-defined]
    module.plot_weight_distribution = lambda *a: calls.setdefault("w", a)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "_lazy", lambda mod, fn: getattr(module, fn))
    monkeypatch.chdir(tmp_path)

    assert cli.main(["tune", "--config", "c.toml", "--space", "s.toml", "--max-trials", "3"]) == 0
    assert calls["tune"] == (tmp_path / "c.toml", tmp_path / "s.toml", 3, False)
    assert cli.main(["plot-log", "--run", "r", "--skip", "5"]) == 0
    assert calls["log"] == (tmp_path / "r", None, 5)
    argv = ["plot-weights", "--run-a", "a", "--run-b", "b", "-o", "w.pdf", "--epoch-b", "3"]
    assert cli.main(argv) == 0
    assert calls["w"] == (tmp_path / "a", tmp_path / "b", tmp_path / "w.pdf", "best", 3)


# --------------------------------------------------------------------------- end to end


@dataclass(frozen=True)
class Pipeline:
    """Artefacts of one prepare → train → evaluate → predict run."""

    workdir: Path
    dataset: Path
    run: Path
    eval_json: Path
    prediction: Path
    input_frames: int


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory, fixtures_dir: Path) -> Iterator[Pipeline]:
    """Run the CLI chain on ``smoke.toml`` from a foreign working directory.

    The config is given as an absolute path, all other CLI paths relative to the
    working directory (B10).
    """
    workdir = tmp_path_factory.mktemp("cli")
    config = str(fixtures_dir / "configs" / "smoke.toml")
    with np.load(fixtures_dir / "synthetic_small.npz") as raw:
        np.savez(workdir / "in.npz", R=raw["R"][:5], z=raw["z"])
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(workdir)
        assert cli.main(["prepare", "--config", config, "--output", "ds", "--workers", "1"]) == 0
        argv = ["train", "-c", config, "-d", "ds", "--run-dir", "run", "--set", "train.epochs=1"]
        assert cli.main(argv) == 0
        assert cli.main(["evaluate", "--run", "run", "--json", "eval.json"]) == 0
        argv = ["predict", "--run", "run", "--input", "in.npz", "--output", "out/pred.npz"]
        assert cli.main(argv) == 0
    yield Pipeline(
        workdir=workdir,
        dataset=workdir / "ds",
        run=workdir / "run",
        eval_json=workdir / "eval.json",
        prediction=workdir / "out" / "pred.npz",
        input_frames=5,
    )


def test_runs_from_other_cwd(pipeline: Pipeline, repo_root: Path) -> None:
    # B10: every artefact lands relative to the working directory, nothing in the repo.
    assert (pipeline.dataset / "metadata.json").is_file()
    assert (pipeline.run / "config.json").is_file()
    assert (pipeline.run / "best.json").is_file()
    assert pipeline.eval_json.is_file()
    assert pipeline.prediction.is_file()
    assert repo_root not in pipeline.workdir.parents


def test_set_override_reaches_run_config(pipeline: Pipeline) -> None:
    config = json.loads((pipeline.run / "config.json").read_text())
    assert config["train"]["epochs"] == 1
    assert config["name"] == "smoke"
    log_lines = (pipeline.run / "training_log.csv").read_text().strip().splitlines()
    assert len(log_lines) == 2  # header + one epoch


def test_evaluate_json(pipeline: Pipeline) -> None:
    metrics = json.loads(pipeline.eval_json.read_text())
    assert metrics["split"] == "test"
    assert metrics["epoch"] == 1
    assert metrics["n_molecules"] == 6
    assert np.isfinite(metrics["energy_mae"]) and metrics["energy_mae"] >= 0
    assert metrics["force_mae"] is None
    assert (pipeline.run / "evaluation_test_1.json").is_file()


def test_evaluate_table_and_json_stdout(
    pipeline: Pipeline, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["evaluate", "--run", str(pipeline.run), "--split", "val"]) == 0
    out = capsys.readouterr().out
    assert "energy MAE [kcal/mol]" in out
    assert "val" in out
    argv = ["evaluate", "--run", str(pipeline.run), "--epoch", "last", "--json", "-"]
    assert cli.main(argv) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["split"] == "test"
    assert metrics["epoch"] == 1


def test_predict_output(pipeline: Pipeline) -> None:
    with np.load(pipeline.prediction) as out:
        assert sorted(out.files) == ["E"]
        energies = out["E"]
    assert energies.shape == (pipeline.input_frames,)
    assert energies.dtype == np.float64
    assert np.all(np.isfinite(energies))


def test_predict_matches_potential(pipeline: Pipeline) -> None:
    from lrmlp.potential import Potential

    with np.load(pipeline.workdir / "in.npz") as raw:
        expected = Potential.from_run(pipeline.run).predict(raw["R"], raw["z"]).energies
    with np.load(pipeline.prediction) as out:
        np.testing.assert_allclose(out["E"], expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"R": np.zeros((2, 3, 3))}, "missing array(s) ['z']"),
        ({"z": np.ones(3, dtype=int)}, "missing array(s) ['R']"),
        ({"R": np.zeros((2, 3)), "z": np.ones(3, dtype=int)}, "R must have shape (S, N, 3)"),
        ({"R": np.zeros((2, 3, 3)), "z": np.ones(4, dtype=int)}, "z must have shape (N,)"),
    ],
)
def test_predict_validates_input(
    pipeline: Pipeline,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arrays: dict[str, np.ndarray],
    message: str,
) -> None:
    np.savez(tmp_path / "bad.npz", **arrays)
    argv = ["predict", "--run", str(pipeline.run), "-i", str(tmp_path / "bad.npz")]
    assert cli.main([*argv, "-o", str(tmp_path / "out.npz")]) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.npz").exists()


def test_info_run(pipeline: Pipeline, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["info", "--run", str(pipeline.run)]) == 0
    out = capsys.readouterr().out
    assert "best epoch" in out
    assert "checkpoints" in out
    assert "smoke" in out
    assert cli.main(["info", "--run", str(pipeline.run), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["name"] == "smoke"
    assert info["best"]["epoch"] == 1
    assert info["checkpoints"] == [1]
    assert info["train"]["epochs"] == 1
    assert info["dataset"] == str(pipeline.dataset.resolve())
    assert "evaluation_test_1.json" in info["evaluations"]


def test_info_dataset(
    pipeline: Pipeline, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["info", "--dataset", str(pipeline.dataset)]) == 0
    out = capsys.readouterr().out
    assert "n_max local / long range" in out
    assert "split train" in out
    target = tmp_path / "info.json"
    assert cli.main(["info", "--dataset", str(pipeline.dataset), "--json", str(target)]) == 0
    info = json.loads(target.read_text())
    assert {name: s["n_molecules"] for name, s in info["splits"].items()} == {
        "train": 48,
        "val": 6,
        "test": 6,
    }
    assert info["species"] == [1, 6, 8]
    assert info["n_max_local"] > 0


def test_train_default_run_dir_below_run_root(
    pipeline: Pipeline, fixtures_dir: Path, tmp_path: Path
) -> None:
    # Without --run-dir a new <name>-<timestamp> directory is created below
    # train.run_root (here set to an absolute path via --set).
    config = str(fixtures_dir / "configs" / "smoke.toml")
    root = tmp_path / "runs"
    argv = ["train", "-c", config, "-d", str(pipeline.dataset)]
    argv += ["--set", "train.epochs=1", "--set", f'train.run_root="{root}"']
    assert cli.main(argv) == 0
    runs = list(root.iterdir())
    assert len(runs) == 1
    assert runs[0].name.startswith("smoke-")
    assert (runs[0] / "best.json").is_file()


def test_prepare_refuses_existing_dataset(
    pipeline: Pipeline, fixtures_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = str(fixtures_dir / "configs" / "smoke.toml")
    argv = ["prepare", "-c", config, "--output", str(pipeline.dataset), "--workers", "1"]
    assert cli.main(argv) == 1
    assert "FileExistsError" in capsys.readouterr().err


@pytest.mark.slow
def test_subprocess_chain_with_spawn_workers(fixtures_dir: Path, tmp_path: Path) -> None:
    # The real entry point in fresh interpreters, from a foreign working directory,
    # with two spawn workers in `prepare` (needs the __main__ guard of __main__.py).
    config = str(fixtures_dir / "configs" / "smoke.toml")
    with np.load(fixtures_dir / "synthetic_small.npz") as raw:
        np.savez(tmp_path / "in.npz", R=raw["R"][:3], z=raw["z"])

    def lrmlp(*argv: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, "-m", "lrmlp", *argv],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result

    prepared = lrmlp("prepare", "-c", config, "-o", "ds", "--workers", "2")
    assert Path(prepared.stdout.strip()) == (tmp_path / "ds").resolve()
    assert " INFO " in prepared.stderr  # log records carry time and level
    trained = lrmlp(
        "train", "-c", config, "-d", "ds", "--run-dir", "run", "--set", "train.epochs=1"
    )
    assert Path(trained.stdout.strip()) == (tmp_path / "run").resolve()
    metrics = json.loads(lrmlp("evaluate", "--run", "run", "--json", "-").stdout)
    assert metrics["n_molecules"] == 6
    lrmlp("predict", "--run", "run", "-i", "in.npz", "-o", "out.npz")
    with np.load(tmp_path / "out.npz") as out:
        assert out["E"].shape == (3,)
    info = json.loads(lrmlp("info", "--dataset", "ds", "--json").stdout)
    assert info["splits"]["train"]["n_molecules"] == 48
