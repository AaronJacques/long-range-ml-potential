"""End-to-end runs of the CLI chain ``prepare → train → evaluate → predict``.

Every chain runs in its own temporary working directory on the synthetic fixture:
``smoke.toml`` (thesis mode, 2 epochs) and ``golden_legacy.toml`` (legacy profile
M-1, 1 epoch). The tests check that training learns, that ``evaluate`` agrees with
:meth:`lrmlp.Potential.predict` on the raw test frames, that the run layout of
:mod:`lrmlp.run` is complete and that two runs with the same seed log identical
values.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from lrmlp import cli
from lrmlp.dataset import load_dataset
from lrmlp.potential import Potential
from lrmlp.run import RunDirectory

pytestmark = pytest.mark.slow

#: Frames of the ``predict`` input file.
PREDICT_FRAMES = 7


@dataclass(frozen=True)
class Chain:
    """Artefacts of one CLI chain."""

    config: Path
    workdir: Path
    dataset: Path
    run: Path
    eval_json: Path
    prediction: Path
    raw: Path


def _run_chain(config: Path, workdir: Path, raw: Path, run_name: str = "run") -> Chain:
    with np.load(raw) as data:
        np.savez(workdir / "in.npz", R=data["R"][:PREDICT_FRAMES], z=data["z"])
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(workdir)
        if not (workdir / "ds").exists():
            argv = ["prepare", "-c", str(config), "-o", "ds", "--workers", "1"]
            assert cli.main(argv) == 0
        assert cli.main(["train", "-c", str(config), "-d", "ds", "--run-dir", run_name]) == 0
        argv = ["evaluate", "--run", run_name, "--json", f"{run_name}_eval.json"]
        assert cli.main(argv) == 0
        argv = ["predict", "-r", run_name, "--input", "in.npz", "-o", f"{run_name}_pred.npz"]
        assert cli.main(argv) == 0
    return Chain(
        config=config,
        workdir=workdir,
        dataset=workdir / "ds",
        run=workdir / run_name,
        eval_json=workdir / f"{run_name}_eval.json",
        prediction=workdir / f"{run_name}_pred.npz",
        raw=raw,
    )


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory, fixtures_dir: Path) -> Iterator[Chain]:
    """``smoke.toml``: thesis mode, 2 epochs."""
    workdir = tmp_path_factory.mktemp("e2e_smoke")
    config = fixtures_dir / "configs" / "smoke.toml"
    yield _run_chain(config, workdir, fixtures_dir / "synthetic_small.npz")


@pytest.fixture(scope="module")
def legacy(tmp_path_factory: pytest.TempPathFactory, fixtures_dir: Path) -> Iterator[Chain]:
    """``golden_legacy.toml``: legacy assignment and model M-1, 1 epoch."""
    workdir = tmp_path_factory.mktemp("e2e_legacy")
    config = fixtures_dir / "configs" / "golden_legacy.toml"
    yield _run_chain(config, workdir, fixtures_dir / "synthetic_small.npz")


def _log(run: Path) -> list[dict[str, float]]:
    with (run / "training_log.csv").open(newline="") as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _check_layout(chain: Chain, epochs: int) -> None:
    rd = RunDirectory(chain.run)
    for path in (rd.config_path, rd.dataset_path, rd.log_path, rd.best_path):
        assert path.is_file(), path
    assert any(rd.tb_dir.iterdir()), "no TensorBoard event file"
    best = rd.read_best()
    assert 1 <= best["epoch"] <= epochs
    assert np.isfinite(best["val_energy_mse"])
    checkpoints = rd.list_checkpoints()
    assert best["epoch"] in checkpoints and epochs in checkpoints  # best_and_last
    assert rd.checkpoint_path(best["epoch"]).is_file()
    assert rd.evaluation_path("test", best["epoch"]).is_file()
    assert len(_log(chain.run)) == epochs
    assert rd.read_config().name == json.loads(rd.config_path.read_text())["name"]
    assert rd.read_dataset_metadata()["n_max_local"] > 0


def _check_evaluate_matches_potential(chain: Chain) -> None:
    """The test MAE of ``evaluate`` equals the MAE of ``Potential.predict`` on the raw frames."""
    metrics = json.loads(chain.eval_json.read_text())
    test = load_dataset(chain.dataset)["test"]
    with np.load(chain.raw) as raw:
        frames = np.asarray(test.sample_index)
        positions, numbers, energies = raw["R"][frames], raw["z"], raw["E"][frames]
    np.testing.assert_allclose(np.asarray(test.energies), energies)
    potential = Potential.from_run(chain.run)
    assert potential.epoch == metrics["epoch"]
    predicted = potential.predict(positions, numbers).energies
    assert metrics["n_molecules"] == len(frames)
    np.testing.assert_allclose(metrics["energy_mae"], np.abs(predicted - energies).mean(), 1e-5)


def _check_prediction(chain: Chain) -> None:
    with np.load(chain.prediction) as out:
        assert sorted(out.files) == ["E"]
        predicted = out["E"]
    with np.load(chain.workdir / "in.npz") as data:
        expected = Potential.from_run(chain.run).predict(data["R"], data["z"]).energies
    assert predicted.shape == (PREDICT_FRAMES,)
    np.testing.assert_allclose(predicted, expected, rtol=1e-6)


# --------------------------------------------------------------------------- smoke.toml


def test_smoke_training_loss_decreases(smoke: Chain) -> None:
    log = _log(smoke.run)
    assert [row["epoch"] for row in log] == [1.0, 2.0]
    assert all(np.isfinite(v) for row in log for v in row.values())
    assert log[1]["train_loss"] < log[0]["train_loss"]
    assert log[1]["val_energy_mse"] < log[0]["val_energy_mse"]


def test_smoke_evaluate_matches_potential(smoke: Chain) -> None:
    _check_evaluate_matches_potential(smoke)


def test_smoke_run_layout(smoke: Chain) -> None:
    _check_layout(smoke, epochs=2)


def test_smoke_predict(smoke: Chain) -> None:
    _check_prediction(smoke)


def test_smoke_same_seed_same_log(smoke: Chain) -> None:
    """A second run on the same dataset with the same seed logs identical values."""
    second = _run_chain(smoke.config, smoke.workdir, smoke.raw, run_name="run_repeat")
    first_log, second_log = _log(smoke.run), _log(second.run)
    assert len(first_log) == len(second_log) == 2
    for a, b in zip(first_log, second_log, strict=True):
        a.pop("seconds"), b.pop("seconds")
        assert a == b
    assert json.loads(second.eval_json.read_text()) == json.loads(smoke.eval_json.read_text())


# --------------------------------------------------------------------------- golden_legacy.toml


def test_legacy_profile_chain(legacy: Chain) -> None:
    config = json.loads((legacy.run / "config.json").read_text())
    assert config["grid"]["assignment"] == "legacy"
    assert config["train"]["epochs"] == 1
    log = _log(legacy.run)
    assert all(np.isfinite(v) for row in log for v in row.values())
    _check_layout(legacy, epochs=1)
    _check_evaluate_matches_potential(legacy)
    _check_prediction(legacy)
