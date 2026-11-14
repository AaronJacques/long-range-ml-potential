"""Tests of the inference API ``lrmlp.potential.Potential``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import keras
import numpy as np
import pytest

from lrmlp.config import ExperimentConfig, load_config
from lrmlp.dataset import build_dataset, load_dataset
from lrmlp.evaluation import evaluate
from lrmlp.grid import assign_cells
from lrmlp.model import build_model
from lrmlp.potential import Potential, Prediction
from lrmlp.run import RunDirectory

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG = FIXTURES / "configs" / "smoke.toml"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> tuple[ExperimentConfig, Path]:
    config = load_config(CONFIG)
    path = build_dataset(config, output_dir=tmp_path_factory.mktemp("ds") / "smoke", n_workers=1)
    return config, path


@pytest.fixture(scope="module")
def frames() -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "synthetic_small.npz") as data:
        return {k: data[k] for k in data.files}


def _make_run(
    root: Path, config: ExperimentConfig, dataset_dir: Path, seed: int = 0
) -> RunDirectory:
    """Run with one randomly initialised checkpoint (epoch 1, also best)."""
    ds = load_dataset(dataset_dir)
    run = RunDirectory(root).create(config, {**ds.metadata, "path": str(dataset_dir)})
    keras.utils.set_random_seed(seed)
    model = build_model(
        config.model, ds.n_max_local, ds.n_max_long_range, config.grid.max_atomic_number
    )
    # Non-zero output bias so that energies and forces are not all near zero.
    weights = model.get_weights()
    weights[-1] = np.linspace(-1.0, 1.0, weights[-1].shape[0]).astype(np.float32)
    model.set_weights(weights)
    model.save_weights(run.checkpoint_path(1))
    run.write_best(1, {"val_energy_mse": 1.0})
    return run


@pytest.fixture(scope="module")
def energy_run(
    tmp_path_factory: pytest.TempPathFactory, smoke: tuple[ExperimentConfig, Path]
) -> RunDirectory:
    config, ds = smoke
    return _make_run(tmp_path_factory.mktemp("energy") / "run", config, ds)


@pytest.fixture(scope="module")
def force_run(
    tmp_path_factory: pytest.TempPathFactory, smoke: tuple[ExperimentConfig, Path]
) -> RunDirectory:
    config, ds = smoke
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, predict_forces=True)
    )
    return _make_run(tmp_path_factory.mktemp("forces") / "run", config, ds, seed=1)


def test_from_run(energy_run: RunDirectory, smoke: tuple[ExperimentConfig, Path]) -> None:
    config, ds = smoke
    pot = Potential.from_run(energy_run.path)
    meta = load_dataset(ds).metadata
    assert pot.epoch == 1
    assert (pot.n_max_local, pot.n_max_long_range) == (
        meta["n_max_local"],
        meta["n_max_long_range"],
    )
    assert pot.config == energy_run.read_config()
    assert not pot.predicts_forces and pot.use_long_range


def test_predict_equals_single_calls(
    energy_run: RunDirectory, frames: dict[str, np.ndarray]
) -> None:
    pot = Potential.from_run(energy_run.path)
    pos, z = frames["R"][:12], frames["z"]
    pred = pot.predict(pos, z)
    assert isinstance(pred, Prediction)
    assert pred.energies.shape == (12,) and pred.forces is None
    single = np.array([pot.predict(pos[s], z).energies[0] for s in range(12)])
    np.testing.assert_allclose(pred.energies, single, rtol=1e-6)
    # Reference: model on one molecule, summed over atoms (no batching code involved).
    x = pot.descriptors(pos[3], z)
    ref = float(np.sum(np.asarray(pot.model(x, training=False))[:, 0], dtype=np.float32))
    np.testing.assert_allclose(pred.energies[3], ref, rtol=1e-6)
    assert np.ptp(pred.energies) > 0


def test_batch_size_invariance(energy_run: RunDirectory, frames: dict[str, np.ndarray]) -> None:
    pot = Potential.from_run(energy_run.path)
    pos, z = frames["R"][:20], frames["z"]
    a = pot.predict(pos, z, batch_size=1).energies
    b = pot.predict(pos, z, batch_size=256).energies
    c = pot.predict(pos, z, batch_size=7).energies
    np.testing.assert_allclose(a, b, rtol=1e-6)
    np.testing.assert_allclose(c, b, rtol=1e-6)
    with pytest.raises(ValueError, match="batch_size"):
        pot.predict(pos, z, batch_size=0)


def test_single_molecule(force_run: RunDirectory, frames: dict[str, np.ndarray]) -> None:
    pot = Potential.from_run(force_run.path)
    pos, z = frames["R"][5], frames["z"]
    one = pot.predict(pos, z)
    batch = pot.predict(frames["R"][5:6], z)
    assert one.energies.shape == (1,)
    assert one.forces is not None and batch.forces is not None
    assert one.forces.shape == (1, z.shape[0], 3)
    np.testing.assert_array_equal(one.energies, batch.energies)
    np.testing.assert_array_equal(one.forces, batch.forces)


def test_n_max_overflow_raises(energy_run: RunDirectory, frames: dict[str, np.ndarray]) -> None:
    pot = Potential.from_run(energy_run.path)
    dense = frames["R"][:2].copy()
    dense[1] *= 0.5  # compressed molecule: more neighbours per atom
    with pytest.raises(ValueError, match=rf"frame 1: .*n_max_local = {pot.n_max_local}"):
        pot.predict(dense, frames["z"])


def test_invalid_input_raises(energy_run: RunDirectory, frames: dict[str, np.ndarray]) -> None:
    pot = Potential.from_run(energy_run.path)
    pos, z = frames["R"][:2], frames["z"].copy()
    bad = z.copy()
    bad[0] = pot.config.grid.max_atomic_number + 1
    with pytest.raises(ValueError, match="max_atomic_number = 100"):
        pot.predict(pos, bad)
    bad[0] = 0
    with pytest.raises(ValueError, match="outside 1"):
        pot.predict(pos, bad)
    with pytest.raises(ValueError, match="atoms"):
        pot.predict(pos, z[:-1])
    with pytest.raises(ValueError, match="shape"):
        pot.predict(pos[..., :2], z)


def test_forces_follow_input_order(force_run: RunDirectory, frames: dict[str, np.ndarray]) -> None:
    """Permuting the atoms permutes the predicted forces the same way (B01)."""
    pot = Potential.from_run(force_run.path)
    pos, z = frames["R"][:4], frames["z"]
    pred = pot.predict(pos, z)
    assert pred.forces is not None and pred.forces.shape == (4, z.shape[0], 3)
    # The descriptors depend on the input order only for atoms sharing a grid cell
    # (row order within a cell, deviation A10). This permutation reverses the cell
    # order but keeps the order inside each cell, so the descriptors of every atom
    # are unchanged.
    cell = assign_cells(pos[0], pot.config.grid.grid_size).cell_of_atom
    perm = np.argsort(-cell, kind="stable")
    assert not np.array_equal(perm, np.arange(z.shape[0]))
    permuted = pot.predict(pos[:1, perm], z[perm])
    assert permuted.forces is not None
    np.testing.assert_allclose(permuted.energies, pred.energies[:1], rtol=1e-5)
    np.testing.assert_allclose(permuted.forces[0], pred.forces[0, perm], rtol=1e-4, atol=1e-6)
    # The forces differ between atoms, so the comparison is not trivial.
    assert np.ptp(pred.forces[0], axis=0).min() > 1e-3


def test_no_retrace(energy_run: RunDirectory, frames: dict[str, np.ndarray]) -> None:
    pot = Potential.from_run(energy_run.path)
    pos, z = frames["R"], frames["z"]
    pot.predict(pos[:10], z, batch_size=1)
    pot.predict(pos[:10], z, batch_size=3)
    pot.predict(pos, z, batch_size=256)
    pot.predict(pos[0, :12], z[:12])  # other atom count
    assert pot.tracing_count == 1


def test_predict_descriptors(
    energy_run: RunDirectory, smoke: tuple[ExperimentConfig, Path], frames: dict[str, np.ndarray]
) -> None:
    _, ds = smoke
    pot = Potential.from_run(energy_run.path)
    split = load_dataset(ds)["test"]
    molecules = [split.molecule(k) for k in range(len(split))]
    from_desc = pot.predict_descriptors(molecules, batch_size=2)
    from_pos = pot.predict(frames["R"][split.sample_index], frames["z"])
    np.testing.assert_allclose(from_desc.energies, from_pos.energies, rtol=1e-6)


def test_evaluate_matches_predict(
    energy_run: RunDirectory, smoke: tuple[ExperimentConfig, Path], frames: dict[str, np.ndarray]
) -> None:
    _, ds = smoke
    split = load_dataset(ds)["test"]
    idx = split.sample_index
    pred = Potential.from_run(energy_run.path).predict(frames["R"][idx], frames["z"])
    target = frames["E"].reshape(-1)[idx].astype(np.float32)
    mae = np.abs(pred.energies.astype(np.float32) - target).astype(np.float64).mean()
    result = evaluate(energy_run.path, split="test")
    np.testing.assert_allclose(result.energy_mae, mae, rtol=1e-6)


def test_missing_checkpoint(energy_run: RunDirectory) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint for epoch 9"):
        Potential.from_run(energy_run.path, epoch=9)


# --------------------------------------------------------------------------- C3-F27


def test_parallel_descriptors_identical(
    energy_run: RunDirectory, frames: dict[str, np.ndarray]
) -> None:
    """The worker path yields exactly the arrays of :meth:`Potential.descriptors`."""
    pot = Potential.from_run(energy_run.path)
    pos, z = frames["R"][:40].astype(np.float64), frames["z"].astype(np.int64)
    parallel = list(pot._parallel_frames(pos, z, 2))
    assert len(parallel) == 40
    for s, got in enumerate(parallel):
        expected = pot.descriptors(pos[s], z)
        assert got.keys() == expected.keys()
        for key in expected:
            assert got[key].dtype == expected[key].dtype
            np.testing.assert_array_equal(got[key], expected[key])


def test_predict_n_workers_identical(
    force_run: RunDirectory, frames: dict[str, np.ndarray]
) -> None:
    pot = Potential.from_run(force_run.path)
    pos, z = frames["R"][:40], frames["z"]
    serial = pot.predict(pos, z, batch_size=16)
    parallel = pot.predict(pos, z, batch_size=16, n_workers=2)
    assert serial.forces is not None and parallel.forces is not None
    np.testing.assert_array_equal(parallel.energies, serial.energies)
    np.testing.assert_array_equal(parallel.forces, serial.forces)
    assert pot.tracing_count == 1


def test_predict_n_workers_errors(energy_run: RunDirectory, frames: dict[str, np.ndarray]) -> None:
    pot = Potential.from_run(energy_run.path)
    dense = frames["R"][:10].copy()
    dense[7] *= 0.5  # second frame of the chunk starting at frame 6
    with pytest.raises(ValueError, match=rf"frame 7: .*n_max_local = {pot.n_max_local}"):
        pot.predict(dense, frames["z"], n_workers=2)
    with pytest.raises(ValueError, match="n_workers"):
        pot.predict(frames["R"][:2], frames["z"], n_workers=-1)
