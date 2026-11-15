"""Tests of ``lrmlp.evaluation.evaluate`` on hand-made run directories."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lrmlp.config import ExperimentConfig, load_config
from lrmlp.dataset import build_dataset, load_dataset
from lrmlp.evaluation import EvaluationResult, dataset_mismatches, evaluate, resolve_dataset_dir
from lrmlp.model import build_model
from lrmlp.run import RunDirectory, read_json

CONFIG = Path(__file__).resolve().parent / "fixtures" / "configs" / "smoke.toml"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> tuple[ExperimentConfig, Path]:
    config = load_config(CONFIG)
    path = build_dataset(config, output_dir=tmp_path_factory.mktemp("ds") / "smoke", n_workers=1)
    return config, path


def _constant_weights(model: Any, output_bias: np.ndarray) -> None:
    """All weights zero except the output bias, so every atom outputs ``output_bias``."""
    weights = [np.zeros_like(w) for w in model.get_weights()]
    weights[-1] = np.asarray(output_bias, dtype=np.float32)
    model.set_weights(weights)


def make_run(
    root: Path,
    config: ExperimentConfig,
    dataset_dir: Path,
    checkpoints: Mapping[int, Callable[[Any], None]],
    best: int | None = None,
) -> RunDirectory:
    """Run directory with one checkpoint per epoch; ``checkpoints[e](model)`` sets the weights."""
    ds = load_dataset(dataset_dir)
    run = RunDirectory(root / "run").create(config, {**ds.metadata, "path": str(dataset_dir)})
    model = build_model(
        config.model, ds.n_max_local, ds.n_max_long_range, config.grid.max_atomic_number
    )
    for epoch, set_weights in checkpoints.items():
        set_weights(model)
        model.save_weights(run.checkpoint_path(epoch))
    if best is not None:
        run.write_best(best, {"val_energy_mse": 1.0})
    return run


def _bias(b: float, width: int = 1) -> Callable[[Any], None]:
    return lambda model: _constant_weights(model, np.r_[b, np.zeros(width - 1)])


def _expected(dataset_dir: Path, split: str, molecule_energy: np.ndarray) -> dict[str, float]:
    energies = load_dataset(dataset_dir)[split].energies.astype(np.float32)
    err = np.abs(molecule_energy.astype(np.float32) - energies).astype(np.float64)
    return {"mae": err.mean(), "mse": (err**2).mean(), "rmse": np.sqrt((err**2).mean())}


def _n_atoms(dataset_dir: Path, split: str) -> np.ndarray:
    return np.diff(load_dataset(dataset_dir)[split].atom_offsets)


def test_metrics_hand_case(tmp_path: Path, smoke: tuple[ExperimentConfig, Path]) -> None:
    config, ds = smoke
    b = -3.25
    run = make_run(tmp_path, config, ds, {1: _bias(b)}, best=1)
    result = evaluate(run.path, split="test")
    assert isinstance(result, EvaluationResult)
    exp = _expected(ds, "test", b * _n_atoms(ds, "test"))
    assert result.n_molecules == len(load_dataset(ds)["test"]) > 0
    assert result.epoch == 1 and result.split == "test"
    np.testing.assert_allclose(result.energy_mae, exp["mae"], rtol=1e-6)
    np.testing.assert_allclose(result.energy_mse, exp["mse"], rtol=1e-6)
    np.testing.assert_allclose(result.energy_rmse, exp["rmse"], rtol=1e-6)
    assert result.force_mae is None and result.force_rmse is None
    assert result.energy_abs_errors.shape == (result.n_molecules,)


def test_batch_size_does_not_change_result(
    tmp_path: Path, smoke: tuple[ExperimentConfig, Path]
) -> None:
    config, ds = smoke
    run = make_run(tmp_path, config, ds, {1: _bias(0.5)}, best=1)
    a = evaluate(run.path, split="val", batch_size=1)
    b = evaluate(run.path, split="val", batch_size=256)
    np.testing.assert_allclose(a.energy_abs_errors, b.energy_abs_errors, rtol=1e-6)


@pytest.mark.parametrize(("epoch", "expected"), [("best", 2), ("last", 3), (1, 1), ("1", 1)])
def test_epoch_selection(
    tmp_path: Path, smoke: tuple[ExperimentConfig, Path], epoch: str | int, expected: int
) -> None:
    config, ds = smoke
    biases = {1: -1.0, 2: -2.0, 3: -3.0}
    run = make_run(tmp_path, config, ds, {e: _bias(b) for e, b in biases.items()}, best=2)
    result = evaluate(run.path, split="test", epoch=epoch)
    assert result.epoch == expected
    exp = _expected(ds, "test", biases[expected] * _n_atoms(ds, "test"))
    np.testing.assert_allclose(result.energy_mae, exp["mae"], rtol=1e-6)


def test_writes_json(tmp_path: Path, smoke: tuple[ExperimentConfig, Path]) -> None:
    config, ds = smoke
    run = make_run(tmp_path, config, ds, {4: _bias(1.0)}, best=4)
    result = evaluate(run.path, split="val")
    path = run.path / "evaluation_val_4.json"
    assert path == run.evaluation_path("val", 4)
    data = read_json(path)
    for key, value in result.to_dict().items():
        assert data[key] == value, key
    assert data["dataset"] == str(ds)
    assert set(result.to_dict()) == {
        "energy_mae",
        "energy_rmse",
        "energy_mse",
        "force_mae",
        "force_rmse",
        "n_molecules",
        "epoch",
        "split",
    }
    json.dumps(result.to_dict())  # plain JSON types


def test_force_model_uses_energy_column(
    tmp_path: Path, smoke: tuple[ExperimentConfig, Path]
) -> None:
    """B05: the energy is the sum of output column 0, not of all outputs."""
    config, ds = smoke
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, predict_forces=True)
    )
    bias = np.array([-2.0, 0.5, -1.0, 3.0])
    run = make_run(tmp_path, config, ds, {1: lambda m: _constant_weights(m, bias)}, best=1)
    result = evaluate(run.path, split="test")

    n = _n_atoms(ds, "test")
    exp = _expected(ds, "test", bias[0] * n)
    legacy = _expected(ds, "test", bias.sum() * n)  # evaluate.py:23 summed all outputs
    np.testing.assert_allclose(result.energy_mae, exp["mae"], rtol=1e-6)
    assert abs(result.energy_mae - legacy["mae"]) > 1.0

    forces = np.asarray(load_dataset(ds)["test"].arrays["forces"], dtype=np.float64)
    diff = bias[1:].astype(np.float32).astype(np.float64) - forces
    assert result.force_mae is not None and result.force_rmse is not None
    np.testing.assert_allclose(result.force_mae, np.abs(diff).mean(), rtol=1e-6)
    np.testing.assert_allclose(result.force_rmse, np.sqrt((diff**2).mean()), rtol=1e-6)


def test_missing_checkpoint_is_clear(tmp_path: Path, smoke: tuple[ExperimentConfig, Path]) -> None:
    config, ds = smoke
    run = make_run(tmp_path, config, ds, {1: _bias(0.0), 2: _bias(0.0)})
    with pytest.raises(FileNotFoundError, match="best.json"):
        evaluate(run.path)
    with pytest.raises(FileNotFoundError, match=r"no checkpoint for epoch 7.*\[1, 2\]"):
        evaluate(run.path, epoch=7)
    run.write_best(5, {"val_energy_mse": 1.0})
    with pytest.raises(FileNotFoundError, match="no checkpoint for epoch 5"):
        evaluate(run.path)


def test_missing_run_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        evaluate(tmp_path)


def test_unknown_split(tmp_path: Path, smoke: tuple[ExperimentConfig, Path]) -> None:
    config, ds = smoke
    run = make_run(tmp_path, config, ds, {1: _bias(0.0)}, best=1)
    with pytest.raises(ValueError, match="unknown split 'holdout'"):
        evaluate(run.path, split="holdout")


def test_dataset_dir_override_and_padding_check(
    tmp_path: Path, smoke: tuple[ExperimentConfig, Path]
) -> None:
    config, ds = smoke
    run = make_run(tmp_path, config, ds, {1: _bias(0.0)}, best=1)
    assert resolve_dataset_dir(run) == ds
    assert resolve_dataset_dir(run, tmp_path / "other") == tmp_path / "other"

    # A dataset built with another grid is not the run's dataset ...
    other = config.with_overrides({"grid.max_local_level": 1})
    other_ds = build_dataset(other, output_dir=tmp_path / "other_ds", n_workers=1)
    with pytest.raises(ValueError, match=r"grid\.max_local_level.*n_max_local"):
        evaluate(run.path, dataset_dir=other_ds)
    # ... and its other N_max is rejected even with the opt-out (B08).
    with pytest.raises(ValueError, match="n_max_local, n_max_long_range"):
        evaluate(run.path, dataset_dir=other_ds, allow_dataset_mismatch=True)


def test_dataset_json_without_path(tmp_path: Path, smoke: tuple[ExperimentConfig, Path]) -> None:
    config, ds = smoke
    meta = load_dataset(ds).metadata
    run = RunDirectory(tmp_path / "run").create(config, meta)
    with pytest.raises(ValueError, match="pass dataset_dir explicitly"):
        resolve_dataset_dir(run)


# --------------------------------------------------------------------------- dataset identity


LEGACY_CONFIG = CONFIG.parent / "golden_legacy.toml"


def test_rebuilt_identical_dataset_is_accepted(
    tmp_path: Path, smoke: tuple[ExperimentConfig, Path]
) -> None:
    """Same configuration, other directory and build time: no mismatch, JSON unchanged."""
    config, ds = smoke
    run = make_run(tmp_path, config, ds, {1: _bias(0.0)}, best=1)
    rebuilt = build_dataset(config, output_dir=tmp_path / "rebuilt", n_workers=1)
    result = evaluate(run.path, dataset_dir=rebuilt)
    assert result.dataset_mismatch == ()
    stored = read_json(run.evaluation_path("test", 1))
    assert "dataset_matches_run" not in stored and "dataset_mismatch" not in stored


def test_legacy_run_on_thesis_dataset_rejected(
    tmp_path: Path, smoke: tuple[ExperimentConfig, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """Mismatch 1: same N_max, other grid assignment."""
    config, thesis_ds = smoke
    legacy = load_config(LEGACY_CONFIG)
    legacy_ds = build_dataset(legacy, output_dir=tmp_path / "legacy_ds", n_workers=1)
    assert (load_dataset(legacy_ds).n_max_local, load_dataset(legacy_ds).n_max_long_range) == (
        load_dataset(thesis_ds).n_max_local,
        load_dataset(thesis_ds).n_max_long_range,
    )
    run = make_run(tmp_path, legacy, legacy_ds, {1: _bias(-1.0)}, best=1)

    with pytest.raises(ValueError, match=r"grid\.assignment \(run: 'legacy', dataset: 'thesis'\)"):
        evaluate(run.path, dataset_dir=thesis_ds)
    # Only the grid differs (n_samples_per_source "all" vs. 65000 select the same frames).
    assert dataset_mismatches(
        RunDirectory(run.path).read_dataset_metadata(), load_dataset(thesis_ds).metadata
    ) == ["grid.assignment (run: 'legacy', dataset: 'thesis')"]
    assert not run.evaluation_path("test", 1).exists()

    with caplog.at_level("WARNING", logger="lrmlp.evaluation"):
        result = evaluate(run.path, dataset_dir=thesis_ds, allow_dataset_mismatch=True)
    assert "NOT the dataset the run was trained on" in caplog.text
    assert "grid.assignment" in caplog.text
    assert any(m.startswith("grid.assignment") for m in result.dataset_mismatch)
    stored = read_json(run.evaluation_path("test", 1))
    assert stored["dataset_matches_run"] is False
    assert stored["dataset_mismatch"] == list(result.dataset_mismatch)
    assert result.to_dict()["dataset_matches_run"] is False


def test_other_split_seed_rejected(
    tmp_path: Path, smoke: tuple[ExperimentConfig, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """Mismatch 2: same grid, other split; its test split holds training frames."""
    config, ds = smoke
    run = make_run(tmp_path, config, ds, {1: _bias(-1.0)}, best=1)
    seed7 = build_dataset(
        config.with_overrides({"data.split_seed": 7}), output_dir=tmp_path / "s7", n_workers=1
    )
    train_frames = set(load_dataset(ds)["train"].sample_index.tolist())
    assert train_frames & set(load_dataset(seed7)["test"].sample_index.tolist())

    with pytest.raises(ValueError, match=r"split_seed \(run: 42, dataset: 7\)") as info:
        evaluate(run.path, dataset_dir=seed7)
    assert "--allow-dataset-mismatch" in str(info.value)

    with caplog.at_level("WARNING", logger="lrmlp.evaluation"):
        result = evaluate(run.path, dataset_dir=seed7, allow_dataset_mismatch=True)
    assert "split_seed" in caplog.text
    assert "split_seed (run: 42, dataset: 7)" in result.dataset_mismatch
    assert np.isfinite(result.energy_mae)
    assert read_json(run.evaluation_path("test", 1))["dataset_matches_run"] is False


def test_dataset_rebuilt_in_place_rejected(
    tmp_path: Path, smoke: tuple[ExperimentConfig, Path]
) -> None:
    """The default dataset path of dataset.json is checked too (prepare --overwrite)."""
    config, ds = smoke
    path = build_dataset(config, output_dir=tmp_path / "ds", n_workers=1)
    run = make_run(tmp_path, config, path, {1: _bias(0.0)}, best=1)
    build_dataset(
        config.with_overrides({"data.val_fraction": 0.2}),
        output_dir=path,
        n_workers=1,
        overwrite=True,
    )
    with pytest.raises(ValueError, match=r"val_fraction \(run: 0\.1, dataset: 0\.2\)"):
        evaluate(run.path)


def test_dataset_mismatches_fields() -> None:
    base: dict[str, Any] = {
        "grid": {"assignment": "thesis", "sort_neighbours": False, "grid_size": 1.0},
        "sources": [{"path": "/a/x.npz", "sha256": "aa", "n_frames_used": 60}],
        "n_samples_per_source": "all",
        "n_samples": 60,
        "split_seed": 42,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "species": [1, 6, 8],
        "n_max_local": 10,
        "n_max_long_range": 20,
        "created": "2026-01-01",
        "name": "a",
        "path": "/ds/a",
    }
    other = json.loads(json.dumps(base))
    other.update(created="2026-02-02", name="b", path="/ds/b")
    other["sources"][0]["path"] = "/moved/x.npz"
    other["n_samples_per_source"] = 1000  # a cap above the frame count: same frames
    assert dataset_mismatches(base, other) == []

    other["grid"]["sort_neighbours"] = True
    other["sources"][0]["sha256"] = "bb"
    other["sources"][0]["n_frames_used"] = 50
    other["test_fraction"] = 0.2
    other["species"] = [1, 6]
    del other["n_max_local"]
    assert dataset_mismatches(base, other) == [
        "grid.sort_neighbours (run: False, dataset: True)",
        "sources[0].n_frames_used (run: 60, dataset: 50)",
        "sources[0].sha256 (run: 'aa', dataset: 'bb')",
        "test_fraction (run: 0.1, dataset: 0.2)",
        "species (run: [1, 6, 8], dataset: [1, 6])",
        "n_max_local (run: 10, dataset: '<missing>')",
    ]
    two_sources = {**base, "sources": base["sources"] * 2}
    assert [m.split(" ")[0] for m in dataset_mismatches(base, two_sources)] == ["sources"]
