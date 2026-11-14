"""Training loop (T09): run layout, logging, model selection, dropout, forces, determinism.

All runs use ``tests/fixtures/configs/smoke.toml`` (48 training and 6 validation
molecules of 21 atoms) and write into ``tmp_path`` only.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lrmlp.config import ExperimentConfig, load_config
from lrmlp.dataset import build_dataset
from lrmlp.model import build_model
from lrmlp.run import RunDirectory
from lrmlp.schedules import loss_weight
from lrmlp.training import Trainer, TrainingResult, log_columns, train

SMOKE = Path(__file__).resolve().parent / "fixtures" / "configs" / "smoke.toml"


def _cfg(*overrides: str) -> ExperimentConfig:
    return load_config(SMOKE, list(overrides))


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_dataset(_cfg(), output_dir=tmp_path_factory.mktemp("smoke") / "ds", n_workers=1)


def _weights(trainer: Trainer) -> list[np.ndarray]:
    return [np.asarray(w) for w in trainer.model.get_weights()]


def _read_log(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


# --------------------------------------------------------------------------- 1. layout and log


def test_run_layout_and_log_columns(dataset_dir: Path, tmp_path: Path) -> None:
    result = train(_cfg(), dataset_dir=dataset_dir, run_dir=tmp_path / "run")
    assert isinstance(result, TrainingResult)
    run = RunDirectory(result.run_dir)
    assert result.run_dir == tmp_path / "run"
    for path in (run.config_path, run.dataset_path, run.log_path, run.best_path):
        assert path.is_file(), path
    assert any(p.name.startswith("events.out.tfevents") for p in run.tb_dir.iterdir())

    config = run.read_config()
    meta = run.read_dataset_metadata()
    assert config.model.k1_local == meta["n_max_local"]
    assert config.model.k1_long == meta["n_max_long_range"]
    assert Path(meta["path"]) == dataset_dir.resolve()

    header, rows = _read_log(run.log_path)
    assert tuple(header) == log_columns(False)
    assert header == [
        "epoch",
        "train_loss",
        "train_energy_mse",
        "val_loss",
        "val_energy_mse",
        "learning_rate",
        "seconds",
    ]
    assert not any("force" in c or c.startswith("p_") for c in header)
    assert len(rows) == 2 == len(result.history)
    for row, hist in zip(rows, result.history, strict=True):
        values = {k: float(v) for k, v in row.items()}
        assert all(math.isfinite(v) for v in values.values()), row
        assert values == pytest.approx(hist, rel=0, abs=0)
        assert values["train_loss"] == values["train_energy_mse"]
        assert values["val_loss"] == values["val_energy_mse"]
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    assert result.steps == 2 * 48
    assert float(rows[-1]["learning_rate"]) == pytest.approx(1e-3 * 0.97 ** (96 // 20), rel=1e-6)


def test_learning_rate_column_is_rate_at_epoch_end(dataset_dir: Path, tmp_path: Path) -> None:
    """C3-F26: ``learning_rate`` is η(t) with t = steps after the epoch, as documented."""
    doc = log_columns.__doc__ or ""
    assert "``learning_rate``" in doc and "**end** of the epoch" in doc
    trainer = Trainer(_cfg("train.epochs=2"), dataset_dir, tmp_path / "run")
    result = trainer.fit()
    _, rows = _read_log(trainer.run.log_path)
    for epoch, row in enumerate(rows, start=1):
        steps = 48 * epoch
        logged = float(row["learning_rate"])
        assert logged == pytest.approx(float(trainer.lr_schedule(steps)), rel=1e-7)
        # Not the rate of the first step of the epoch (a decay boundary lies inside).
        assert logged != pytest.approx(float(trainer.lr_schedule(steps - 48)), rel=1e-7)
    assert result.steps == 96


def test_force_mode_log_columns(dataset_dir: Path, tmp_path: Path) -> None:
    result = train(
        _cfg("model.predict_forces=true", "train.epochs=1"),
        dataset_dir=dataset_dir,
        run_dir=tmp_path / "run",
    )
    header, rows = _read_log(result.run_dir / "training_log.csv")
    assert tuple(header) == log_columns(True)
    assert header[-4:] == ["train_force_mse", "val_force_mse", "p_energy", "p_force"]
    values = {k: float(v) for k, v in rows[0].items()}
    assert all(math.isfinite(v) for v in values.values())
    p_e, p_f = values["p_energy"], values["p_force"]
    assert p_e == pytest.approx(loss_weight(48, 1.0, 1.0, 7.5e3, 0.97))
    assert p_f == pytest.approx(loss_weight(48, 1.0, 0.0, 7.5e3, 0.97))
    assert values["val_loss"] == pytest.approx(
        p_e * values["val_energy_mse"] + p_f * values["val_force_mse"]
    )


def test_step_logging_interval(
    dataset_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = Trainer(_cfg("train.log_every_steps=10"), dataset_dir, tmp_path / "run")
    with caplog.at_level(logging.INFO, logger="lrmlp.training"):
        trainer.train_epoch(1)
    steps = [r.getMessage() for r in caplog.records if "step" in r.getMessage()]
    assert [m.split("step ")[1].split("/")[0] for m in steps] == ["10", "20", "30", "40"]


# ------------------------------------------------------------------- 2./3. best, checkpoints


def _scripted_validation(trainer: Trainer, values: list[float]) -> None:
    it: Iterator[float] = iter(values)

    def validate() -> dict[str, float]:
        return {"val_energy_mse": next(it)}

    trainer.validate = validate  # type: ignore[method-assign]


@pytest.mark.parametrize(("policy", "expected"), [("best_and_last", [2, 4]), ("all", [1, 2, 3, 4])])
def test_best_json_and_checkpoint_policy(
    dataset_dir: Path, tmp_path: Path, policy: str, expected: list[int]
) -> None:
    cfg = _cfg("train.epochs=4", f"train.checkpoints={policy}")
    trainer = Trainer(cfg, dataset_dir, tmp_path / "run")
    _scripted_validation(trainer, [3.0, 1.0, 2.0, 5.0])
    result = trainer.fit()
    assert result.best_epoch == 2
    assert result.best_val_energy_mse == 1.0
    assert trainer.run.read_best() == {"epoch": 2, "val_energy_mse": 1.0}
    assert trainer.run.list_checkpoints() == expected
    assert trainer.run.resolve_epoch("best") == 2
    assert trainer.run.resolve_epoch("last") == 4


def test_best_json_is_argmin_of_log(dataset_dir: Path, tmp_path: Path) -> None:
    result = train(
        _cfg("train.epochs=4", "train.checkpoints=all"),
        dataset_dir=dataset_dir,
        run_dir=tmp_path / "run",
    )
    _, rows = _read_log(result.run_dir / "training_log.csv")
    val = [float(r["val_energy_mse"]) for r in rows]
    best = RunDirectory(result.run_dir).read_best()
    assert best["epoch"] == int(np.argmin(val)) + 1 == result.best_epoch
    assert best["val_energy_mse"] == min(val) == result.best_val_energy_mse
    assert RunDirectory(result.run_dir).list_checkpoints() == [1, 2, 3, 4]


def test_best_checkpoint_reproduces_best_validation(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = Trainer(_cfg("train.epochs=3"), dataset_dir, tmp_path / "run")
    result = trainer.fit()
    trainer.model.load_weights(str(trainer.run.checkpoint_path(result.best_epoch)))
    assert trainer.validate()["val_energy_mse"] == pytest.approx(
        result.best_val_energy_mse, rel=1e-12
    )


def test_failed_checkpoint_keeps_best_json_valid(
    dataset_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3-F25: a checkpoint write that fails never leaves best.json pointing at it."""
    trainer = Trainer(_cfg("train.epochs=3"), dataset_dir, tmp_path / "run")
    _scripted_validation(trainer, [3.0, 1.0, 0.5])
    save = trainer.model.save_weights
    calls: list[str] = []

    def failing_save(path: str, *args: Any, **kwargs: Any) -> None:
        calls.append(path)
        if len(calls) == 2:  # epoch 2, the new best: write half a file, then fail
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")
        save(path, *args, **kwargs)

    monkeypatch.setattr(trainer.model, "save_weights", failing_save)
    with pytest.raises(OSError, match="disk full"):
        trainer.fit()
    # The checkpoint went to a hidden temporary file, never to the final name.
    assert Path(calls[1]).name.startswith(".") and Path(calls[1]).name.endswith(".weights.h5")
    assert Path(calls[1]).parent == trainer.run.checkpoint_dir
    assert trainer.run.read_best() == {"epoch": 1, "val_energy_mse": 3.0}
    assert trainer.run.list_checkpoints() == [1]
    assert sorted(p.name for p in trainer.run.checkpoint_dir.iterdir()) == ["epoch_0001.weights.h5"]
    assert trainer.run.resolve_epoch("best") == 1
    trainer.model.load_weights(str(trainer.run.checkpoint_path(1)))


def test_checkpoints_leave_no_temporary_files(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = Trainer(_cfg("train.epochs=3"), dataset_dir, tmp_path / "run")
    _scripted_validation(trainer, [3.0, 1.0, 2.0])
    trainer.fit()
    names = sorted(p.name for p in trainer.run.checkpoint_dir.iterdir())
    assert names == ["epoch_0002.weights.h5", "epoch_0003.weights.h5"]


def test_diverging_training_raises(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = Trainer(_cfg(), dataset_dir, tmp_path / "run")
    _scripted_validation(trainer, [float("nan"), 1.0])
    with pytest.raises(FloatingPointError, match="epoch 1"):
        trainer.fit()


# --------------------------------------------------------------------------- 4. dropout (B02)


@pytest.mark.parametrize("rate", [0.0, 0.5])
def test_dropout_active_in_train_step(dataset_dir: Path, tmp_path: Path, rate: float) -> None:
    trainer = Trainer(_cfg(f"model.dropout_rate={rate}"), dataset_dir, tmp_path / "run")
    inputs, targets = next(iter(trainer.train_dataset))
    before = _weights(trainer)
    inference = trainer.compute_loss(inputs, targets, training=False)
    for w, b in zip(_weights(trainer), before, strict=True):
        np.testing.assert_array_equal(w, b)
    step = trainer.train_step(inputs, targets)
    if rate == 0.0:
        assert step["loss"] == pytest.approx(inference["loss"], rel=1e-5)
    else:
        assert step["loss"] != pytest.approx(inference["loss"], rel=1e-3)


# --------------------------------------------------------------------------- 5. forces (B01)


def test_force_targets_aligned(dataset_dir: Path, tmp_path: Path) -> None:
    p = (2.0, 1.0, 0.5, 0.1, 1.0, 0.97)  # p_E start/limit, p_F start/limit, decay steps/rate
    cfg = _cfg(
        "model.predict_forces=true",
        "model.dropout_rate=0.0",
        "train.shuffle_buffer=1",
        f"train.p_energy_start={p[0]}",
        f"train.p_energy_limit={p[1]}",
        f"train.p_force_start={p[2]}",
        f"train.p_force_limit={p[3]}",
        f"train.loss_weight_decay_steps={p[4]}",
        f"train.loss_weight_decay_rate={p[5]}",
    )
    trainer = Trainer(cfg, dataset_dir, tmp_path / "run")
    split = trainer.dataset["train"]
    keys = [t.name for t in trainer.model.inputs]
    iterator = iter(trainer.train_dataset)
    for k in range(3):
        inputs, targets = next(iterator)
        mol = split.molecule(k)
        x = [mol[key].astype(np.float32) for key in keys]
        pred = np.asarray(trainer.model(x, training=False), dtype=np.float64)
        energy = float(np.float32(mol["energy"]))
        energy_se = (pred[:, 0].sum() - energy) ** 2
        force_mse = np.mean((pred[:, 1:] - mol["forces"]) ** 2)
        misaligned = np.mean((pred[:, 1:] - np.roll(mol["forces"], 1, axis=0)) ** 2)
        assert not np.isclose(force_mse, misaligned, rtol=1e-3)
        p_e = loss_weight(k, p[0], p[1], p[4], p[5])
        p_f = loss_weight(k, p[2], p[3], p[4], p[5])
        got = trainer.train_step(inputs, targets)
        assert got["energy_se"] == pytest.approx(energy_se, rel=1e-4, abs=1e-6)
        assert got["force_mse"] == pytest.approx(force_mse, rel=1e-5)
        assert got["loss"] == pytest.approx(p_e * energy_se + p_f * force_mse, rel=1e-4)


# --------------------------------------------------------------------------- 6./7. determinism


def _fit_weights(dataset_dir: Path, run_dir: Path, *overrides: str) -> list[np.ndarray]:
    trainer = Trainer(_cfg(*overrides), dataset_dir, run_dir)
    result = trainer.fit()
    assert result.steps == 48 * trainer.config.train.epochs
    return _weights(trainer)


@pytest.mark.parametrize("k", [4, 5])
def test_steps_per_execution_equivalent(dataset_dir: Path, tmp_path: Path, k: int) -> None:
    single = _fit_weights(dataset_dir, tmp_path / "k1", "train.steps_per_execution=1")
    multi = _fit_weights(dataset_dir, tmp_path / f"k{k}", f"train.steps_per_execution={k}")
    for a, b in zip(single, multi, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)


def test_seed_reproducible(dataset_dir: Path, tmp_path: Path) -> None:
    a = _fit_weights(dataset_dir, tmp_path / "a", "train.epochs=1")
    b = _fit_weights(dataset_dir, tmp_path / "b", "train.epochs=1")
    c = _fit_weights(dataset_dir, tmp_path / "c", "train.epochs=1", "train.seed=7")
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    assert any(not np.allclose(x, z) for x, z in zip(a, c, strict=True))


# --------------------------------------------------------------------------- 8. tracing


def test_no_retracing(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = Trainer(
        _cfg("train.epochs=3", "train.steps_per_execution=8"), dataset_dir, tmp_path / "run"
    )
    trainer.fit()
    trainer.validate()
    assert trainer.tracing_counts() == {"train_step": 1, "train_steps": 1, "val_step": 1}


# --------------------------------------------------------------------------- 9. shape checks (B08)


def test_n_max_from_metadata(dataset_dir: Path, tmp_path: Path) -> None:
    cfg = _cfg()
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    n_loc, n_lr = meta["n_max_local"], meta["n_max_long_range"]
    wrong = build_model(cfg.model, n_loc + 1, n_lr)
    with pytest.raises(ValueError, match=f"n_max_local={n_loc}"):
        Trainer(cfg, dataset_dir, tmp_path / "a", model=wrong)
    wrong = build_model(cfg.model, n_loc, n_lr - 1)
    with pytest.raises(ValueError, match=f"n_max_long_range={n_lr}"):
        Trainer(cfg, dataset_dir, tmp_path / "b", model=wrong)
    forces = build_model(_cfg("model.predict_forces=true").model, n_loc, n_lr)
    with pytest.raises(ValueError, match="predict_forces"):
        Trainer(cfg, dataset_dir, tmp_path / "c", model=forces)
    assert not (tmp_path / "a").exists()


def test_grid_mismatch_raises(dataset_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rebuild the dataset"):
        Trainer(_cfg("grid.max_local_level=3"), dataset_dir, tmp_path / "run")


def test_missing_validation_split_raises(tmp_path: Path) -> None:
    cfg = _cfg("data.val_fraction=0.0")
    path = build_dataset(cfg, output_dir=tmp_path / "ds", n_workers=1)
    with pytest.raises(ValueError, match="no validation molecules"):
        Trainer(cfg, path, tmp_path / "run")


def test_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build the dataset first"):
        Trainer(_cfg(), tmp_path / "nope", tmp_path / "run")


def test_run_dir_never_overwritten(dataset_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        Trainer(_cfg(), dataset_dir, tmp_path / "run")


def test_default_run_dir_under_run_root(dataset_dir: Path, tmp_path: Path) -> None:
    cfg = _cfg("train.epochs=1", f"train.run_root={json.dumps(str(tmp_path / 'runs'))}")
    result = train(cfg, dataset_dir=dataset_dir)
    assert result.run_dir.parent == tmp_path / "runs"
    assert result.run_dir.name.startswith("smoke-")


# --------------------------------------------------------------------------- 10. XLA


def test_jit_compile_smoke(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = Trainer(
        _cfg("train.epochs=1", "train.jit_compile=true", "train.steps_per_execution=4"),
        dataset_dir,
        tmp_path / "run",
    )
    result = trainer.fit()
    assert result.steps == 48
    assert all(math.isfinite(v) for v in result.history[0].values())
    assert trainer.tracing_counts()["train_step"] == 1


@pytest.mark.slow
def test_jit_compile_equivalent(dataset_dir: Path, tmp_path: Path) -> None:
    # The XLA step draws other dropout masks than the plain step, so the numerical
    # comparison is only meaningful without dropout.
    nodrop = ("train.epochs=1", "model.dropout_rate=0.0")
    plain = _fit_weights(dataset_dir, tmp_path / "plain", *nodrop)
    xla = _fit_weights(dataset_dir, tmp_path / "xla", *nodrop, "train.jit_compile=true")
    for a, b in zip(plain, xla, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


_DETERMINISTIC_SCRIPT = """
import sys
import numpy as np
from lrmlp.config import load_config
from lrmlp.training import Trainer
cfg = load_config(sys.argv[1], ["train.deterministic=true", "train.epochs=1"])
t = Trainer(cfg, sys.argv[2], sys.argv[3])
t.fit()
np.savez(sys.argv[4], *t.model.get_weights())
"""


@pytest.mark.slow
def test_deterministic_runs_identical(dataset_dir: Path, tmp_path: Path) -> None:
    # enable_op_determinism() is process-wide, so the runs are subprocesses.
    env: dict[str, Any] = {**os.environ}
    env.pop("TF_USE_LEGACY_KERAS", None)
    outs = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.npz"
        subprocess.run(
            [
                sys.executable,
                "-c",
                _DETERMINISTIC_SCRIPT,
                str(SMOKE),
                str(dataset_dir),
                str(tmp_path / name),
                str(out),
            ],
            check=True,
            env=env,
            capture_output=True,
        )
        outs.append(np.load(out))
    assert outs[0].files == outs[1].files
    for key in outs[0].files:
        np.testing.assert_array_equal(outs[0][key], outs[1][key])
