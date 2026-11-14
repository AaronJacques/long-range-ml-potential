"""Tests for the run directory contract (``lrmlp.run``)."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime
from pathlib import Path

import pytest

from lrmlp.config import load_config
from lrmlp.run import RunDirectory, new_run_dir, read_json, write_json

NOW = datetime(2026, 11, 14, 9, 5, 7)


@pytest.fixture
def config(fixtures_dir: Path):  # type: ignore[no-untyped-def]
    return load_config(fixtures_dir / "configs" / "smoke.toml")


@pytest.fixture
def run(tmp_path: Path, config) -> RunDirectory:  # type: ignore[no-untyped-def]
    metadata = {"n_max_local": 12, "n_max_long_range": 5, "path": "/data/smoke"}
    return RunDirectory(tmp_path / "run").create(config, metadata)


def touch_checkpoints(run: RunDirectory, *epochs: int) -> None:
    for epoch in epochs:
        run.checkpoint_path(epoch).write_bytes(b"")


# --------------------------------------------------------------------------- new_run_dir


def test_new_run_dir_name_format(tmp_path: Path) -> None:
    path = new_run_dir(tmp_path / "runs", "smoke", now=NOW)
    assert path == tmp_path / "runs" / "smoke-20261114-090507"
    assert path.is_dir()
    assert list(path.iterdir()) == []


def test_run_dir_unique(tmp_path: Path) -> None:
    paths = [new_run_dir(tmp_path, "smoke", now=NOW) for _ in range(3)]
    assert [p.name for p in paths] == [
        "smoke-20261114-090507",
        "smoke-20261114-090507-1",
        "smoke-20261114-090507-2",
    ]
    assert len(set(paths)) == 3


def test_new_run_dir_never_reuses_existing(tmp_path: Path) -> None:
    existing = tmp_path / "smoke-20261114-090507"
    existing.mkdir()
    (existing / "training_log.csv").write_text("old", encoding="utf-8")
    path = new_run_dir(tmp_path, "smoke", now=NOW)
    assert path != existing
    assert (existing / "training_log.csv").read_text(encoding="utf-8") == "old"


def test_new_run_dir_uses_current_time(tmp_path: Path) -> None:
    before = datetime.now()
    path = new_run_dir(tmp_path, "x")
    after = datetime.now()
    assert any(path.name.startswith(f"x-{t:%Y%m%d}-") for t in (before, after))


def test_write_json_uses_default_file_mode(tmp_path: Path) -> None:
    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    path = write_json(tmp_path / "config.json", {"a": 1})
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)


@pytest.mark.parametrize("umask", [0o077, 0o027])
def test_write_json_follows_current_umask(tmp_path: Path, umask: int) -> None:
    """C2-F15: the OS applies the umask at creation; the umask itself is not touched."""
    previous = os.umask(umask)
    try:
        path = write_json(tmp_path / "best.json", {"epoch": 1})
        assert os.umask(umask) == umask  # write_json left the umask unchanged
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask
    assert read_json(path) == {"epoch": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["best.json"]  # no temporary file left


@pytest.mark.parametrize("name", ["", "a/b", "..", "a\\b"])
def test_new_run_dir_rejects_bad_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="run name"):
        new_run_dir(tmp_path, name)


# --------------------------------------------------------------------------- layout


def test_layout(run: RunDirectory) -> None:
    p = run.path
    assert run.config_path == p / "config.json"
    assert run.dataset_path == p / "dataset.json"
    assert run.log_path == p / "training_log.csv"
    assert run.best_path == p / "best.json"
    assert run.tb_dir == p / "tb"
    assert run.checkpoint_dir == p / "checkpoints"
    assert run.checkpoint_path(7) == p / "checkpoints" / "epoch_0007.weights.h5"
    assert run.checkpoint_path(12345).name == "epoch_12345.weights.h5"
    assert run.evaluation_path("test", 7) == p / "evaluation_test_7.json"
    assert run.tb_dir.is_dir() and run.checkpoint_dir.is_dir()
    assert sorted(x.name for x in p.iterdir()) == [
        "checkpoints",
        "config.json",
        "dataset.json",
        "tb",
    ]


def test_config_and_metadata_roundtrip(run: RunDirectory, config) -> None:  # type: ignore[no-untyped-def]
    assert run.read_config() == config
    assert run.read_dataset_metadata() == {
        "n_max_local": 12,
        "n_max_long_range": 5,
        "path": "/data/smoke",
    }
    stored = json.loads(run.config_path.read_text(encoding="utf-8"))
    assert stored["derived"]["cutoff"] == config.grid.cutoff


def test_json_utf8_sorted_keys(tmp_path: Path) -> None:
    path = write_json(tmp_path / "x.json", {"b": 1, "a": {"z": "Δg", "y": 2}})
    text = path.read_bytes().decode("utf-8")
    assert "Δg" in text
    assert text.index('"a"') < text.index('"b"')
    assert text.index('"y"') < text.index('"z"')
    assert read_json(path) == {"a": {"y": 2, "z": "Δg"}, "b": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]  # no temp file left


def test_create_never_overwrites(run: RunDirectory, config) -> None:  # type: ignore[no-untyped-def]
    before = run.config_path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        RunDirectory(run.path).create(config.with_overrides(["train.epochs=9"]), {})
    assert run.config_path.read_bytes() == before


def test_create_accepts_existing_empty_dir(tmp_path: Path, config) -> None:  # type: ignore[no-untyped-def]
    path = new_run_dir(tmp_path, config.name, now=NOW)
    run = RunDirectory(path).create(config.to_dict(), {})
    assert run.read_config() == config


def test_missing_files_raise_clear_errors(tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "nothing")
    with pytest.raises(FileNotFoundError, match="config.json"):
        run.read_config()
    with pytest.raises(FileNotFoundError, match="dataset.json"):
        run.read_dataset_metadata()
    assert run.list_checkpoints() == []


# --------------------------------------------------------------------------- best / epochs


def test_write_and_read_best(run: RunDirectory) -> None:
    run.write_best(3, {"val_energy_mse": 0.25})
    assert run.read_best() == {"epoch": 3, "val_energy_mse": 0.25}
    run.write_best(5, {"val_energy_mse": 0.125})
    assert run.read_best() == {"epoch": 5, "val_energy_mse": 0.125}


def test_list_checkpoints_sorted_and_filtered(run: RunDirectory) -> None:
    touch_checkpoints(run, 10, 2, 1)
    (run.checkpoint_dir / "notes.txt").write_text("x", encoding="utf-8")
    (run.checkpoint_dir / "epoch_3.weights.h5").write_bytes(b"")  # wrong width
    assert run.list_checkpoints() == [1, 2, 10]


def test_resolve_epoch(run: RunDirectory) -> None:
    touch_checkpoints(run, 1, 2, 3)
    run.write_best(2, {"val_energy_mse": 0.5})
    assert run.resolve_epoch("best") == 2
    assert run.resolve_epoch("last") == 3
    assert run.resolve_epoch(1) == 1
    assert run.resolve_epoch("3") == 3


def test_resolve_epoch_without_files(run: RunDirectory) -> None:
    with pytest.raises(FileNotFoundError, match="best.json"):
        run.resolve_epoch("best")
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        run.resolve_epoch("last")
    with pytest.raises(FileNotFoundError, match="epoch 4.*available epochs: none"):
        run.resolve_epoch(4)


def test_resolve_epoch_missing_checkpoint(run: RunDirectory) -> None:
    touch_checkpoints(run, 1, 2)
    run.write_best(5, {"val_energy_mse": 0.5})
    with pytest.raises(FileNotFoundError, match=r"epoch 5.*\[1, 2\]"):
        run.resolve_epoch("best")
    with pytest.raises(FileNotFoundError, match="epoch 9"):
        run.resolve_epoch(9)


@pytest.mark.parametrize("epoch", ["first", "-1", 1.5, True])
def test_resolve_epoch_rejects_bad_values(run: RunDirectory, epoch: object) -> None:
    with pytest.raises(ValueError, match="epoch"):
        run.resolve_epoch(epoch)  # type: ignore[arg-type]


@pytest.mark.parametrize("epoch", [-1, 1.0, True])
def test_checkpoint_path_rejects_bad_epochs(run: RunDirectory, epoch: object) -> None:
    with pytest.raises(ValueError, match="epoch"):
        run.checkpoint_path(epoch)  # type: ignore[arg-type]
