"""Tests of the hyperparameter search ``lrmlp.tuning`` (T13, B04)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lrmlp import cli, tuning
from lrmlp.config import ExperimentConfig, load_config
from lrmlp.run import RunDirectory, read_json
from lrmlp.tuning import SearchSpace, load_space, run_search

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "configs" / "hparam_search.example.toml"
SMOKE = REPO / "tests" / "fixtures" / "configs" / "smoke.toml"
LEGACY_SIZE = 3 * 4 * 4 * 4 * 3 * 3 * 3 * 3


def _write_space(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _fake_dataset(tmp_path: Path) -> Path:
    """Directory that passes the dataset pre-check of run_search."""
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "metadata.json").write_text("{}", encoding="utf-8")
    return ds


def _event_tags(directory: Path) -> set[str]:
    import tensorflow as tf

    tags: set[str] = set()
    for event_file in directory.glob("events.out.tfevents*"):
        for event in tf.compat.v1.train.summary_iterator(str(event_file)):
            tags.update(v.tag for v in event.summary.value)
    return tags


@pytest.fixture
def fake_train(monkeypatch: pytest.MonkeyPatch) -> list[ExperimentConfig]:
    """Replace ``lrmlp.training.train`` by a fast stand-in that writes config.json."""
    import lrmlp.training

    seen: list[ExperimentConfig] = []

    def train(config: ExperimentConfig, dataset_dir: Any = None, run_dir: Any = None) -> Any:
        seen.append(config)
        RunDirectory(run_dir).create(config, {"path": "fake"})
        if config.train.learning_rate == 0.5:
            raise RuntimeError("diverged")
        return SimpleNamespace(best_val_energy_mse=config.train.learning_rate * 10, best_epoch=1)

    monkeypatch.setattr(lrmlp.training, "train", train)
    return seen


# --------------------------------------------------------------------------- space


def test_dry_run_counts_trials(tmp_path: Path) -> None:
    results = run_search(SMOKE, EXAMPLE, dry_run=True, run_root=tmp_path)
    assert len(load_space(EXAMPLE)) == LEGACY_SIZE == 15552
    assert len(results) == results.n_combinations == 15552
    assert {t.status for t in results} == {"planned"}
    assert "15552 of 15552" in str(results)
    limited = run_search(SMOKE, EXAMPLE, max_trials=7, dry_run=True, run_root=tmp_path)
    assert [t.name for t in limited] == [f"trial-{i:04d}" for i in range(7)]
    assert limited.n_combinations == 15552
    assert [t.overrides for t in limited] == [t.overrides for t in results[:7]]
    assert not any(tmp_path.iterdir())  # a dry run writes nothing


def test_grid_order_is_deterministic_last_axis_fastest(tmp_path: Path) -> None:
    space = _write_space(
        tmp_path / "s.toml", '[space]\n"train.epochs" = [1, 2]\ntrain.seed = [7, 8, 9]\n'
    )
    results = run_search(SMOKE, space, dry_run=True)
    pairs = [(t.overrides["train.epochs"], t.overrides["train.seed"]) for t in results]
    assert pairs == [(1, 7), (1, 8), (1, 9), (2, 7), (2, 8), (2, 9)]


def test_hparam_names_unique() -> None:
    space = load_space(EXAMPLE)
    assert len(set(space.names)) == len(space.names) == 9
    assert len(set(space.keys)) == len(space.keys)
    # B04: the legacy code registered "lr_decay_steps" for two different schedules
    assert "train.lr_decay_steps" in space.names
    assert "train.loss_weight_decay_steps" in space.names
    assert "decay_rate" in space.names
    with pytest.raises(ValueError, match="duplicate hyperparameter name"):
        SearchSpace.from_dict(
            {
                "space": {"train.epochs": [1]},
                "linked": [{"name": "train.epochs", "keys": ["train.seed"], "values": [1]}],
            }
        )
    with pytest.raises(ValueError, match="more than one axis"):
        SearchSpace.from_dict(
            {
                "space": {"train.seed": [1]},
                "linked": [{"name": "s", "keys": ["train.seed"], "values": [2]}],
            }
        )


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({}, "no axes"),
        ({"space": {"train.epochs": []}}, "no values"),
        ({"space": {"train.epochs": 3}}, "list of values"),
        ({"spaces": {}}, "unknown table"),
        ({"linked": [{"keys": "train.seed", "values": [1]}]}, "list of strings"),
        ({"linked": [{"keys": ["train.seed"], "values": [1], "x": 1}]}, "unknown field"),
    ],
)
def test_invalid_space(data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SearchSpace.from_dict(data)


def test_linked_keys_set_both_fields() -> None:
    space = load_space(EXAMPLE)
    combos = list(space.combinations())
    assert all(c["train.lr_decay_rate"] == c["train.loss_weight_decay_rate"] for c in combos)
    assert {c["train.lr_decay_rate"] for c in combos} == {0.96, 0.97, 0.98, 0.99}
    config = load_config(SMOKE, combos[1])
    assert config.train.lr_decay_rate == config.train.loss_weight_decay_rate == 0.97
    assert space.hparams(combos[1])["decay_rate"] == 0.97


def test_nested_tables_are_dotted_keys(tmp_path: Path) -> None:
    space = load_space(_write_space(tmp_path / "s.toml", "[space.train]\nseed = [1, 2]\n"))
    assert space.keys == ("train.seed",)


def test_unknown_key_rejected_before_training(tmp_path: Path) -> None:
    space = _write_space(tmp_path / "s.toml", '[space]\n"train.nope" = [1]\n')
    with pytest.raises(ValueError, match="unknown override key"):
        run_search(SMOKE, space, run_root=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_max_trials_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_trials"):
        run_search(SMOKE, EXAMPLE, max_trials=0, dry_run=True)


def test_missing_dataset_fails_early(tmp_path: Path) -> None:
    space = _write_space(
        tmp_path / "s.toml", f'[space]\n"data.output_dir" = ["{(tmp_path / "none").as_posix()}"]\n'
    )
    with pytest.raises(FileNotFoundError, match="lrmlp prepare"):
        run_search(SMOKE, space, run_root=tmp_path / "runs")


# --------------------------------------------------------------------------- search


def _small_space(tmp_path: Path, rates: str) -> Path:
    ds = _fake_dataset(tmp_path)
    return _write_space(
        tmp_path / "s.toml",
        f'[space]\n"data.output_dir" = ["{ds.as_posix()}"]\n"train.learning_rate" = [{rates}]\n'
        '[[linked]]\nname = "decay_rate"\n'
        'keys = ["train.lr_decay_rate", "train.loss_weight_decay_rate"]\nvalues = [0.9, 0.8]\n',
    )


def test_overrides_reach_config(tmp_path: Path, fake_train: list[ExperimentConfig]) -> None:
    space = _small_space(tmp_path, "0.01, 0.02")
    results = run_search(SMOKE, space, run_root=tmp_path / "runs")
    assert len(results) == 4
    for trial in results:
        assert trial.run_dir is not None
        stored = read_json(trial.run_dir / "config.json")
        assert stored["train"]["learning_rate"] == trial.overrides["train.learning_rate"]
        rate = trial.overrides["train.lr_decay_rate"]
        assert stored["train"]["lr_decay_rate"] == stored["train"]["loss_weight_decay_rate"] == rate
        assert stored["name"] == "smoke"


def test_trial_dirs_unique(tmp_path: Path, fake_train: list[ExperimentConfig]) -> None:
    space = _small_space(tmp_path, "0.01, 0.02")
    first = run_search(SMOKE, space, run_root=tmp_path / "runs")
    second = run_search(SMOKE, space, run_root=tmp_path / "runs")
    assert first.search_dir != second.search_dir
    assert first.search_dir is not None and first.search_dir.name.startswith("tune-smoke-")
    dirs = [t.run_dir for t in [*first, *second]]
    assert len(set(dirs)) == len(dirs) == 8
    assert [t.run_dir.name for t in first if t.run_dir] == [f"trial-{i:04d}" for i in range(4)]
    assert all(t.run_dir.parent == first.search_dir for t in first if t.run_dir)


def test_failed_trial_does_not_stop_search(
    tmp_path: Path, fake_train: list[ExperimentConfig]
) -> None:
    space = _small_space(tmp_path, "0.5, 0.01")
    results = run_search(SMOKE, space, run_root=tmp_path / "runs")
    assert [t.status for t in results] == ["failed", "failed", "succeeded", "succeeded"]
    assert results[0].message == "RuntimeError: diverged"
    assert results[0].best_val_energy_mse is None
    assert results[2].best_val_energy_mse == pytest.approx(0.1)
    best = results.best()
    assert best is not None and best.name == "trial-0002"
    assert results.search_dir is not None
    with (results.search_dir / "summary.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["status"] for r in rows] == ["failed", "failed", "succeeded", "succeeded"]
    assert rows[0]["message"] == "RuntimeError: diverged"
    assert rows[2]["train.learning_rate"] == "0.01"
    assert float(rows[2]["best_val_energy_mse"]) == pytest.approx(0.1)
    assert Path(rows[3]["run_dir"]) == results[3].run_dir


def test_hparams_written(tmp_path: Path, fake_train: list[ExperimentConfig]) -> None:
    space = _small_space(tmp_path, "0.01")
    results = run_search(SMOKE, space, run_root=tmp_path / "runs")
    assert results.search_dir is not None
    assert "_hparams_/experiment" in _event_tags(results.search_dir)
    trial_tags = _event_tags(results[0].run_dir or Path())
    assert {"_hparams_/session_start_info", tuning.METRIC} <= trial_tags
    search = json.loads((results.search_dir / "search.json").read_text())
    assert search["n_combinations"] == 2
    assert [a["name"] for a in search["axes"]] == [
        "data.output_dir",
        "train.learning_rate",
        "decay_rate",
    ]


@pytest.mark.slow
def test_two_real_trials(tmp_path: Path) -> None:
    from lrmlp.dataset import build_dataset

    ds = build_dataset(load_config(SMOKE), output_dir=tmp_path / "ds", n_workers=1)
    space = _write_space(
        tmp_path / "s.toml",
        f'[space]\n"data.output_dir" = ["{ds.as_posix()}"]\n"train.epochs" = [1]\n'
        '"train.learning_rate" = [1e-3, 1e-4]\n',
    )
    results = run_search(SMOKE, space, run_root=tmp_path / "runs")
    assert [t.status for t in results] == ["succeeded", "succeeded"], [t.message for t in results]
    for trial in results:
        assert trial.run_dir is not None
        rd = RunDirectory(trial.run_dir)
        assert rd.read_config().train.learning_rate == trial.overrides["train.learning_rate"]
        assert rd.read_best()["epoch"] == trial.best_epoch == 1
        assert trial.best_val_energy_mse == pytest.approx(rd.read_best()["val_energy_mse"])
        assert {"_hparams_/session_start_info", tuning.METRIC} <= _event_tags(trial.run_dir)
    assert results.search_dir is not None
    with (results.search_dir / "summary.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["trial"] for r in rows] == ["trial-0000", "trial-0001"]
    assert all(float(r["best_val_energy_mse"]) >= 0 for r in rows)
    assert "_hparams_/experiment" in _event_tags(results.search_dir)


# --------------------------------------------------------------------------- CLI


def test_cli_tune_dry_run(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    argv = ["tune", "--config", str(SMOKE), "--space", str(EXAMPLE), "--dry-run"]
    with caplog.at_level("INFO", logger="lrmlp"):
        assert cli.main(argv) == 0
    assert "15552 combination(s)" in caplog.text
    assert cli.main([*argv, "--max-trials", "3"]) == 0


def test_energy_only_example_space(tmp_path: Path) -> None:
    """C4-F34: the energy-only example trains the method of the thesis (no force mode)."""
    energy = REPO / "configs" / "hparam_search.energy.toml"
    space = load_space(energy)
    assert len(space) == 3 * 4 * 4
    assert set(space.keys) == {
        "train.learning_rate",
        "train.lr_decay_steps",
        "train.lr_decay_rate",
        "model.predict_forces",
    }
    results = run_search(SMOKE, energy, dry_run=True, run_root=tmp_path)
    assert len(results) == 48
    assert all(t.overrides["model.predict_forces"] is False for t in results)
    for combo in list(space.combinations())[:3]:
        assert load_config(SMOKE, combo).model.predict_forces is False
