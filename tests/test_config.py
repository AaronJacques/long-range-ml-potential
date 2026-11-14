"""Tests for the typed TOML configuration (``lrmlp.config``)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from lrmlp.config import (
    DataConfig,
    ExperimentConfig,
    GridConfig,
    ModelConfig,
    TrainConfig,
    load_config,
    parse_overrides,
)

# Explicit list, no glob: other tasks add non-preset files to configs/.
PRESETS = ["paracetamol", "aspirin", "ac-ala3-nhme", "stachyose", "legacy-head"]


@pytest.fixture(scope="module")
def configs_dir(repo_root: Path) -> Path:
    return repo_root / "configs"


@pytest.fixture(scope="module")
def test_configs_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "configs"


def write_toml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL = """
name = "mini"
[data]
sources = ["raw/a.npz"]
"""


# --------------------------------------------------------------------------- presets


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_loads(configs_dir: Path, preset: str) -> None:
    config = load_config(configs_dir / f"{preset}.toml")
    assert config.name == preset
    assert config.data.sources == (configs_dir.parent / "data" / "raw" / _raw_name(preset),)
    assert config.data.output_dir == configs_dir.parent / "data" / "processed" / preset
    assert config.train.run_root == configs_dir.parent / "runs"


def _raw_name(preset: str) -> str:
    return "aspirin.npz" if preset == "legacy-head" else f"{preset}.npz"


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_has_header_comment(configs_dir: Path, preset: str) -> None:
    text = (configs_dir / f"{preset}.toml").read_text(encoding="utf-8")
    assert text.startswith("#")
    assert "docs/methodology.md" in text
    assert "sGDML dataset [16] of the thesis; place the file here" in text


# Tab. 5.1/5.2 of the thesis: (Δg, l_max, K1 loc/long, K2 loc/long, η, s, epochs, samples)
THESIS_TABLE: dict[str, tuple[Any, ...]] = {
    "paracetamol": (1.0, 6, "auto", "auto", 4, 1, 5e-5, 2.08e5, 200, 65000),
    "aspirin": (1.0, 6, "auto", "auto", 4, 1, 1e-4, 2.08e5, 200, 65000),
    "ac-ala3-nhme": (2.0, 2, "auto", "auto", 4, 2, 5e-5, 2.08e5, 150, 65000),
    "stachyose": (2.0, 4, 30, 6, 4, 2, 5e-5, 8.727e4, 130, "all"),
}


@pytest.mark.parametrize("preset", sorted(THESIS_TABLE))
def test_preset_matches_thesis_tables(configs_dir: Path, preset: str) -> None:
    c = load_config(configs_dir / f"{preset}.toml")
    actual = (
        c.grid.grid_size,
        c.grid.max_local_level,
        c.model.k1_local,
        c.model.k1_long,
        c.model.k2_local,
        c.model.k2_long,
        c.train.learning_rate,
        c.train.lr_decay_steps,
        c.train.epochs,
        c.data.n_samples_per_source,
    )
    assert actual == THESIS_TABLE[preset]
    assert c.grid.assignment == "thesis"
    assert c.model.dropout_rate == 0.2
    assert c.model.embedding_dims == (64, 128, 128, 256)
    assert c.model.fitting_res_blocks == (128, 128, 128)
    assert (c.data.val_fraction, c.data.test_fraction) == (0.1, 0.1)
    assert c.train.lr_decay_rate == 0.97
    assert c.train.lr_staircase is True


@pytest.mark.parametrize(
    ("preset", "cutoff"),
    [
        ("paracetamol", 7.707),
        ("aspirin", 7.707),
        ("ac-ala3-nhme", 7.414),
        ("stachyose", 11.414),
        ("legacy-head", 7.707),
    ],
)
def test_cutoff_derived_from_grid(configs_dir: Path, preset: str, cutoff: float) -> None:
    grid = load_config(configs_dir / f"{preset}.toml").grid
    assert round(grid.cutoff, 3) == cutoff
    assert grid.cutoff == (grid.max_local_level + 1 + 0.5**0.5) * grid.grid_size
    assert grid.inner_cutoff == pytest.approx(grid.cutoff - 0.2)
    assert grid.inner_cutoff == grid.cutoff - grid.inner_cutoff_offset


def test_cutoff_is_exact_legacy_expression() -> None:
    # Constants.py:34/36 with GRID_SIZE = 1, MAX_LOCAL_LEVEL = 6.
    legacy_cutoff = (6 + 1 + 0.5**0.5) * 1
    grid = GridConfig(grid_size=1, max_local_level=6)
    assert grid.cutoff == legacy_cutoff
    assert grid.inner_cutoff == legacy_cutoff - 0.2


def test_legacy_head_matches_constants_at_9413bfd(configs_dir: Path) -> None:
    c = load_config(configs_dir / "legacy-head.toml")
    assert (c.grid.grid_size, c.grid.max_local_level, c.grid.assignment) == (1.0, 6, "legacy")
    assert c.grid.max_atomic_number == 100
    assert c.grid.sort_neighbours is False
    m = c.model
    assert (m.k1_local, m.k2_local, m.k1_long, m.k2_long) == (30, 4, 6, 2)
    assert m.embedding_dims == (64, 128, 128, 128)
    assert (m.fitting_width, m.fitting_res_blocks) == (128, (128, 128))
    assert m.dropout_rate == 0.0
    assert m.activation == "elu"
    assert (m.use_long_range, m.predict_forces) == (True, False)
    t = c.train
    assert (t.learning_rate, t.lr_decay_steps, t.lr_decay_rate, t.lr_staircase) == (
        5e-5,
        8.727e4,
        0.97,
        True,
    )
    assert t.epochs == 200
    assert (t.p_energy_start, t.p_energy_limit, t.p_force_start, t.p_force_limit) == (1, 1, 1, 0)
    assert (t.loss_weight_decay_steps, t.loss_weight_decay_rate) == (7.5e3, 0.97)
    assert t.shuffle_buffer == 1000


# --------------------------------------------------------------------------- test configs


def test_golden_legacy_matches_model_m1(test_configs_dir: Path, fixtures_dir: Path) -> None:
    c = load_config(test_configs_dir / "golden_legacy.toml")
    g = c.grid
    assert (g.grid_size, g.max_local_level, g.assignment, g.sort_neighbours) == (
        1.0,
        2,
        "legacy",
        False,
    )
    m = c.model
    assert m.embedding_dims == (16, 32, 32)
    assert (m.fitting_width, m.fitting_res_blocks, m.dropout_rate) == (128, (128, 128), 0.0)
    assert (m.k1_local, m.k1_long, m.k2_local, m.k2_long) == ("auto", "auto", 4, 1)
    assert (m.activation, m.use_long_range, m.predict_forces) == ("elu", True, False)
    t = c.train
    assert (t.learning_rate, t.lr_decay_steps, t.lr_decay_rate, t.lr_staircase) == (
        1e-3,
        20,
        0.97,
        True,
    )
    assert (t.epochs, t.shuffle_buffer) == (1, 1)
    assert (t.p_energy_start, t.p_energy_limit, t.p_force_start, t.p_force_limit) == (1, 1, 1, 0)
    assert (t.loss_weight_decay_steps, t.loss_weight_decay_rate) == (7.5e3, 0.97)
    d = c.data
    assert d.sources == (fixtures_dir / "synthetic_small.npz",)
    assert d.n_samples_per_source == "all"
    assert (d.val_fraction, d.test_fraction, d.split_seed) == (0.1, 0.1, 42)


def test_smoke_config(test_configs_dir: Path, fixtures_dir: Path, repo_root: Path) -> None:
    c = load_config(test_configs_dir / "smoke.toml")
    assert c.data.sources == (fixtures_dir / "synthetic_small.npz",)
    assert (c.grid.grid_size, c.grid.max_local_level, c.grid.assignment) == (1.0, 2, "thesis")
    m = c.model
    assert (m.embedding_dims, m.fitting_width, m.fitting_res_blocks) == ((16, 32), 32, (32,))
    assert (m.dropout_rate, m.k1_local, m.k1_long, m.k2_local, m.k2_long) == (
        0.2,
        "auto",
        "auto",
        4,
        1,
    )
    t = c.train
    assert (t.epochs, t.learning_rate, t.lr_decay_steps, t.shuffle_buffer) == (2, 1e-3, 20, 16)
    assert (t.seed, t.log_every_steps) == (42, 10)
    assert c.data.output_dir == repo_root / "data" / "processed" / "smoke"
    assert c.train.run_root == repo_root / "runs"


# --------------------------------------------------------------------------- loading


def test_defaults_follow_plan(tmp_path: Path) -> None:
    c = load_config(write_toml(tmp_path / "c.toml", MINIMAL))
    assert c.grid == GridConfig()
    assert c.model == ModelConfig()
    assert dataclasses.replace(c.train, run_root=Path("runs")) == TrainConfig()
    assert c.data.n_samples_per_source == 65000
    assert (c.data.val_fraction, c.data.test_fraction, c.data.split_seed) == (0.1, 0.1, 42)
    assert c.data.n_workers == 0
    assert c.data.output_dir == tmp_path / "data" / "processed" / "mini"
    assert c.train.run_root == tmp_path / "runs"


@pytest.mark.parametrize(
    "text",
    [
        MINIMAL + "\nunknown = 1\n",
        MINIMAL + "[grid]\ngridsize = 1.0\n",
        MINIMAL + "[model]\nk1 = 3\n",
        MINIMAL + "[extra]\na = 1\n",
        MINIMAL + "[derived]\ncutoff = 1.0\n",
    ],
)
def test_unknown_key_raises(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError, match=r"unknown|derived"):
        load_config(write_toml(tmp_path / "c.toml", text))


def test_missing_sources_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sources"):
        load_config(write_toml(tmp_path / "c.toml", 'name = "x"\n[data]\n'))


def test_relative_paths_resolved_to_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sub = tmp_path / "cfg"
    sub.mkdir()
    path = write_toml(
        sub / "c.toml",
        'name = "x"\n[data]\nsources = ["../raw/a.npz", "/abs/b.npz"]\n'
        'output_dir = "out"\n[train]\nrun_root = "../runs"\n',
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    c = load_config(Path("..") / "cfg" / "c.toml")
    assert c.data.sources == (tmp_path / "raw" / "a.npz", Path("/abs/b.npz"))
    assert c.data.output_dir == sub / "out"
    assert c.train.run_root == tmp_path / "runs"
    assert c == load_config(path)


def test_relative_override_paths_resolved_to_config_dir(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", MINIMAL)
    c = load_config(path, ["data.output_dir=elsewhere", 'train.run_root="r"'])
    assert c.data.output_dir == tmp_path / "elsewhere"
    assert c.train.run_root == tmp_path / "r"


# --------------------------------------------------------------------------- overrides


def test_overrides_parse_toml_literals() -> None:
    parsed = parse_overrides(
        [
            "train.epochs=2",
            "train.learning_rate=1e-3",
            "model.use_long_range=false",
            "model.embedding_dims=[8, 16]",
            "model.activation=relu",
            'model.k1_local="auto"',
            "name = spaced",
        ]
    )
    assert parsed == {
        "train.epochs": 2,
        "train.learning_rate": 1e-3,
        "model.use_long_range": False,
        "model.embedding_dims": [8, 16],
        "model.activation": "relu",
        "model.k1_local": "auto",
        "name": "spaced",
    }


def test_load_config_with_string_overrides(test_configs_dir: Path) -> None:
    c = load_config(
        test_configs_dir / "smoke.toml",
        ["train.epochs=5", "model.k1_local=12", "grid.grid_size=2", "model.embedding_dims=[8]"],
    )
    assert c.train.epochs == 5
    assert c.model.k1_local == 12
    assert c.grid.grid_size == 2.0 and isinstance(c.grid.grid_size, float)
    assert c.model.embedding_dims == (8,)


def test_load_config_with_mapping_overrides(test_configs_dir: Path) -> None:
    c = load_config(test_configs_dir / "smoke.toml", {"train.seed": 7, "name": "other"})
    assert (c.train.seed, c.name) == (7, "other")


def test_with_overrides_returns_new_config(test_configs_dir: Path) -> None:
    c = load_config(test_configs_dir / "smoke.toml")
    d = c.with_overrides(["train.epochs=3", "grid.assignment=legacy"])
    assert (d.train.epochs, d.grid.assignment) == (3, "legacy")
    assert (c.train.epochs, c.grid.assignment) == (2, "thesis")
    assert d.data == c.data


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("train.epochs=two", "train.epochs"),
        ("train.epochs=2.5", "train.epochs"),
        ("model.use_long_range=1", "model.use_long_range"),
        ("model.k1_local=automatic", "model.k1_local"),
        ("train.unknown=1", "unknown override key"),
        ("other.epochs=1", "section"),
        ("epochs", "key=value"),
        ("=3", "key=value"),
    ],
)
def test_bad_overrides_raise(test_configs_dir: Path, override: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_config(test_configs_dir / "smoke.toml", [override])


# --------------------------------------------------------------------------- roundtrip


@pytest.mark.parametrize("preset", PRESETS)
def test_roundtrip_to_dict_from_dict(configs_dir: Path, preset: str) -> None:
    c = load_config(configs_dir / f"{preset}.toml")
    d = c.to_dict()
    assert ExperimentConfig.from_dict(d) == c
    # Also through JSON, as stored in a run directory.
    assert ExperimentConfig.from_dict(json.loads(json.dumps(d))) == c
    assert d["derived"] == {"cutoff": c.grid.cutoff, "inner_cutoff": c.grid.inner_cutoff}
    assert set(d) == {"name", "grid", "data", "model", "train", "derived"}


def test_from_dict_rejects_inconsistent_derived(test_configs_dir: Path) -> None:
    d = load_config(test_configs_dir / "smoke.toml").to_dict()
    d["derived"]["cutoff"] = 1.0
    with pytest.raises(ValueError, match="derived"):
        ExperimentConfig.from_dict(d)


def test_config_is_frozen(test_configs_dir: Path) -> None:
    c = load_config(test_configs_dir / "smoke.toml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.train.epochs = 3  # type: ignore[misc]


# --------------------------------------------------------------------------- validation

SOURCES = (Path("a.npz"),)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: GridConfig(grid_size=0), "grid.grid_size"),
        (lambda: GridConfig(max_local_level=0), "grid.max_local_level"),
        (lambda: GridConfig(max_atomic_number=-1), "grid.max_atomic_number"),
        (lambda: GridConfig(assignment="other"), "grid.assignment"),  # type: ignore[arg-type]
        (lambda: GridConfig(inner_cutoff_offset=50.0), "inner_cutoff_offset"),
        (lambda: DataConfig(sources=()), "data.sources"),
        (lambda: DataConfig(SOURCES, n_samples_per_source=0), "n_samples_per_source"),
        (lambda: DataConfig(SOURCES, val_fraction=0.5, test_fraction=0.5), "smaller than 1"),
        (lambda: DataConfig(SOURCES, val_fraction=-0.1), "data.val_fraction"),
        (lambda: DataConfig(SOURCES, n_workers=-1), "data.n_workers"),
        (lambda: ModelConfig(dropout_rate=1.0), "dropout_rate"),
        (lambda: ModelConfig(dropout_rate=-0.1), "dropout_rate"),
        (lambda: ModelConfig(embedding_dims=()), "embedding_dims"),
        (lambda: ModelConfig(embedding_dims=(16, 0)), r"embedding_dims\[1\]"),
        (lambda: ModelConfig(k1_local=0), "k1_local"),
        (lambda: ModelConfig(k1_long="all"), "k1_long"),  # type: ignore[arg-type]
        (lambda: ModelConfig(k2_long=0), "k2_long"),
        (lambda: ModelConfig(fitting_width=0), "fitting_width"),
        (lambda: ModelConfig(activation="tanh"), "activation"),  # type: ignore[arg-type]
        (lambda: TrainConfig(epochs=0), "train.epochs"),
        (lambda: TrainConfig(learning_rate=0.0), "train.learning_rate"),
        (lambda: TrainConfig(checkpoints="some"), "train.checkpoints"),  # type: ignore[arg-type]
        (lambda: TrainConfig(steps_per_execution=0), "steps_per_execution"),
        (lambda: TrainConfig(shuffle_buffer=0), "shuffle_buffer"),
        (lambda: TrainConfig(p_force_limit=-1.0), "p_force_limit"),
        (lambda: ExperimentConfig("", GridConfig(), DataConfig(SOURCES)), "name"),
        (lambda: ExperimentConfig("a/b", GridConfig(), DataConfig(SOURCES)), "name"),
    ],
)
def test_validation_errors(factory: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_validation_error_in_file_names_the_file(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "bad.toml", MINIMAL + "[model]\ndropout_rate = 1.5\n")
    with pytest.raises(ValueError, match=r"bad\.toml.*dropout_rate"):
        load_config(path)


def test_type_errors_in_file(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", MINIMAL + '[train]\nepochs = "200"\n')
    with pytest.raises(ValueError, match="train.epochs"):
        load_config(path)


def test_default_output_dir_uses_name() -> None:
    c = ExperimentConfig("abc", GridConfig(), DataConfig(SOURCES))
    assert c.data.output_dir == Path("data/processed/abc")


def test_config_module_does_not_import_tensorflow(repo_root: Path) -> None:
    code = (
        "import sys; from lrmlp.config import load_config; import lrmlp.run; "
        "load_config('configs/stachyose.toml'); sys.exit('tensorflow' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=False)
    assert result.returncode == 0
