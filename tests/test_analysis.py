"""Tests of the figures in ``lrmlp.analysis`` (legacy defects B17, B18)."""

from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from lrmlp import cli
from lrmlp.analysis.plots import (
    LEGACY_COLUMNS,
    energy_title,
    plot_training_log,
    read_training_log,
)
from lrmlp.config import ExperimentConfig, load_config
from lrmlp.run import RunDirectory
from lrmlp.training import log_columns

REPO = Path(__file__).resolve().parent.parent
SMOKE = REPO / "tests" / "fixtures" / "configs" / "smoke.toml"
N_EPOCHS = 6


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _new_log(path: Path, forces: bool) -> Path:
    header = list(log_columns(forces))
    rows = []
    for epoch in range(1, N_EPOCHS + 1):
        values = {
            "epoch": epoch,
            "train_loss": 10.0 / epoch + (1.0 if forces else 0.0),
            "train_energy_mse": 10.0 / epoch,
            "val_loss": 12.0 / epoch,
            "val_energy_mse": 12.0 / epoch,
            "learning_rate": 1e-3,
            "seconds": 1.5,
            "train_force_mse": 3.0 / epoch,
            "val_force_mse": 4.0 / epoch,
            "p_energy": 1.0,
            "p_force": 0.5,
        }
        rows.append([values[h] for h in header])
    return _write_csv(path, header, rows)


def _legacy_log(path: Path) -> Path:
    """Legacy header of ``Constants.Logging`` with an all-NaN validation force column (B07)."""
    header = list(LEGACY_COLUMNS)
    rows = [
        [e, 20.0 / e, 10.0 / e, 3.0 / e, 25.0 / e, 12.0 / e, "nan", 0.1 * e, 100.0 / e]
        for e in range(1, N_EPOCHS + 1)
    ]
    return _write_csv(path, header, rows)


def _smoke_config(use_long_range: bool = True) -> ExperimentConfig:
    config = load_config(SMOKE)
    return dataclasses.replace(
        config, model=dataclasses.replace(config.model, use_long_range=use_long_range)
    )


# --------------------------------------------------------------------------- logs


def test_reads_new_header_and_skip(tmp_path: Path) -> None:
    log = read_training_log(_new_log(tmp_path / "training_log.csv", forces=False), skip=2)
    assert not log.legacy
    assert len(log) == N_EPOCHS - 2
    np.testing.assert_array_equal(log["epoch"], [3, 4, 5, 6])
    np.testing.assert_allclose(log["val_energy_mse"], [12 / 3, 12 / 4, 12 / 5, 12 / 6])
    assert "train_force_mse" not in log


def test_reads_legacy_header(tmp_path: Path) -> None:
    log = read_training_log(_legacy_log(tmp_path / "training_log.csv"), skip=1)
    assert log.legacy
    np.testing.assert_array_equal(log["epoch"], [2, 3, 4, 5, 6])
    np.testing.assert_allclose(log["train_energy_mse"], [10 / e for e in range(2, 7)])
    np.testing.assert_allclose(log["val_energy_mse"], [12 / e for e in range(2, 7)])
    np.testing.assert_allclose(log["p_force"], [100 / e for e in range(2, 7)])
    assert "val_force_mse" not in log  # NaN-only column dropped
    assert "train_force_mse" in log


def test_skip_beyond_log_rejected(tmp_path: Path) -> None:
    path = _new_log(tmp_path / "training_log.csv", forces=False)
    with pytest.raises(ValueError, match="no epochs left"):
        read_training_log(path, skip=N_EPOCHS)
    with pytest.raises(ValueError, match="skip"):
        read_training_log(path, skip=-1)


def test_energy_only_run_writes_energy_figure(tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run").create(_smoke_config(True), {})
    _new_log(run.log_path, forces=False)
    paths = plot_training_log(run.path)
    assert paths == [run.path / "plots" / "smoke_long_range_energy_mse.pdf"]
    assert all(p.is_file() and p.read_bytes().startswith(b"%PDF") for p in paths)


def test_force_run_writes_all_figures(tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run").create(_smoke_config(False), {})
    _new_log(run.log_path, forces=True)
    paths = plot_training_log(run.path, tmp_path / "out", skip=1)
    names = [p.name for p in paths]
    assert names == [
        f"smoke_no_long_range_{n}.pdf"
        for n in ("energy_mse", "total_loss", "force_mse", "loss_weights")
    ]
    assert all(p.parent == tmp_path / "out" and p.is_file() for p in paths)


def test_legacy_csv_without_config(tmp_path: Path) -> None:
    path = _legacy_log(tmp_path / "legacy" / "training_log.csv")
    paths = plot_training_log(path, skip=1)
    assert [p.name for p in paths] == [
        "energy_mse.pdf",
        "total_loss.pdf",
        "force_mse.pdf",
        "loss_weights.pdf",
    ]
    assert all(p.parent == path.parent / "plots" and p.is_file() for p in paths)


@pytest.mark.parametrize("use_long_range", [True, False])
def test_title_reflects_long_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_long_range: bool
) -> None:
    import lrmlp.analysis.plots as plots

    titles: list[str | None] = []
    original = plots._curve_figure

    def spy(log, series, ylabel, title, path, **kw):  # type: ignore[no-untyped-def]
        titles.append(title)
        return original(log, series, ylabel, title, path, **kw)

    monkeypatch.setattr(plots, "_curve_figure", spy)
    run = RunDirectory(tmp_path / "run").create(_smoke_config(use_long_range), {})
    _new_log(run.log_path, forces=False)
    plot_training_log(run.path)
    expected = "with Long-Range" if use_long_range else "without Long-Range"
    assert titles[0] == f"Energy Loss smoke {expected}"
    plot_training_log(run.path, title="Custom")
    assert titles[-1] == "Custom"


def test_energy_title() -> None:
    assert energy_title("Aspirin", True) == "Energy Loss Aspirin with Long-Range"
    assert energy_title("Aspirin", False) == "Energy Loss Aspirin without Long-Range"
    assert energy_title(None, None) == "Energy Loss"


def test_pyplot_not_used(tmp_path: Path) -> None:
    import matplotlib

    backend = matplotlib.get_backend()
    plot_training_log(_new_log(tmp_path / "training_log.csv", forces=False))
    assert matplotlib.get_backend() == backend
    if "matplotlib.pyplot" in sys.modules:
        assert not sys.modules["matplotlib.pyplot"].get_fignums()


# --------------------------------------------------------------------------- weights


def _naive_histogram(
    values: list[float], others: list[list[float]], bins: int, boundary: tuple[float, float]
) -> list[int]:
    """Reference: Python loops as in the legacy filter, edges over all kept values."""
    kept_all = [w for vs in others for w in vs if boundary[0] < w < boundary[1]]
    lo, hi = min(kept_all), max(kept_all)
    width = (hi - lo) / bins
    counts = [0] * bins
    for w in values:
        if not boundary[0] < w < boundary[1]:
            continue
        i = min(int((w - lo) / width), bins - 1)
        counts[i] += 1
    return counts


def test_weight_histogram_counts() -> None:
    from lrmlp.analysis.weights import flatten_weights, weight_histograms

    rng = np.random.default_rng(0)
    a = flatten_weights([rng.normal(0, 0.6, (7, 5)), rng.normal(0, 0.3, (11,))])
    b = flatten_weights([rng.normal(0.1, 0.5, (4, 3, 2))])
    assert a.shape == (46,) and b.shape == (24,)
    a = np.append(a, [1.0, -1.0, 3.0])  # boundary values are excluded (strict)
    counts, edges = weight_histograms([a, b], bins=8, boundary=(-1.0, 1.0), density=False)
    assert edges.shape == (9,)
    for values, got in zip([a, b], counts, strict=True):
        ref = _naive_histogram(values.tolist(), [a.tolist(), b.tolist()], 8, (-1.0, 1.0))
        np.testing.assert_array_equal(got, ref)
    dens, _ = weight_histograms([a, b], bins=8, boundary=(-1.0, 1.0))
    for d in dens:
        assert np.sum(d * np.diff(edges)) == pytest.approx(1.0)


def test_flatten_weights_order() -> None:
    from lrmlp.analysis.weights import flatten_weights

    arrays = [np.arange(6).reshape(2, 3), np.array([10.0, 11.0])]
    np.testing.assert_array_equal(flatten_weights(arrays), [0, 1, 2, 3, 4, 5, 10, 11])
    assert flatten_weights([]).shape == (0,)


def _mini_run(path: Path, use_long_range: bool, seed: int) -> RunDirectory:
    """Run directory with one randomly initialised checkpoint (epoch 2 = best)."""
    import keras

    from lrmlp.model import build_model

    config = _smoke_config(use_long_range)
    run = RunDirectory(path).create(config, {"n_max_local": 6, "n_max_long_range": 4})
    keras.utils.set_random_seed(seed)
    model = build_model(config.model, 6, 4, config.grid.max_atomic_number)
    model.save_weights(run.checkpoint_path(2))
    run.write_best(2, {"val_energy_mse": 1.0})
    return run


@pytest.fixture(scope="module")
def mini_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunDirectory, RunDirectory]:
    root = tmp_path_factory.mktemp("weights")
    return _mini_run(root / "with", True, 1), _mini_run(root / "without", False, 2)


def test_load_run_weights(mini_runs: tuple[RunDirectory, RunDirectory]) -> None:
    from lrmlp.analysis.weights import load_run_weights
    from lrmlp.model import build_model

    with_lr, without_lr = mini_runs
    w = load_run_weights(with_lr.path)
    config = with_lr.read_config()
    model = build_model(config.model, 6, 4, config.grid.max_atomic_number)
    assert w.size == sum(int(np.prod(v.shape)) for v in model.weights)
    model.load_weights(with_lr.checkpoint_path(2))
    np.testing.assert_array_equal(
        w, np.concatenate([np.asarray(v).ravel() for v in model.get_weights()])
    )
    assert load_run_weights(without_lr.path, 2).size < w.size  # no long-range branch


def test_plot_weight_distribution(
    mini_runs: tuple[RunDirectory, RunDirectory], tmp_path: Path
) -> None:
    from lrmlp.analysis.weights import plot_weight_distribution

    with_lr, without_lr = mini_runs
    out = plot_weight_distribution(with_lr.path, without_lr.path, tmp_path / "fig" / "w.pdf")
    assert out == tmp_path / "fig" / "w.pdf"
    assert out.read_bytes().startswith(b"%PDF")
    out2 = plot_weight_distribution(
        with_lr.path, without_lr.path, tmp_path / "w2.pdf", "last", 2, boundary=None, bins=20
    )
    assert out2.is_file()


# --------------------------------------------------------------------------- CLI


def test_cli_plot_log(tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run").create(_smoke_config(True), {})
    _new_log(run.log_path, forces=False)
    assert cli.main(["plot-log", "--run", str(run.path), "--skip", "1"]) == 0
    assert (run.path / "plots" / "smoke_long_range_energy_mse.pdf").is_file()
    legacy = _legacy_log(tmp_path / "legacy.csv")
    assert cli.main(["plot-log", "--log", str(legacy), "--output", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "energy_mse.pdf").is_file()


def test_cli_plot_weights(mini_runs: tuple[RunDirectory, RunDirectory], tmp_path: Path) -> None:
    with_lr, without_lr = mini_runs
    out = tmp_path / "w.pdf"
    argv = ["plot-weights", "--run-a", str(with_lr.path), "--run-b", str(without_lr.path)]
    assert cli.main([*argv, "--output", str(out), "--epoch-a", "2"]) == 0
    assert out.is_file()
