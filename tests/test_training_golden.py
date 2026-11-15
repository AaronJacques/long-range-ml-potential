"""Golden regression of one training epoch against the legacy ``Training.start_training``.

The golden fixture (``train_lr_energy.npz`` of ``generate_model_golden.py``) holds the
weights of model M-1
after one epoch of the unchanged legacy code on dataset G-B (``golden_legacy.toml``,
no shuffling, start weights ``model_lr_energy.npz``). The legacy step never
activated dropout (B02), so the profile uses ``dropout_rate = 0``; the energy loss
is invariant under the atom order, so the fixed row order (B01) does not matter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lrmlp.config import load_config
from lrmlp.dataset import build_dataset
from lrmlp.model import layers_in_construction_order
from lrmlp.training import Trainer, TrainingResult

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

pytestmark = pytest.mark.golden

CONFIG = Path(__file__).resolve().parent / "fixtures" / "configs" / "golden_legacy.toml"
# One float32 epoch (48 steps) vs the legacy run; the reduction order of the CPU
# kernels differs between platforms and accumulates over the steps. Observed on macOS
# arm64: weights max |Δ| 5.9e-7 (factor 6 below the former atol 1e-6), losses rel.
# 1.7e-5 (factor 6 below the former rtol 1e-4). The values below leave a factor
# ~29 on both. Real changes are still caught: learning rate +1 % gives weights
# 170 times the tolerance and losses rel. 3e-2-0.15; one decay step earlier
# (lr_decay_steps 19) gives weights 5 times the tolerance and losses rel. 2e-3-8e-3.
RTOL, ATOL = 1e-4, 1e-5  # weights
LOSS_RTOL = 5e-4


@pytest.fixture(scope="module")
def trained(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Trainer, TrainingResult, dict[str, Any]]:
    config = load_config(CONFIG)
    assert config.model.dropout_rate == 0.0
    assert config.train.shuffle_buffer == 1
    assert config.train.epochs == 1
    golden = fx.load_train_golden("lr_energy")
    root = tmp_path_factory.mktemp("golden_train")
    dataset = build_dataset(config, output_dir=root / "ds", n_workers=1)
    trainer = Trainer(config, dataset, root / "run", golden["start_weights_by_class"])
    return trainer, trainer.fit(), golden


def _errors(trainer: Trainer, golden: dict[str, Any]) -> list[tuple[str, float, float]]:
    """(layer/weight, max abs error, max allowed) per weight array."""
    by_class: dict[str, list[Any]] = {}
    for layer in layers_in_construction_order(trainer.model):
        by_class.setdefault(type(layer).__name__, []).append(layer)
    assert sorted(by_class) == sorted(golden["weights_by_class"])
    rows = []
    for cls, layers in by_class.items():
        expected_layers = golden["weights_by_class"][cls]
        assert len(layers) == len(expected_layers), cls
        for k, (layer, expected) in enumerate(zip(layers, expected_layers, strict=True)):
            for j, (got, ref) in enumerate(zip(layer.get_weights(), expected, strict=True)):
                assert got.shape == ref.shape, (layer.name, j)
                err = np.abs(got - ref)
                excess = float(np.max(err - (ATOL + RTOL * np.abs(ref))))
                rows.append((f"{cls}#{k} {layer.name}[{j}]", float(err.max()), excess))
    return rows


def test_weights_after_one_epoch(trained: tuple[Trainer, TrainingResult, dict[str, Any]]) -> None:
    trainer, _, golden = trained
    rows = _errors(trainer, golden)
    failing = [r for r in rows if r[2] > 0]
    report = "\n".join(f"{name}: max abs error {err:.3e}" for name, err, _ in rows)
    assert not failing, f"weights differ from the golden (rtol={RTOL}, atol={ATOL}):\n{report}"


def test_epoch_losses(trained: tuple[Trainer, TrainingResult, dict[str, Any]]) -> None:
    _, result, golden = trained
    row = result.history[0]
    np.testing.assert_allclose(
        row["train_energy_mse"], float(golden["energy_loss"]), rtol=LOSS_RTOL
    )
    np.testing.assert_allclose(row["train_loss"], float(golden["energy_loss"]), rtol=LOSS_RTOL)
    np.testing.assert_allclose(
        row["val_energy_mse"], float(golden["val_energy_loss"]), rtol=LOSS_RTOL
    )
    np.testing.assert_allclose(
        result.best_val_energy_mse, float(golden["final_loss"]), rtol=LOSS_RTOL
    )
    assert result.best_epoch == 1


def test_step_count(trained: tuple[Trainer, TrainingResult, dict[str, Any]]) -> None:
    trainer, result, golden = trained
    assert result.steps == int(golden["steps"]) == golden["info"]["steps"]
    assert int(trainer.optimizer.iterations) == result.steps
