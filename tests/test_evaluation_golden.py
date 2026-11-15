"""Golden regression of ``lrmlp.evaluation`` against the legacy ``evaluate.py`` (model M-1).

The legacy test MAE (``eval_lr_energy.json`` of ``generate_model_golden.py``) was
computed with the weights of
``train_lr_energy.npz`` (one epoch on the ``golden_legacy`` profile) on the test split
of the same data. Here the dataset is rebuilt with ``golden_legacy.toml``, the weights
are written as checkpoint of epoch 0 into a run directory and :func:`evaluate` must
reproduce the MAE.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lrmlp.config import ExperimentConfig, load_config
from lrmlp.dataset import build_dataset, load_dataset
from lrmlp.evaluation import evaluate
from lrmlp.model import assign_legacy_weights, build_model
from lrmlp.potential import Potential
from lrmlp.run import RunDirectory

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

pytestmark = pytest.mark.golden

CONFIG = Path(__file__).resolve().parent / "fixtures" / "configs" / "golden_legacy.toml"


@pytest.fixture(scope="module")
def golden_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunDirectory, ExperimentConfig]:
    config = load_config(CONFIG)
    root = tmp_path_factory.mktemp("golden_eval")
    ds_path = build_dataset(config, output_dir=root / "golden_legacy", n_workers=1)
    ds = load_dataset(ds_path)
    run = RunDirectory(root / "run").create(config, {**ds.metadata, "path": str(ds_path)})
    model = build_model(
        config.model, ds.n_max_local, ds.n_max_long_range, config.grid.max_atomic_number
    )
    golden = fx.load_train_golden("lr_energy")
    assign_legacy_weights(model, golden["weights_by_class"])
    model.save_weights(run.checkpoint_path(0))  # legacy checkpoint model_epoch_0.h5
    run.write_best(0, {"val_energy_mse": float(golden["val_energy_loss"])})
    return run, config


def test_test_mae_matches_legacy(golden_run: tuple[RunDirectory, ExperimentConfig]) -> None:
    run, _ = golden_run
    expected = fx.load_eval_golden("lr_energy")
    result = evaluate(run.path, split="test")
    assert result.epoch == 0
    assert result.n_molecules == expected["test_size"]
    np.testing.assert_allclose(result.energy_mae, expected["mae"], rtol=1e-5)
    # The legacy test split is in ascending frame order, as ours.
    np.testing.assert_allclose(
        result.energy_abs_errors, expected["per_molecule_abs_error"], rtol=1e-4, atol=1e-5
    )
    print(
        f"test MAE {result.energy_mae:.7f} vs legacy {expected['mae']:.7f} "
        f"(rel. dev. {abs(result.energy_mae / expected['mae'] - 1):.2e})"
    )


def test_batched_equals_unbatched(golden_run: tuple[RunDirectory, ExperimentConfig]) -> None:
    run, _ = golden_run
    one = evaluate(run.path, split="test", batch_size=1)
    many = evaluate(run.path, split="test", batch_size=256)
    # Errors are differences of float32 energies of a few hundred kcal/mol, and the
    # batch shape changes the kernel blocking: observed max |Δ| 1.9e-6 on macOS arm64.
    # atol 1e-4 (~3 float32 ulps at 300 kcal/mol) keeps a factor > 10 for other CPUs;
    # mixing up molecules between batches would change the errors by O(1).
    np.testing.assert_allclose(one.energy_abs_errors, many.energy_abs_errors, rtol=0, atol=1e-4)


def test_force_model_energy_is_column_0(
    golden_run: tuple[RunDirectory, ExperimentConfig],
) -> None:
    """B05 on the legacy force model: E = Σ output[:, 0] (``sum_energy``), not ``sum_all``."""
    _, config = golden_run
    g: dict[str, Any] = fx.load_model_golden("lr_forces")
    model_cfg = dataclasses.replace(config.model, predict_forces=True)
    n_loc, n_lr = int(g["n_max_local"]), int(g["n_max_long_range"])
    model = build_model(model_cfg, n_loc, n_lr, config.grid.max_atomic_number)
    assign_legacy_weights(model, g["weights_by_class"])
    pot = Potential(model, dataclasses.replace(config, model=model_cfg), n_loc, n_lr)

    offsets = g["atom_offsets"]
    keys = ("local_matrix", "local_numbers", "long_range_matrix", "long_range_features")
    molecules = [
        {k: g[f"x__{k}"][a:b] for k in keys} for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    pred = pot.predict_descriptors(molecules)
    np.testing.assert_allclose(pred.energies, g["sum_energy"], rtol=1e-5, atol=1e-4)
    assert np.min(np.abs(pred.energies - g["sum_all"])) > 1e-2
    assert pred.forces is not None
    # Same float32 model outputs as test_model_golden.py (OUTPUT_RTOL/ATOL there):
    # observed max |Δ| 5.3e-6 on macOS arm64, up to 2.2e-5 on Linux for the outputs.
    np.testing.assert_allclose(pred.forces, g["output"][:, 1:4], rtol=2e-4, atol=2e-4)
