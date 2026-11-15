"""Keras 3 model with legacy weights reproduces the legacy outputs (golden model M-1)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lrmlp.config import ModelConfig
from lrmlp.model import assign_legacy_weights, build_model, layers_in_construction_order

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

pytestmark = pytest.mark.golden

VARIANTS = ["lr_energy", "local_energy", "lr_forces", "lr_energy_k1"]
INPUTS = ["local_matrix", "local_numbers", "long_range_matrix", "long_range_features"]
#: Energy network of M-1: Dense 128 + 2 residual blocks 128 (Model.py:54-59).
FITTING_RES_BLOCKS = (128, 128)
# float32 forward pass vs the legacy (tf_keras) outputs. The outputs are O(1-4); each
# passes through an embedding of width 32, a feature matrix that sums over up to
# N_max = 24 neighbour rows (fan-in ~ 24 * 32) and a 128-wide fitting net, so the
# result depends on the reduction order of the CPU kernels (float32 eps = 1.2e-7).
# Observed max |Δ|: 1.7e-5 on macOS arm64 and 2.2e-5 on a Linux x86_64 runner
# (lr_energy_k1, 1 of 105 elements). 2e-4 leaves a factor >= 10 on both, while
# swapping two weights of one Dense kernel changes the outputs by 0.05-0.2, i.e.
# >= 150 times the tolerance.
OUTPUT_RTOL = OUTPUT_ATOL = 2e-4
# Per-molecule sums over 21 atoms (x 4 columns for sum_all): the per-element errors
# partly add up; observed max 1.7e-5 on macOS.
SUM_ATOL = 1e-3


def _k1(value: int, n_max: int) -> int | str:
    # K1 = N_max is configured as "auto" to cover the resolution path against the golden.
    return "auto" if value == n_max else value


def _build(g: dict[str, Any], variant: str) -> Any:
    info = fx.load_model_manifest()["variants"][variant]
    hp = info["hyperparameters"]
    n_loc, n_lr = int(g["n_max_local"]), int(g["n_max_long_range"])
    cfg = ModelConfig(
        activation=info["constants"]["activation"],
        embedding_dims=tuple(hp["embedding_dims"]),
        fitting_width=128,
        fitting_res_blocks=FITTING_RES_BLOCKS,
        dropout_rate=0.2,
        k1_local=_k1(hp["M1_local"], n_loc),
        k1_long=_k1(hp["M1_long"], n_lr),
        k2_local=hp["M2_local"],
        k2_long=hp["M2_long"],
        use_long_range=bool(g["use_long_range"]),
        predict_forces=not bool(g["predict_only_energy"]),
    )
    return build_model(cfg, n_loc, n_lr)


@pytest.fixture(scope="module", params=VARIANTS)
def case(request: pytest.FixtureRequest) -> tuple[str, dict[str, Any], Any]:
    variant = request.param
    g = fx.load_model_golden(variant)
    model = _build(g, variant)
    return variant, g, model


def test_count_params(case: tuple[str, dict[str, Any], Any]) -> None:
    _, g, model = case
    assert model.count_params() == int(g["count_params"])


def test_weight_shapes_match_layer_table(case: tuple[str, dict[str, Any], Any]) -> None:
    _, g, model = case
    ours = [
        (type(layer).__name__, [list(w.shape) for w in layer.get_weights()])
        for layer in layers_in_construction_order(model)
    ]
    legacy = [(e["class"], e["weight_shapes"]) for e in g["layers"]]
    for cls in ("Dense", "LayerNormalization"):
        assert [s for c, s in ours if c == cls] == [s for c, s in legacy if c == cls], cls


def test_outputs_match_legacy(case: tuple[str, dict[str, Any], Any]) -> None:
    variant, g, model = case
    assign_legacy_weights(model, g["weights_by_class"])
    names = [n for n in INPUTS if f"x__{n}" in g]
    assert names == [t.name for t in model.inputs]
    x = {n: g[f"x__{n}"] for n in names}
    out = np.asarray(model(x, training=False))
    expected = g["output"]
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL, err_msg=variant)

    offsets = g["atom_offsets"]
    out64 = out.astype(np.float64)
    sum_energy = [out64[a:b, 0].sum() for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    sum_all = [out64[a:b].sum() for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    np.testing.assert_allclose(sum_energy, g["sum_energy"], rtol=0, atol=SUM_ATOL)
    np.testing.assert_allclose(sum_all, g["sum_all"], rtol=0, atol=SUM_ATOL)
    max_dev = float(np.max(np.abs(out.astype(np.float64) - expected)))
    print(f"{variant}: max |Δ| per atom = {max_dev:.3e}")


def test_shape_mismatch_raises_with_layer_name() -> None:
    g = fx.load_model_golden("lr_energy")
    model = _build(g, "lr_energy")
    weights = {cls: [list(ws) for ws in layers] for cls, layers in g["weights_by_class"].items()}
    weights["Dense"][3] = [np.zeros((5, 5), np.float32), np.zeros(5, np.float32)]
    before = [w.copy() for w in model.get_weights()]
    with pytest.raises(ValueError, match="local_embedding_res_block_0_dense_out"):
        assign_legacy_weights(model, weights)
    for a, b in zip(before, model.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)


def test_layer_count_mismatch_raises() -> None:
    g = fx.load_model_golden("local_energy")
    model = _build(g, "local_energy")
    weights = dict(g["weights_by_class"])
    weights["LayerNormalization"] = weights["LayerNormalization"][:-1]
    with pytest.raises(ValueError, match="LayerNormalization"):
        assign_legacy_weights(model, weights)
    with pytest.raises(ValueError, match="unknown layer classes"):
        assign_legacy_weights(model, {**g["weights_by_class"], "Conv1D": []})
