"""Keras 3 model with legacy weights reproduces the legacy outputs (model M-1, T03 golden)."""

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
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5, err_msg=variant)

    offsets = g["atom_offsets"]
    out64 = out.astype(np.float64)
    sum_energy = [out64[a:b, 0].sum() for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    sum_all = [out64[a:b].sum() for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    np.testing.assert_allclose(sum_energy, g["sum_energy"], rtol=0, atol=1e-4)
    np.testing.assert_allclose(sum_all, g["sum_all"], rtol=0, atol=1e-4)
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
