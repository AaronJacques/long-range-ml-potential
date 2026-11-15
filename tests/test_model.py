"""Unit tests of the Keras 3 model (lrmlp.model)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import keras
import numpy as np
import pytest

from lrmlp.config import ModelConfig, load_config
from lrmlp.model import (
    DenseResBlock,
    FeatureMatrix,
    FirstColumn,
    build_model,
    layers_in_construction_order,
    resolve_k1,
)

PRESETS = ["paracetamol", "aspirin", "ac-ala3-nhme", "stachyose", "legacy-head"]
N_LOC, N_LR, MAX_Z = 7, 5, 100

SMALL = ModelConfig(
    embedding_dims=(8, 16, 16),
    fitting_width=16,
    fitting_res_blocks=(16, 16),
    dropout_rate=0.2,
    k1_local="auto",
    k1_long="auto",
    k2_local=3,
    k2_long=1,
)


def _preset(repo_root: Path, name: str) -> ModelConfig:
    return load_config(repo_root / "configs" / f"{name}.toml").model


def _inputs(n_atoms: int, n_loc: int, n_lr: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "local_matrix": rng.normal(size=(n_atoms, n_loc, 4)).astype(np.float32),
        "local_numbers": rng.integers(1, 9, size=(n_atoms, n_loc, 2)).astype(np.float32),
        "long_range_matrix": rng.normal(size=(n_atoms, n_lr, 4)).astype(np.float32),
        "long_range_features": rng.integers(0, 3, size=(n_atoms, n_lr, MAX_Z + 1)).astype(
            np.float32
        ),
    }


def _all_layers(layers: list[keras.layers.Layer]) -> list[keras.layers.Layer]:
    out: list[keras.layers.Layer] = []
    for layer in layers:
        out.append(layer)
        if isinstance(layer, DenseResBlock):
            out += _all_layers(layer.sublayers)
    return out


# --------------------------------------------------------------------------- building


@pytest.mark.parametrize("name", PRESETS)
def test_build_presets(repo_root: Path, name: str) -> None:
    cfg = _preset(repo_root, name)
    model = build_model(cfg, N_LOC, N_LR)
    assert [t.name for t in model.inputs] == [
        "local_matrix",
        "local_numbers",
        "long_range_matrix",
        "long_range_features",
    ]
    assert tuple(model.inputs[3].shape) == (None, N_LR, MAX_Z + 1)
    assert model.output.shape == (None, 1)
    assert model.layers[-1].name == "atomic_output"
    out = model(_inputs(3, N_LOC, N_LR), training=False)
    assert out.shape == (3, 1)
    assert np.all(np.isfinite(np.asarray(out)))


def _res_block_units(model: keras.Model, prefix: str) -> list[int]:
    return [
        layer.units
        for layer in model.layers
        if isinstance(layer, DenseResBlock) and layer.name.startswith(prefix)
    ]


def test_thesis_vs_head_structure(repo_root: Path) -> None:
    thesis = build_model(_preset(repo_root, "aspirin"), N_LOC, N_LR)
    head = build_model(_preset(repo_root, "legacy-head"), N_LOC, N_LR)
    for prefix in ("local_embedding", "long_range_embedding"):
        # Fig. 4.3: 64 -> res blocks 128, 128, 256 -> N_max·K1 (HEAD: 128, 128, 128).
        assert _res_block_units(thesis, prefix) == [128, 128, 256]
        assert _res_block_units(head, prefix) == [128, 128, 128]
        for model in (thesis, head):
            assert model.get_layer(f"{prefix}_dense_0").units == 64
    # Fig. 4.4: 3 res blocks 128 (HEAD: 2).
    assert _res_block_units(thesis, "fitting") == [128, 128, 128]
    assert _res_block_units(head, "fitting") == [128, 128]
    assert thesis.get_layer("fitting_dense").units == head.get_layer("fitting_dense").units == 128
    # Each residual block returns its input width (Fig. 4.2).
    for layer in thesis.layers:
        if isinstance(layer, DenseResBlock):
            assert layer.dense_out is not None
            assert layer.dense_out.units == layer.input.shape[-1] == layer.output.shape[-1]
    n_dense = sum(isinstance(la, keras.layers.Dense) for la in layers_in_construction_order(head))
    # 2 branches × (2 + 3·3) + fitting (2 + 2·3)
    assert n_dense == 2 * (2 + 3 * 3) + 2 + 2 * 3


def test_local_only_model() -> None:
    cfg = dataclasses.replace(SMALL, use_long_range=False)
    model = build_model(cfg, N_LOC, N_LR)
    assert [t.name for t in model.inputs] == ["local_matrix", "local_numbers"]
    assert not any(layer.name.startswith("long_range") for layer in model.layers)
    x = _inputs(2, N_LOC, N_LR)
    out = model({k: x[k] for k in ("local_matrix", "local_numbers")}, training=False)
    assert out.shape == (2, 1)


# --------------------------------------------------------------------------- layers


def test_feature_matrix_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    batch, n, k1, k2 = 3, 6, 5, 2
    r = rng.normal(size=(batch, n, 4))
    g = rng.normal(size=(batch, n, k1))
    layer = FeatureMatrix(k2, dtype="float64")
    d = np.asarray(layer(r, g))
    expected = np.einsum("bnk,bnc,bmc,bml->bkl", g, r, r, g[:, :, :k2])
    assert d.shape == (batch, k1, k2)
    np.testing.assert_allclose(d, expected, rtol=1e-6)
    for i in range(batch):
        ref = g[i].T @ r[i] @ r[i].T @ g[i][:, :k2]
        np.testing.assert_allclose(d[i], ref, rtol=1e-6)


def test_feature_matrix_padding_rows_are_masked() -> None:
    rng = np.random.default_rng(2)
    n, n_valid, k1, k2 = 8, 5, 6, 4
    r = rng.normal(size=(2, n, 4)).astype(np.float32)
    r[:, n_valid:] = 0.0
    g = rng.normal(size=(2, n, k1)).astype(np.float32)
    g_noise = g.copy()
    g_noise[:, n_valid:] = rng.normal(scale=100.0, size=(2, n - n_valid, k1))
    layer = FeatureMatrix(k2)
    d = np.asarray(layer(r, g))
    d_noise = np.asarray(layer(r, g_noise))
    np.testing.assert_allclose(d_noise, d, rtol=1e-6, atol=0)
    # The truncated problem without padding rows gives the same D.
    np.testing.assert_allclose(
        np.asarray(layer(r[:, :n_valid], g[:, :n_valid])), d, rtol=1e-5, atol=1e-5
    )


def test_feature_matrix_rejects_k2_above_k1() -> None:
    with pytest.raises(ValueError, match="K2"):
        build_model(dataclasses.replace(SMALL, k1_local=2, k2_local=3), N_LOC, N_LR)
    with pytest.raises(ValueError, match="k2"):
        FeatureMatrix(0)


def test_first_column() -> None:
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.testing.assert_array_equal(np.asarray(FirstColumn()(x)), x[:, :, 0])


def test_custom_layers_serialisable() -> None:
    for layer in (
        FirstColumn(name="fc"),
        FeatureMatrix(3, name="fm"),
        DenseResBlock(7, "relu", 0.1),
    ):
        config = keras.saving.serialize_keras_object(layer)
        clone = keras.saving.deserialize_keras_object(config)
        assert type(clone) is type(layer)
        assert clone.get_config() == layer.get_config()
        assert config["module"] == "lrmlp" or config["registered_name"].startswith("lrmlp>")


# --------------------------------------------------------------------------- K1


def test_resolve_k1() -> None:
    assert resolve_k1("auto", 17) == 17
    assert resolve_k1(5, 17) == 5
    for bad in (0, -1, "max", True):
        with pytest.raises(ValueError):
            resolve_k1(bad, 17)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        resolve_k1("auto", 0)


def test_k1_auto_equals_n_max() -> None:
    auto = build_model(SMALL, N_LOC, N_LR)
    explicit = build_model(dataclasses.replace(SMALL, k1_local=N_LOC, k1_long=N_LR), N_LOC, N_LR)
    assert auto.get_layer("local_embedding_matrix").output.shape == (None, N_LOC, N_LOC)
    assert auto.get_layer("long_range_embedding_matrix").output.shape == (None, N_LR, N_LR)
    assert auto.get_layer("local_feature_matrix").output.shape == (None, N_LOC, 3)
    assert auto.count_params() == explicit.count_params()
    other = build_model(dataclasses.replace(SMALL, k1_local=4, k1_long=2), N_LOC, N_LR)
    assert other.get_layer("local_embedding_matrix").output.shape == (None, N_LOC, 4)
    assert other.get_layer("descriptor").output.shape == (None, 4 * 3 + 2 * 1)


def test_equal_k_uses_flatten_concat() -> None:
    # B16/A06: equal K1/K2 locally and long range -> still Flatten + Concatenate.
    cfg = dataclasses.replace(SMALL, k1_local=4, k1_long=4, k2_local=2, k2_long=2)
    model = build_model(cfg, N_LOC, N_LR)
    concat = model.get_layer("descriptor")
    assert isinstance(concat, keras.layers.Concatenate)
    assert concat.output.shape == (None, 2 * 4 * 2)
    fitting = model.get_layer("fitting_dense")
    assert fitting.input is concat.output
    for prefix in ("local", "long_range"):
        flat = model.get_layer(f"{prefix}_flatten_descriptor")
        assert isinstance(flat, keras.layers.Flatten)
        assert isinstance(model.get_layer(f"{prefix}_feature_matrix"), FeatureMatrix)
    dense = [la for la in layers_in_construction_order(model) if isinstance(la, keras.layers.Dense)]
    # 2 branches × (2 + 2·3) + fitting (2 + 2·3): no extra Dense(K2).
    assert len(dense) == 2 * (2 + 2 * 3) + 2 + 2 * 3


def test_no_lambda_layers() -> None:
    for cfg in (SMALL, dataclasses.replace(SMALL, use_long_range=False, predict_forces=True)):
        model = build_model(cfg, N_LOC, N_LR)
        layers = _all_layers(model.layers)
        assert not any(isinstance(layer, keras.layers.Lambda) for layer in layers)
        assert not any("lambda" in type(layer).__name__.lower() for layer in layers)
        assert all(type(layer).__module__.startswith(("keras", "lrmlp")) for layer in layers)


def test_layer_names_deterministic() -> None:
    a = [layer.name for layer in _all_layers(build_model(SMALL, N_LOC, N_LR).layers)]
    b = [layer.name for layer in _all_layers(build_model(SMALL, N_LOC, N_LR).layers)]
    assert a == b
    assert len(set(a)) == len(a)
    assert "local_embedding_dense_0" in a
    assert "fitting_res_block_1" in a
    assert "fitting_res_block_1_dense_out" in a


def test_layers_in_construction_order() -> None:
    model = build_model(SMALL, N_LOC, N_LR)
    names = [layer.name for layer in layers_in_construction_order(model)]
    block = ["norm_in", "dense_0", "dense_1", "dense_out", "norm_out"]
    expected: list[str] = []
    for prefix in ("local", "long_range"):
        expected.append(f"{prefix}_embedding_dense_0")
        for i in range(2):
            expected += [f"{prefix}_embedding_res_block_{i}_{s}" for s in block]
        expected.append(f"{prefix}_embedding_projection")
    expected.append("fitting_dense")
    for i in range(2):
        expected += [f"fitting_res_block_{i}_{s}" for s in block]
    expected.append("atomic_output")
    assert names == expected
    weighted = {id(w) for layer in layers_in_construction_order(model) for w in layer.weights}
    assert weighted == {id(w) for w in model.weights}


# ------------------------------------------------------------------- weights, dropout, forces


def test_weights_roundtrip(tmp_path: Path) -> None:
    keras.utils.set_random_seed(1)
    a = build_model(SMALL, N_LOC, N_LR)
    keras.utils.set_random_seed(2)
    b = build_model(SMALL, N_LOC, N_LR)
    x = _inputs(4, N_LOC, N_LR)
    assert not np.allclose(np.asarray(a(x)), np.asarray(b(x)))
    path = tmp_path / "model.weights.h5"
    a.save_weights(path)
    b.load_weights(path)
    np.testing.assert_array_equal(
        np.asarray(a(x, training=False)), np.asarray(b(x, training=False))
    )
    for wa, wb in zip(a.get_weights(), b.get_weights(), strict=True):
        np.testing.assert_array_equal(wa, wb)


def test_training_flag_controls_dropout() -> None:
    keras.utils.set_random_seed(3)
    model = build_model(dataclasses.replace(SMALL, dropout_rate=0.5), N_LOC, N_LR)
    x = _inputs(6, N_LOC, N_LR)
    inference = [np.asarray(model(x, training=False)) for _ in range(2)]
    # Two eager calls run the same kernels: bit-identical.
    np.testing.assert_array_equal(inference[0], inference[1])
    # predict() runs a compiled graph (other kernel fusion), so only float32-close:
    # observed max |Δ| 9e-7 (rel 2e-6) on a Linux x86_64 runner, 0 on macOS arm64.
    # Dropout 0.5 (below) moves the outputs by > 1e5 times this tolerance.
    np.testing.assert_allclose(
        np.asarray(model.predict(x, verbose=0)), inference[0], rtol=1e-5, atol=1e-5
    )
    train_a = np.asarray(model(x, training=True))
    train_b = np.asarray(model(x, training=True))
    assert not np.allclose(train_a, inference[0])
    assert not np.allclose(train_a, train_b)


def test_zero_dropout_is_deterministic_in_training() -> None:
    model = build_model(dataclasses.replace(SMALL, dropout_rate=0.0), N_LOC, N_LR)
    x = _inputs(3, N_LOC, N_LR)
    np.testing.assert_allclose(
        np.asarray(model(x, training=True)), np.asarray(model(x, training=False)), rtol=1e-6
    )


@pytest.mark.parametrize("use_long_range", [True, False])
def test_predict_forces_output(use_long_range: bool) -> None:
    cfg = dataclasses.replace(SMALL, predict_forces=True, use_long_range=use_long_range)
    model = build_model(cfg, N_LOC, N_LR)
    assert model.output.shape == (None, 4)
    x = _inputs(5, N_LOC, N_LR)
    if not use_long_range:
        x = {k: x[k] for k in ("local_matrix", "local_numbers")}
    assert model(x, training=False).shape == (5, 4)


def test_invalid_n_max() -> None:
    with pytest.raises(ValueError, match="n_max_local"):
        build_model(SMALL, 0, N_LR)
    with pytest.raises(ValueError, match="n_max_long_range"):
        build_model(SMALL, N_LOC, 0)
    # Without long range N_max^long is not used.
    build_model(dataclasses.replace(SMALL, use_long_range=False), N_LOC, 0)
