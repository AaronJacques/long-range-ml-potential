"""Atomic energy network of the thesis in native Keras 3 (Sec. 4.3-4.5, Fig. 4.2-4.6).

The network maps the padded descriptors of one atom to its atomic energy (and, in
the experimental force mode, three force components). Per branch (local, Fig. 4.3,
and long range, Fig. 4.5) an embedding network turns the flattened neighbour list
into the embedding matrix G (N_max × K1). The descriptor of Eq. 4.8,

    D = G1ᵀ R̃ R̃ᵀ G2,    G1 = G,  G2 = G[:, :K2],

is a (K1 × K2) matrix. The flattened descriptors of both branches are concatenated
and fed into the fitting network (Fig. 4.4). The local-only model (Fig. 4.6) drops
the long-range branch.

The layer structure mirrors ``Neural-Network/Model.py@9413bfd``
(``res_embedding_model``, ``dense_res_block``, ``feature_matrix``, ``res_model``)
layer by layer, so that the weights of a legacy model can be taken over with
:func:`assign_legacy_weights`. Differences to the legacy code:

* No ``Lambda`` layers and no ``tf.*`` ops on symbolic tensors (bug B14); the
  column slice and Eq. 4.8 are the serialisable layers :class:`FirstColumn` and
  :class:`FeatureMatrix`.
* The extra ``Dense(K2)`` that the legacy code inserts when the local and the
  long-range K1/K2 are equal (``Model.py:151-156``) is gone (deviation A06, bug
  B16): the descriptors are always flattened and concatenated, as in Fig. 4.4.
* The dropout rate is configurable (``ModelConfig.dropout_rate``) instead of a
  hard-coded 0.2, and ``model(x, training=True)`` activates it (B02).

Notes
-----
Order dependence of the flattened embedding (deviation A10): the embedding network
does not act on single neighbours but on the whole flattened, ordered neighbour list
``s_i ⊕ Z_i`` (resp. ``1/r_i ⊕ F_i``), exactly as described on p. 45 and in Fig. 4.3.
The descriptor D is therefore only invariant under permutations that keep the row
order of the padded descriptor matrices; the invariance claim of p. 38 (taken from
DeepPot-SE) does not hold for this embedding. This is a property of the method and
is deliberately not changed here.

Padding rows of R̃ are zero, so they do not contribute to D, whatever G holds in
these rows (Eq. 4.8): no explicit mask is needed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import keras
import numpy as np
from keras import ops

from lrmlp.config import ModelConfig

__all__ = [
    "DenseResBlock",
    "FeatureMatrix",
    "FirstColumn",
    "assign_legacy_weights",
    "build_model",
    "layers_in_construction_order",
    "resolve_k1",
]

#: Branch prefixes of the layer names, in construction order (local first, Model.py:118-148).
BRANCHES = ("local", "long_range")
#: Layer classes that carry weights, i.e. the keys of ``weights_by_class``.
WEIGHTED_CLASSES = ("Dense", "LayerNormalization")

_RES_BLOCK_NAME = re.compile(r"^(?P<prefix>.+)_res_block_(?P<index>\d+)$")


# --------------------------------------------------------------------------- layers


@keras.saving.register_keras_serializable(package="lrmlp")
class FirstColumn(keras.layers.Layer):
    """First column of a descriptor matrix, ``x[:, :, 0]`` (s_i resp. 1/r_i, Fig. 4.3).

    Replaces ``Lambda(lambda x: x[:, :, 0])`` of the legacy code (``Model.py:33``).

    Input shape ``(batch, n_max, c)``, output shape ``(batch, n_max)``.
    """

    def call(self, inputs: Any) -> Any:
        """Return the first column of ``inputs``."""
        return inputs[:, :, 0]

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        """Drop the last axis."""
        return tuple(input_shape[:-1])


@keras.saving.register_keras_serializable(package="lrmlp")
class FeatureMatrix(keras.layers.Layer):
    """Descriptor D = Gᵀ R̃ R̃ᵀ G[:, :K2] of Eq. 4.8 (Fig. 4.3/4.5).

    Called as ``FeatureMatrix(k2)(r, g)`` with the padded descriptor matrix R̃ of
    shape ``(batch, n_max, 4)`` and the embedding matrix G of shape
    ``(batch, n_max, K1)``. The output D has shape ``(batch, K1, K2)``. Zero rows of
    R̃ (padding) do not contribute to D.

    Mirrors ``feature_matrix`` of the legacy code (``Model.py:10-15``):
    ``p1 = R̃ᵀ G2``, ``p2 = R̃ p1``, ``D = G1ᵀ p2``.

    Parameters
    ----------
    k2 : int
        Number of leading embedding columns G2 = G[:, :K2] (Tab. 5.2).
    **kwargs
        Passed to :class:`keras.layers.Layer`.
    """

    def __init__(self, k2: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if k2 <= 0:
            raise ValueError(f"k2 must be positive, got {k2}")
        self.k2 = int(k2)

    def call(self, r: Any, g: Any) -> Any:
        """Compute D = Gᵀ R̃ R̃ᵀ G[:, :K2]."""
        g2 = g[:, :, : self.k2]
        p1 = ops.matmul(ops.transpose(r, (0, 2, 1)), g2)  # (batch, 4, K2)
        p2 = ops.matmul(r, p1)  # (batch, n_max, K2)
        return ops.matmul(ops.transpose(g, (0, 2, 1)), p2)  # (batch, K1, K2)

    def compute_output_shape(
        self, r_shape: tuple[int | None, ...], g_shape: tuple[int | None, ...]
    ) -> tuple[int | None, ...]:
        """Return ``(batch, K1, K2)``."""
        return (g_shape[0], g_shape[-1], self.k2)

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration."""
        return {**super().get_config(), "k2": self.k2}


@keras.saving.register_keras_serializable(package="lrmlp")
class DenseResBlock(keras.layers.Layer):
    """Dense residual block of Fig. 4.2.

    ``LN → Dense(units, act) → Dense(units, act) → Dense(D, act) → Dropout → + input → LN``
    with D the width of the input, so the block returns its input width. Mirrors
    ``dense_res_block`` of the legacy code (``Model.py:18-28``); the sublayers are
    created in the same order (LayerNormalization, three Dense, LayerNormalization).

    Parameters
    ----------
    units : int
        Width of the two inner Dense layers.
    activation : str
        Activation of the three Dense layers (ELU in the thesis).
    dropout_rate : float
        Dropout rate before the residual connection (0.2 in Fig. 4.2).
    **kwargs
        Passed to :class:`keras.layers.Layer`.
    """

    def __init__(
        self, units: int, activation: str = "elu", dropout_rate: float = 0.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = activation
        self.dropout_rate = float(dropout_rate)
        self.norm_in = keras.layers.LayerNormalization(name=f"{self.name}_norm_in")
        self.dense_0 = keras.layers.Dense(units, activation=activation, name=f"{self.name}_dense_0")
        self.dense_1 = keras.layers.Dense(units, activation=activation, name=f"{self.name}_dense_1")
        self.dense_out: keras.layers.Dense | None = None
        self.dropout = keras.layers.Dropout(dropout_rate, name=f"{self.name}_dropout")
        self.norm_out = keras.layers.LayerNormalization(name=f"{self.name}_norm_out")

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        """Create the output projection to the input width D and build the sublayers."""
        width = input_shape[-1]
        if width is None:
            raise ValueError(f"{self.name}: the input width must be known")
        self.dense_out = keras.layers.Dense(
            width, activation=self.activation, name=f"{self.name}_dense_out"
        )
        inner = (*input_shape[:-1], self.units)
        self.norm_in.build(input_shape)
        self.dense_0.build(input_shape)
        self.dense_1.build(inner)
        self.dense_out.build(inner)
        self.norm_out.build(input_shape)

    @property
    def sublayers(self) -> list[keras.layers.Layer]:
        """All sublayers in construction order (as in ``Model.py:18-28``)."""
        layers: list[keras.layers.Layer] = [self.norm_in, self.dense_0, self.dense_1]
        if self.dense_out is not None:
            layers.append(self.dense_out)
        return [*layers, self.dropout, self.norm_out]

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Apply the residual block."""
        assert self.dense_out is not None
        r = self.norm_in(inputs)
        r = self.dense_0(r)
        r = self.dense_1(r)
        r = self.dense_out(r)
        r = self.dropout(r, training=training)
        return self.norm_out(r + inputs)

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        """Return the input shape (the block keeps its width)."""
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration."""
        return {
            **super().get_config(),
            "units": self.units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
        }


# --------------------------------------------------------------------------- building


def resolve_k1(value: int | Literal["auto"], n_max: int) -> int:
    """Resolve K1 of a descriptor (Tab. 5.2).

    Parameters
    ----------
    value : int or "auto"
        Configured K1; ``"auto"`` means K1 = N_max ("N_max = K1 always chosen", p. 52).
    n_max : int
        N_max of the branch (number of padded rows).

    Returns
    -------
    int
        The resolved K1.
    """
    if n_max <= 0:
        raise ValueError(f"n_max must be positive, got {n_max}")
    if value == "auto":
        return int(n_max)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"K1 must be a positive integer or 'auto', got {value!r}")
    return value


def _embedding_branch(
    prefix: str,
    matrix: Any,
    features: Any,
    n_max: int,
    k1: int,
    k2: int,
    cfg: ModelConfig,
) -> Any:
    """Embedding network and descriptor of one branch (Fig. 4.3, Eq. 4.8).

    Mirrors ``res_embedding_model`` + ``feature_matrix`` (``Model.py:10-51``) and
    returns the flattened descriptor D of shape ``(batch, K1·K2)``.
    """
    if k2 > k1:
        raise ValueError(f"{prefix}: K2 ({k2}) must not exceed K1 ({k1})")
    first_column = FirstColumn(name=f"{prefix}_first_column")(matrix)
    flat_features = keras.layers.Flatten(name=f"{prefix}_flatten_features")(features)
    x = keras.layers.Concatenate(name=f"{prefix}_embedding_input")([first_column, flat_features])
    x = keras.layers.Dense(
        cfg.embedding_dims[0], activation=cfg.activation, name=f"{prefix}_embedding_dense_0"
    )(x)
    for i, units in enumerate(cfg.embedding_dims[1:]):
        x = DenseResBlock(
            units, cfg.activation, cfg.dropout_rate, name=f"{prefix}_embedding_res_block_{i}"
        )(x)
    x = keras.layers.Dense(
        n_max * k1, activation=cfg.activation, name=f"{prefix}_embedding_projection"
    )(x)
    g = keras.layers.Reshape((n_max, k1), name=f"{prefix}_embedding_matrix")(x)
    d = FeatureMatrix(k2, name=f"{prefix}_feature_matrix")(matrix, g)
    return keras.layers.Flatten(name=f"{prefix}_flatten_descriptor")(d)


def build_model(
    model_cfg: ModelConfig,
    n_max_local: int,
    n_max_long_range: int,
    max_atomic_number: int = 100,
) -> keras.Model:
    """Build the atomic network of Fig. 4.5 (or Fig. 4.6 without long range).

    Parameters
    ----------
    model_cfg : ModelConfig
        Architecture (activation, embedding and fitting widths, dropout, K1/K2, modes).
    n_max_local : int
        N_max^loc, number of padded local rows per atom.
    n_max_long_range : int
        N_max^long, number of padded long-range rows per atom (unused without long range).
    max_atomic_number : int
        Largest atomic number m_max of the one-hot species counts; the long-range
        features have ``max_atomic_number + 1`` columns.

    Returns
    -------
    keras.Model
        Inputs (by name) ``local_matrix (n_loc, 4)``, ``local_numbers (n_loc, 2)`` and,
        with long range, ``long_range_matrix (n_lr, 4)``,
        ``long_range_features (n_lr, max_Z + 1)``. Output ``atomic_output`` of shape
        ``(atoms, 1)``, or ``(atoms, 4)`` (energy and force) with ``predict_forces``.
    """
    if n_max_local <= 0:
        raise ValueError(f"n_max_local must be positive, got {n_max_local}")
    if model_cfg.use_long_range and n_max_long_range <= 0:
        raise ValueError(f"n_max_long_range must be positive, got {n_max_long_range}")

    local_matrix = keras.Input((n_max_local, 4), name="local_matrix")
    local_numbers = keras.Input((n_max_local, 2), name="local_numbers")
    inputs = [local_matrix, local_numbers]
    descriptor = _embedding_branch(
        "local",
        local_matrix,
        local_numbers,
        n_max_local,
        resolve_k1(model_cfg.k1_local, n_max_local),
        model_cfg.k2_local,
        model_cfg,
    )
    if model_cfg.use_long_range:
        long_range_matrix = keras.Input((n_max_long_range, 4), name="long_range_matrix")
        long_range_features = keras.Input(
            (n_max_long_range, max_atomic_number + 1), name="long_range_features"
        )
        inputs += [long_range_matrix, long_range_features]
        long_range_descriptor = _embedding_branch(
            "long_range",
            long_range_matrix,
            long_range_features,
            n_max_long_range,
            resolve_k1(model_cfg.k1_long, n_max_long_range),
            model_cfg.k2_long,
            model_cfg,
        )
        descriptor = keras.layers.Concatenate(name="descriptor")(
            [descriptor, long_range_descriptor]
        )

    # Fitting network (Fig. 4.4, Model.py:54-59).
    x = keras.layers.Dense(
        model_cfg.fitting_width, activation=model_cfg.activation, name="fitting_dense"
    )(descriptor)
    x = keras.layers.Dropout(model_cfg.dropout_rate, name="fitting_dropout")(x)
    for i, units in enumerate(model_cfg.fitting_res_blocks):
        x = DenseResBlock(
            units, model_cfg.activation, model_cfg.dropout_rate, name=f"fitting_res_block_{i}"
        )(x)
    output = keras.layers.Dense(
        4 if model_cfg.predict_forces else 1, activation="linear", name="atomic_output"
    )(x)
    name = "long_range_model" if model_cfg.use_long_range else "local_model"
    return keras.Model(inputs=inputs, outputs=output, name=name)


# --------------------------------------------------------------------------- legacy weights


def _res_blocks(model: keras.Model, prefix: str) -> list[DenseResBlock]:
    blocks: dict[int, DenseResBlock] = {}
    for layer in model.layers:
        m = _RES_BLOCK_NAME.match(layer.name)
        if m and m.group("prefix") == prefix and isinstance(layer, DenseResBlock):
            blocks[int(m.group("index"))] = layer
    if sorted(blocks) != list(range(len(blocks))):
        raise ValueError(f"{prefix}: residual block indices are not contiguous: {sorted(blocks)}")
    return [blocks[i] for i in range(len(blocks))]


def _weighted(layers: Sequence[keras.layers.Layer]) -> list[keras.layers.Layer]:
    out: list[keras.layers.Layer] = []
    for layer in layers:
        if isinstance(layer, DenseResBlock):
            out += _weighted(layer.sublayers)
        elif isinstance(layer, keras.layers.Dense | keras.layers.LayerNormalization):
            out.append(layer)
    return out


def layers_in_construction_order(model: keras.Model) -> list[keras.layers.Layer]:
    """Weighted layers (Dense, LayerNormalization) in legacy construction order.

    The order is that of ``Model.py@9413bfd``: local embedding, long-range embedding,
    fitting network; inside each, the first Dense, the residual blocks (LN, three
    Dense, LN each) and the last Dense. Filtered by class, it equals the Keras 2
    name suffixes of the legacy layers (``dense``, ``dense_1``, … and
    ``layer_normalization``, …).

    Parameters
    ----------
    model : keras.Model
        A model built with :func:`build_model`.

    Returns
    -------
    list of keras.layers.Layer
        Dense and LayerNormalization layers, including those inside the residual blocks.
    """
    names = {layer.name for layer in model.layers}
    ordered: list[keras.layers.Layer] = []
    for prefix in BRANCHES:
        if f"{prefix}_embedding_dense_0" not in names:
            continue
        ordered.append(model.get_layer(f"{prefix}_embedding_dense_0"))
        ordered += _res_blocks(model, f"{prefix}_embedding")
        ordered.append(model.get_layer(f"{prefix}_embedding_projection"))
    ordered.append(model.get_layer("fitting_dense"))
    ordered += _res_blocks(model, "fitting")
    ordered.append(model.get_layer("atomic_output"))
    return _weighted(ordered)


def assign_legacy_weights(
    model: keras.Model, weights_by_class: Mapping[str, Sequence[Sequence[np.ndarray]]]
) -> None:
    """Take over the weights of a legacy model (``Model.py@9413bfd``).

    Parameters
    ----------
    model : keras.Model
        A model built with :func:`build_model` with the legacy architecture.
    weights_by_class : mapping
        ``{"Dense": [[kernel, bias], …], "LayerNormalization": [[gamma, beta], …]}``,
        per class in legacy construction order (Keras 2 name suffix ``_k``).

    Raises
    ------
    ValueError
        If classes, layer counts or weight shapes do not match; the message names the
        layer concerned. No weight is changed in that case.
    """
    unknown = set(weights_by_class) - set(WEIGHTED_CLASSES)
    if unknown:
        raise ValueError(f"unknown layer classes in weights_by_class: {sorted(unknown)}")
    by_class: dict[str, list[keras.layers.Layer]] = {cls: [] for cls in WEIGHTED_CLASSES}
    for layer in layers_in_construction_order(model):
        by_class[type(layer).__name__].append(layer)

    assignments: list[tuple[keras.layers.Layer, list[np.ndarray]]] = []
    for cls, layers in by_class.items():
        legacy = weights_by_class.get(cls, [])
        if len(legacy) != len(layers):
            raise ValueError(
                f"{cls}: the model has {len(layers)} layers, weights_by_class has {len(legacy)}"
            )
        for k, (layer, weights) in enumerate(zip(layers, legacy, strict=True)):
            current = layer.get_weights()
            new = [np.asarray(w) for w in weights]
            expected = [tuple(w.shape) for w in current]
            got = [tuple(w.shape) for w in new]
            if expected != got:
                raise ValueError(
                    f"layer {layer.name!r} ({cls} #{k}): expected weight shapes {expected}, "
                    f"got {got}"
                )
            assignments.append((layer, new))
    for layer, new in assignments:
        layer.set_weights(new)
