"""Inference API: energies (and forces) of arbitrary molecules from a trained run.

:class:`Potential` loads a run directory written by the trainer (plan §1.7):
the resolved configuration (``config.json``), the padding N_max^local /
N_max^long of the training dataset (``dataset.json``) and the weights of one
checkpoint. :meth:`Potential.predict` then

1. computes the descriptors of every frame with
   :func:`lrmlp.descriptors.compute_descriptors` (vectorised NumPy) using the
   grid configuration of the run, and pads them to the N_max of the run;
2. concatenates the atoms of up to ``batch_size`` molecules into one model call;
3. sums the atomic energies per molecule with :func:`tf.math.segment_sum`.

The model call is a single :class:`tf.function` with a fixed input signature
whose atom axis is ``None``: changing the number of molecules or atoms per call
does not retrace it.

The molecular energy is the sum of the atomic energies, E = Σ_i E_i (thesis
Eq. 4.1); the energy is always column 0 of the network output (bug B05 of the
legacy ``evaluate.py`` summed all outputs).

Notes
-----
Forces exist only for models trained with ``model.predict_forces = true``
(experimental, deviation A09). They are the direct three-component output of the
atomic network (columns 1-3), **not** the negative gradient −∇E of the energy.
Because the network sees rotation-invariant descriptors, these forces are not
rotation-equivariant. For energy-only models :attr:`Prediction.forces` is
``None``.

Units are those of the training data: kcal/mol for energies, kcal/(mol·Å) for
forces, Å for positions.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lrmlp.config import ExperimentConfig
from lrmlp.descriptors import compute_descriptors, max_neighbour_counts, pad_descriptors
from lrmlp.run import RunDirectory

if TYPE_CHECKING:
    import keras
    import tensorflow as tf

__all__ = ["Potential", "Prediction"]

#: Names of the model inputs (``lrmlp.model.build_model``).
LOCAL_KEYS = ("local_matrix", "local_numbers")
LONG_RANGE_KEYS = ("long_range_matrix", "long_range_features")
#: Extra input of :meth:`Potential.forward`: molecule index of every atom.
SEGMENT_KEY = "segment_ids"


@dataclass(frozen=True, slots=True, eq=False)
class Prediction:
    """Result of :meth:`Potential.predict`.

    Attributes
    ----------
    energies : ndarray of float64, shape (S,)
        Total energy of every frame, kcal/mol (computed in float32).
    forces : ndarray of float64, shape (S, N, 3), or None
        Forces in input atom order, kcal/(mol·Å); ``None`` for energy-only models.
        Direct network output, not −∇E (see the module docstring).
    """

    energies: npt.NDArray[np.float64]
    forces: npt.NDArray[np.float64] | None


def _read_n_max(metadata: Mapping[str, Any], source: Path) -> tuple[int, int]:
    try:
        return int(metadata["n_max_local"]), int(metadata["n_max_long_range"])
    except KeyError as err:
        raise ValueError(f"{source} has no {err.args[0]!r}; cannot determine the padding") from err


class Potential:
    """A trained atomic network ready for inference.

    Usually created with :meth:`from_run`.

    Parameters
    ----------
    model : keras.Model
        Atomic network built with :func:`lrmlp.model.build_model` for
        ``n_max_local`` / ``n_max_long_range``, weights loaded.
    config : ExperimentConfig
        Configuration of the run; ``config.grid`` defines the descriptors and
        ``config.model`` the inputs and outputs.
    n_max_local, n_max_long_range : int
        Padding N_max^local and N_max^long the model was built for.

    Attributes
    ----------
    model : keras.Model
        The atomic network.
    config : ExperimentConfig
        Run configuration.
    n_max_local, n_max_long_range : int
        Padding of the model inputs.
    epoch : int or None
        Epoch of the loaded checkpoint (set by :meth:`from_run`).
    """

    def __init__(
        self,
        model: keras.Model,
        config: ExperimentConfig,
        n_max_local: int,
        n_max_long_range: int,
    ) -> None:
        import tensorflow as tf

        self.model = model
        self.config = config
        self.n_max_local = operator.index(n_max_local)
        self.n_max_long_range = operator.index(n_max_long_range)
        self.epoch: int | None = None
        width = self.config.grid.max_atomic_number + 1
        specs: dict[str, tf.TensorSpec] = {
            "local_matrix": tf.TensorSpec((None, self.n_max_local, 4), tf.float32),
            "local_numbers": tf.TensorSpec((None, self.n_max_local, 2), tf.float32),
        }
        if self.use_long_range:
            specs["long_range_matrix"] = tf.TensorSpec((None, self.n_max_long_range, 4), tf.float32)
            specs["long_range_features"] = tf.TensorSpec(
                (None, self.n_max_long_range, width), tf.float32
            )
        specs[SEGMENT_KEY] = tf.TensorSpec((None,), tf.int32)
        self.input_signature = specs
        self._model_keys = tuple(k for k in specs if k != SEGMENT_KEY)
        self._forward = tf.function(self._forward_impl, input_signature=[specs])

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_run(cls, run_dir: str | Path, epoch: str | int = "best") -> Potential:
        """Load a trained model from a run directory.

        Parameters
        ----------
        run_dir : str or Path
            Run directory (layout in :mod:`lrmlp.run`).
        epoch : {"best", "last"} or int
            Checkpoint to load, resolved with :meth:`lrmlp.run.RunDirectory.resolve_epoch`.
            ``"best"`` is the epoch with the lowest validation energy MSE (p. 52).

        Returns
        -------
        Potential
            The model with the checkpoint weights; :attr:`epoch` is set.

        Raises
        ------
        FileNotFoundError
            If ``config.json``, ``dataset.json``, ``best.json`` or the checkpoint is
            missing.
        ValueError
            If ``dataset.json`` holds no N_max or ``epoch`` is invalid.
        """
        from lrmlp.model import build_model

        rd = RunDirectory(run_dir)
        config = rd.read_config()
        n_max_local, n_max_long_range = _read_n_max(rd.read_dataset_metadata(), rd.dataset_path)
        resolved = rd.resolve_epoch(epoch)
        model = build_model(
            config.model, n_max_local, n_max_long_range, config.grid.max_atomic_number
        )
        model.load_weights(rd.checkpoint_path(resolved))
        potential = cls(model, config, n_max_local, n_max_long_range)
        potential.epoch = resolved
        return potential

    # ------------------------------------------------------------------ properties

    @property
    def use_long_range(self) -> bool:
        """Whether the model has the long-range branch (Fig. 4.5) or not (Fig. 4.6)."""
        return bool(self.config.model.use_long_range)

    @property
    def predicts_forces(self) -> bool:
        """Whether the model has the experimental force output (A09)."""
        return bool(self.config.model.predict_forces)

    @property
    def tracing_count(self) -> int:
        """Number of times the model function has been traced (≤ 1 by design)."""
        return int(self._forward.experimental_get_tracing_count())

    # ------------------------------------------------------------------ model call

    def _forward_impl(self, inputs: Mapping[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        import tensorflow as tf

        x = {k: inputs[k] for k in self._model_keys}
        atomic = self.model(x, training=False)
        energies = tf.math.segment_sum(atomic[:, 0], inputs[SEGMENT_KEY])
        return energies, atomic

    def forward(self, inputs: Mapping[str, Any]) -> tuple[tf.Tensor, tf.Tensor]:
        """Run the model on the concatenated atoms of several molecules.

        Parameters
        ----------
        inputs : mapping
            Exactly the keys of :attr:`input_signature`: the model inputs (float32,
            atoms of all molecules concatenated, A rows) and ``segment_ids`` (A,)
            int32, the molecule index 0 … B−1 of every atom in ascending order and
            without gaps (as :func:`lrmlp.dataset.make_batched_tf_dataset` yields).

        Returns
        -------
        energies : tf.Tensor, shape (B,)
            Σ_i E_i per molecule (energy = output column 0, B05).
        atomic : tf.Tensor, shape (A, 1) or (A, 4)
            Raw network output per atom (energy, and force in columns 1-3).
        """
        result: tuple[tf.Tensor, tf.Tensor] = self._forward(dict(inputs))
        return result

    # ------------------------------------------------------------------ prediction

    def _check_numbers(self, numbers: npt.ArrayLike) -> npt.NDArray[np.int64]:
        z = np.asarray(numbers)
        if z.ndim != 1:
            raise ValueError(f"numbers must have shape (N,), got {z.shape}")
        if z.shape[0] == 0:
            raise ValueError("a molecule needs at least one atom")
        if not np.issubdtype(z.dtype, np.integer) and not np.all(np.mod(z, 1) == 0):
            raise ValueError("atomic numbers must be integers")
        z = z.astype(np.int64)
        max_z = self.config.grid.max_atomic_number
        bad = (z < 1) | (z > max_z)
        if bad.any():
            raise ValueError(
                f"atomic number {int(z[bad][0])} is outside 1 … max_atomic_number = {max_z}"
            )
        return z

    def descriptors(self, positions: npt.ArrayLike, numbers: npt.ArrayLike) -> dict[str, Any]:
        """Compute the padded model inputs of one molecule (in input atom order).

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Atom positions in Å.
        numbers : array_like of int, shape (N,)
            Atomic numbers.

        Returns
        -------
        dict of str to ndarray
            The model inputs of :func:`lrmlp.descriptors.pad_descriptors` for the
            run's grid and N_max (long-range keys only with long range).

        Raises
        ------
        ValueError
            If an atom has more local / long-range neighbours than the N_max the
            model was trained with, or the input is malformed.
        """
        z = self._check_numbers(numbers)
        desc = compute_descriptors(positions, z, self.config.grid)
        n_local, n_long = max_neighbour_counts(desc)
        if n_local > self.n_max_local:
            raise ValueError(
                f"the molecule has an atom with {n_local} local neighbours, but the model "
                f"was trained with n_max_local = {self.n_max_local}"
            )
        if self.use_long_range and n_long > self.n_max_long_range:
            raise ValueError(
                f"the molecule has an atom with {n_long} long-range cells, but the model "
                f"was trained with n_max_long_range = {self.n_max_long_range}"
            )
        padded = pad_descriptors(
            desc, self.n_max_local, self.n_max_long_range if self.use_long_range else n_long
        )
        return {k: padded[k] for k in self._model_keys}

    def predict(
        self,
        positions: npt.ArrayLike,
        numbers: npt.ArrayLike,
        batch_size: int = 256,
    ) -> Prediction:
        """Predict energies (and forces) of frames of one molecule.

        Parameters
        ----------
        positions : array_like, shape (S, N, 3) or (N, 3)
            Atom positions in Å; a single frame may be given as (N, 3).
        numbers : array_like of int, shape (N,)
            Atomic numbers, shared by all frames.
        batch_size : int
            Molecules per model call. The result does not depend on it (up to
            float32 rounding).

        Returns
        -------
        Prediction
            ``energies`` (S,) and, for force models, ``forces`` (S, N, 3) in input
            atom order. For an (N, 3) input S = 1.

        Raises
        ------
        ValueError
            If the shapes do not match, an atomic number is outside
            1 … ``grid.max_atomic_number``, an atom has more neighbours than the
            model's N_max, or two atoms coincide.
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 2:
            pos = pos[None]
        if pos.ndim != 3 or pos.shape[-1] != 3:
            raise ValueError(f"positions must have shape (S, N, 3) or (N, 3), got {pos.shape}")
        z = self._check_numbers(numbers)
        if pos.shape[1] != z.shape[0]:
            raise ValueError(f"positions have {pos.shape[1]} atoms, numbers have {z.shape[0]}")

        def frames() -> Iterator[dict[str, Any]]:
            for s in range(pos.shape[0]):
                try:
                    yield self.descriptors(pos[s], z)
                except ValueError as err:
                    raise ValueError(f"frame {s}: {err}") from err

        prediction = self._predict_padded(frames(), pos.shape[0], batch_size)
        forces = None
        if prediction.forces is not None:
            forces = prediction.forces.reshape(pos.shape)
        return Prediction(prediction.energies, forces)

    def predict_descriptors(
        self, molecules: Sequence[Mapping[str, Any]], batch_size: int = 256
    ) -> Prediction:
        """Predict from precomputed, padded descriptors.

        Parameters
        ----------
        molecules : sequence of mapping
            One mapping per molecule with the model inputs of shape (N_k, …), e.g.
            the output of :meth:`descriptors`, of
            :func:`lrmlp.descriptors.pad_descriptors` or of
            :meth:`lrmlp.dataset.DescriptorSplit.molecule` (extra keys are ignored).
            The padding must equal the model's N_max.
        batch_size : int
            Molecules per model call.

        Returns
        -------
        Prediction
            ``energies`` (S,); for force models ``forces`` (Σ N_k, 3), the atoms of
            all molecules concatenated in the given order.
        """
        return self._predict_padded(iter(molecules), len(molecules), batch_size)

    def _predict_padded(
        self, molecules: Iterator[Mapping[str, Any]], n: int, batch_size: int
    ) -> Prediction:
        batch_size = operator.index(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        energies: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        remaining = n
        while remaining > 0:
            chunk = [next(molecules) for _ in range(min(batch_size, remaining))]
            remaining -= len(chunk)
            e, atomic = self.forward(self._concatenate(chunk))
            energies.append(e.numpy())
            if self.predicts_forces:
                forces.append(atomic.numpy()[:, 1:4])
        e_all = np.concatenate(energies).astype(np.float64) if energies else np.zeros(0)
        f_all = None
        if self.predicts_forces:
            f_all = np.concatenate(forces).astype(np.float64) if forces else np.zeros((0, 3))
        return Prediction(e_all, f_all)

    def _concatenate(self, chunk: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key in self._model_keys:
            spec = self.input_signature[key]
            parts = [np.asarray(m[key], dtype=np.float32) for m in chunk]
            for k, part in enumerate(parts):
                if part.ndim != 3 or tuple(part.shape[1:]) != tuple(spec.shape[1:]):
                    raise ValueError(
                        f"molecule {k}: {key} has shape {part.shape}, expected "
                        f"(N, {', '.join(str(d) for d in spec.shape[1:])})"
                    )
            out[key] = np.concatenate(parts)
        counts = [int(np.shape(m["local_matrix"])[0]) for m in chunk]
        if min(counts) < 1:
            raise ValueError("a molecule needs at least one atom")
        out[SEGMENT_KEY] = np.repeat(np.arange(len(chunk), dtype=np.int32), counts)
        return out
