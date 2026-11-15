"""Evaluation of a trained run on a dataset split (replaces ``Neural-Network/evaluate.py``).

:func:`evaluate` reproduces the test metric of the thesis (Tab. 5.3, p. 53): the
checkpoint of the epoch with the lowest validation energy MSE (``best.json``,
deviation A12) is evaluated on the test split, and the mean absolute error of the
molecular energies is reported,

    MAE = 1/S Σ_k |Σ_i E_i^(k) − E^(k)|,

exactly the per-molecule formula of the legacy ``evaluate.py:23-43`` (float32
per-molecule errors, then the mean). In addition the RMSE and MSE of the energies
and, for force models (experimental, A09), the MAE and RMSE over all force
components are computed.

Differences to the legacy script:

* The molecular energy is the sum of output column 0 only. The legacy code summed
  **all** outputs, i.e. energy plus the three force columns of a force model (B05).
* Model, input signature and padding come from the run directory (``config.json``,
  ``dataset.json``), not from ``Constants`` (B05, B08), and no paths are hard-coded
  (B10).
* Several molecules are evaluated per model call (one :class:`tf.function` with a
  fixed signature, see :class:`lrmlp.potential.Potential`); the split is read in
  stored order without shuffling (B19).
* The errors are computed with explicit TensorFlow ops instead of
  ``keras.losses`` on scalars, which break under Keras 3 (B15).

Before evaluating, the dataset is checked against the copy of its metadata
that the trainer stored in the run's ``dataset.json`` (grid incl. ``assignment``
and ``sort_neighbours``, sources and frame counts, split seed and fractions,
N_max, species; see :data:`CHECKED_DATASET_KEYS`). A different dataset, e.g. one
built with another grid or split seed (whose test split may contain training
frames of the run), raises a :class:`ValueError` naming the differing fields.
``allow_dataset_mismatch=True`` (CLI ``--allow-dataset-mismatch``) evaluates it
anyway, logs a warning and records the difference in the result.

The result is also written to ``evaluation_<split>_<epoch>.json`` in the run
directory.
"""

from __future__ import annotations

import logging
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lrmlp.potential import Potential
from lrmlp.run import RunDirectory, write_json

__all__ = ["EvaluationResult", "dataset_mismatches", "evaluate", "resolve_dataset_dir"]

logger = logging.getLogger(__name__)

#: Keys of ``dataset.json`` that may hold the dataset directory, in lookup order.
DATASET_PATH_KEYS = ("path", "dataset_dir", "dataset_path")
#: Dataset metadata (``metadata.json``) that must equal the run's ``dataset.json``.
#: Not compared: ``created``, ``name``, ``lrmlp_version``, ``format_version``,
#: ``units``, ``arrays``, the dataset ``path``, the source paths (files may move;
#: their content is compared via ``sha256``) and ``n_samples_per_source`` (only an
#: upper bound; the frames actually used are compared via ``sources[*].n_frames_used``
#: and ``frame_offset``, so "all" and a cap above the frame count are equal).
CHECKED_DATASET_KEYS = (
    "grid",
    "sources",
    "n_samples",
    "split_seed",
    "val_fraction",
    "test_fraction",
    "splits",
    "species",
    "max_atomic_number",
    "n_max_local",
    "n_max_long_range",
)
#: Keys of every ``sources`` entry that are not compared.
_IGNORED_SOURCE_KEYS = ("path",)
_MISSING = "<missing>"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics of one checkpoint on one split.

    Attributes
    ----------
    energy_mae : float
        Mean absolute error of the molecular energies, kcal/mol (Tab. 5.3).
    energy_rmse : float
        Root mean squared error of the molecular energies, kcal/mol.
    energy_mse : float
        Mean squared error of the molecular energies, (kcal/mol)².
    force_mae : float or None
        Mean absolute error over all force components, kcal/(mol·Å); ``None`` for
        energy-only models.
    force_rmse : float or None
        Root mean squared error over all force components; ``None`` for energy-only
        models.
    n_molecules : int
        Number of evaluated molecules.
    epoch : int
        Evaluated checkpoint epoch.
    split : str
        Evaluated split.
    energy_abs_errors : ndarray of float32, shape (n_molecules,)
        |Σ_i E_i − E| per molecule in stored order (not written to the JSON file).
    dataset_mismatch : tuple of str
        Differences between the evaluated dataset and the run's ``dataset.json``
        (only non-empty with ``allow_dataset_mismatch=True``). If non-empty,
        :meth:`to_dict` adds ``"dataset_matches_run": false`` and the list as
        ``"dataset_mismatch"``.
    """

    energy_mae: float
    energy_rmse: float
    energy_mse: float
    force_mae: float | None
    force_rmse: float | None
    n_molecules: int
    epoch: int
    split: str
    energy_abs_errors: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32), repr=False, compare=False
    )
    dataset_mismatch: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the scalar metrics as plain JSON types."""
        out: dict[str, Any] = {
            "energy_mae": self.energy_mae,
            "energy_rmse": self.energy_rmse,
            "energy_mse": self.energy_mse,
            "force_mae": self.force_mae,
            "force_rmse": self.force_rmse,
            "n_molecules": self.n_molecules,
            "epoch": self.epoch,
            "split": self.split,
        }
        if self.dataset_mismatch:
            out["dataset_matches_run"] = False
            out["dataset_mismatch"] = list(self.dataset_mismatch)
        return out


def resolve_dataset_dir(run: RunDirectory, dataset_dir: str | Path | None = None) -> Path:
    """Return the dataset directory of a run.

    Parameters
    ----------
    run : RunDirectory
        The run.
    dataset_dir : str or Path, optional
        Explicit dataset directory; takes precedence over ``dataset.json``.

    Returns
    -------
    Path
        ``dataset_dir`` or the path stored in ``dataset.json`` (key ``path``,
        relative paths are taken relative to the run directory).

    Raises
    ------
    ValueError
        If no ``dataset_dir`` is given and ``dataset.json`` holds no path.
    """
    if dataset_dir is not None:
        return Path(dataset_dir)
    metadata = run.read_dataset_metadata()
    for key in DATASET_PATH_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else run.path / path
    raise ValueError(
        f"{run.dataset_path} holds no dataset path (keys {DATASET_PATH_KEYS}); "
        "pass dataset_dir explicitly"
    )


def _diff(run_value: Any, value: Any, key: str) -> list[str]:
    if isinstance(run_value, Mapping) and isinstance(value, Mapping):
        out = []
        for k in sorted(set(run_value) | set(value)):
            out += _diff(run_value.get(k, _MISSING), value.get(k, _MISSING), f"{key}.{k}")
        return out
    if (
        isinstance(run_value, list)
        and isinstance(value, list)
        and len(run_value) == len(value)
        and any(isinstance(v, Mapping) for v in run_value)
    ):
        out = []
        for i, (a, b) in enumerate(zip(run_value, value, strict=True)):
            out += _diff(a, b, f"{key}[{i}]")
        return out
    if run_value != value:
        return [f"{key} (run: {run_value!r}, dataset: {value!r})"]
    return []


def _strip_source_paths(sources: Any) -> Any:
    if not isinstance(sources, list):
        return sources
    return [
        {k: v for k, v in s.items() if k not in _IGNORED_SOURCE_KEYS}
        if isinstance(s, Mapping)
        else s
        for s in sources
    ]


def dataset_mismatches(
    run_metadata: Mapping[str, Any], dataset_metadata: Mapping[str, Any]
) -> list[str]:
    """Differences between a run's ``dataset.json`` and a dataset's metadata.

    Parameters
    ----------
    run_metadata : mapping
        ``dataset.json`` of the run (the dataset metadata at training time).
    dataset_metadata : mapping
        ``metadata.json`` of the dataset to evaluate.

    Returns
    -------
    list of str
        One entry ``"<field> (run: <value>, dataset: <value>)"`` per differing
        field of :data:`CHECKED_DATASET_KEYS` (nested fields as ``grid.assignment``
        or ``sources[0].sha256``, a missing field as ``'<missing>'``); empty if the
        dataset is the one the run was trained on.
    """
    out = []
    for key in CHECKED_DATASET_KEYS:
        run_value = run_metadata.get(key, _MISSING)
        value = dataset_metadata.get(key, _MISSING)
        if key == "sources":
            run_value, value = _strip_source_paths(run_value), _strip_source_paths(value)
        out += _diff(run_value, value, key)
    return out


def _check_dataset(
    run: RunDirectory, dataset: Any, path: Path, allow_mismatch: bool
) -> tuple[str, ...]:
    mismatch = dataset_mismatches(run.read_dataset_metadata(), dataset.metadata)
    if not mismatch:
        return ()
    details = "; ".join(mismatch)
    if not allow_mismatch:
        raise ValueError(
            f"dataset {path} is not the dataset the run {run.path} was trained on "
            f"(differences to {run.dataset_path}: {details}); the metrics would not be the "
            "run's test error. Pass allow_dataset_mismatch=True (CLI: "
            "--allow-dataset-mismatch) to evaluate on it anyway"
        )
    logger.warning(
        "evaluating run %s on dataset %s, which is NOT the dataset the run was trained on; "
        "the metrics are not the run's test error. Differences: %s",
        run.path,
        path,
        details,
    )
    return tuple(mismatch)


def _check_padding(potential: Potential, dataset: Any, path: Path) -> None:
    expected = (potential.n_max_local, potential.n_max_long_range)
    got = (dataset.n_max_local, dataset.n_max_long_range)
    if expected != got:
        raise ValueError(
            f"dataset {path} has (n_max_local, n_max_long_range) = {got}, but the run "
            f"was trained with {expected}"
        )
    if dataset.metadata.get("max_atomic_number") != potential.config.grid.max_atomic_number:
        raise ValueError(
            f"dataset {path} has max_atomic_number = {dataset.metadata.get('max_atomic_number')}, "
            f"the run {potential.config.grid.max_atomic_number}"
        )


def evaluate(
    run_dir: str | Path,
    split: str = "test",
    epoch: str | int = "best",
    dataset_dir: str | Path | None = None,
    *,
    batch_size: int = 256,
    allow_dataset_mismatch: bool = False,
) -> EvaluationResult:
    """Evaluate one checkpoint of a run on a dataset split.

    Parameters
    ----------
    run_dir : str or Path
        Run directory (layout in :mod:`lrmlp.run`).
    split : {"test", "val", "train"}
        Split of the dataset to evaluate.
    epoch : {"best", "last"} or int
        Checkpoint (:meth:`lrmlp.run.RunDirectory.resolve_epoch`). The default
        ``"best"`` (lowest validation energy MSE) gives the test MAE of Tab. 5.3.
    dataset_dir : str or Path, optional
        Dataset directory; default the path stored in the run's ``dataset.json``.
    batch_size : int
        Molecules per model call; the result does not depend on it.
    allow_dataset_mismatch : bool
        Evaluate even if the dataset differs from the run's ``dataset.json`` (see
        :func:`dataset_mismatches`). A warning is logged and the result carries
        the differences (``dataset_matches_run: false`` in the JSON file).

    Returns
    -------
    EvaluationResult
        Energy MAE / RMSE / MSE, force MAE / RMSE (force models only), number of
        molecules, epoch and split. Also written to
        ``evaluation_<split>_<epoch>.json`` in the run directory.

    Raises
    ------
    FileNotFoundError
        If a run file, ``best.json`` or the requested checkpoint is missing (the
        message lists the available epochs), or the dataset does not exist.
    ValueError
        If the split is unknown or empty, the dataset is not the one the run was
        trained on and ``allow_dataset_mismatch`` is false (the message names the
        differing fields), or the dataset's padding does not match the run (B08;
        also with ``allow_dataset_mismatch``).
    """
    import tensorflow as tf

    from lrmlp.dataset import load_dataset, make_batched_tf_dataset

    batch_size = operator.index(batch_size)
    run = RunDirectory(run_dir)
    potential = Potential.from_run(run.path, epoch)
    assert potential.epoch is not None
    path = resolve_dataset_dir(run, dataset_dir)
    dataset = load_dataset(path)
    if split not in dataset.splits:
        raise ValueError(f"unknown split {split!r}; available: {sorted(dataset.splits)}")
    mismatch = _check_dataset(run, dataset, path, allow_dataset_mismatch)
    _check_padding(potential, dataset, path)
    data = dataset[split]
    if len(data) == 0:
        raise ValueError(f"split {split!r} of {path} is empty")

    pipeline = make_batched_tf_dataset(
        data, batch_size, include_long_range=potential.use_long_range
    )
    abs_errors: list[np.ndarray] = []
    force_abs_sum = 0.0
    force_sq_sum = 0.0
    n_force = 0
    for inputs, targets in pipeline:
        energies, atomic = potential.forward(inputs)
        # Explicit ops (B15); float32 per-molecule errors as in evaluate.py:26.
        abs_errors.append(tf.abs(energies - targets["energy"]).numpy())
        if potential.predicts_forces:
            diff = tf.cast(atomic[:, 1:4] - targets["forces"], tf.float64)
            force_abs_sum += float(tf.reduce_sum(tf.abs(diff)))
            force_sq_sum += float(tf.reduce_sum(tf.square(diff)))
            n_force += int(tf.size(diff))

    errors = np.concatenate(abs_errors).astype(np.float32)
    if errors.shape[0] != len(data):
        raise RuntimeError(f"evaluated {errors.shape[0]} molecules, expected {len(data)}")
    e64 = errors.astype(np.float64)
    mse = float(np.mean(e64**2))
    force_mae = force_rmse = None
    if potential.predicts_forces:
        force_mae = force_abs_sum / n_force
        force_rmse = math.sqrt(force_sq_sum / n_force)
    result = EvaluationResult(
        energy_mae=float(np.mean(e64)),
        energy_rmse=math.sqrt(mse),
        energy_mse=mse,
        force_mae=force_mae,
        force_rmse=force_rmse,
        n_molecules=int(errors.shape[0]),
        epoch=potential.epoch,
        split=split,
        energy_abs_errors=errors,
        dataset_mismatch=mismatch,
    )
    write_json(
        run.evaluation_path(split, potential.epoch),
        {**result.to_dict(), "dataset": str(path), "batch_size": batch_size},
    )
    return result
