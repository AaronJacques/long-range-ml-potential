"""Training of the atomic energy network (thesis Sec. 4.3.4, Eq. 4.21, 4.22 and 4.10).

This module replaces ``Neural-Network/Training.py@9413bfd``. The method is
unchanged:

* **Loss (Eq. 4.21).** One molecule per optimizer step (|B| = 1, p. 51, A15). The
  molecule energy is the sum of the atomic energies, E = Σ_i E_i, and the loss is
  the squared energy error ``(Σ_i E_i − E)²``.
* **Optimizer (Eq. 4.22).** Adam with Keras defaults and the exponentially decaying
  learning rate of :func:`lrmlp.schedules.learning_rate_schedule` (stepwise,
  deviation A07).
* **Experimental force mode (Eq. 4.10, A09).** With ``model.predict_forces`` the
  network has four outputs per atom (energy, three force components) and the loss
  is ``p_E · (Σ_i E_i − E)² + p_F · mean((F_pred − F)²)`` over all atoms and
  components, with p_E, p_F from :func:`lrmlp.schedules.loss_weight_tf`; step k
  uses p(k − 1). Forces are a direct network output, not −∇E. The force labels
  are aligned with the descriptor rows (atoms in input order, fixes B01).
* **Validation and model selection.** After every epoch the mean squared energy
  error over the validation molecules is computed; the epoch with the lowest
  validation MSE is the selected model (p. 52, A12) and recorded in ``best.json``.

What is new is the execution:

* The training step is a :func:`tf.function` with a fixed input signature (atom
  axis ``None``), so it is traced once per run and optionally XLA-compiled
  (``train.jit_compile``). ``train.steps_per_execution`` = K runs K steps per call
  in a graph loop over the dataset iterator; the result is identical to K single
  steps (bit-identical; default K = 8). With dropout active, the XLA-compiled step
  draws different (equally distributed) dropout masks than the plain step, so XLA
  runs are statistically equivalent but not bit-identical to non-XLA runs; without
  dropout both agree to rounding. If XLA is not available (:func:`xla_available`),
  ``train.jit_compile = true`` falls back to the plain step with a warning, and
  ``config.json`` records ``jit_compile = false``. Losses are accumulated in
  :class:`keras.metrics.Mean` and read on the host only every
  ``train.log_every_steps`` steps and at the end of an epoch.
* The model is called with ``training=True``, so dropout is active (B02). The
  legacy behaviour is ``model.dropout_rate = 0`` (profile ``legacy-head``).
* Seeds: ``keras.utils.set_random_seed(train.seed)`` and a seeded shuffle (B11);
  ``train.deterministic`` additionally enables deterministic TensorFlow ops.
* Validation runs several molecules per model call (atoms concatenated,
  ``segment_ids``), with the same result as one molecule per call.
* The log has no NaN columns (B07), squared errors are called ``*_mse`` (B06),
  unit (kcal/mol)² resp. (kcal/(mol·Å))², and mixed precision is not used
  (|E| ≈ 4·10⁵ kcal/mol, see ``docs/benchmarks.md``).

Run directory (:mod:`lrmlp.run`)
--------------------------------
``config.json`` (resolved configuration, K1 as integers), ``dataset.json``
(dataset metadata plus ``path``), ``training_log.csv`` (one row per epoch, see
:func:`log_columns`), ``tb/`` (TensorBoard scalars of every log column per epoch
and ``step/train_loss`` at every log interval; only if TensorBoard is installed,
otherwise one INFO line says so), ``checkpoints/epoch_{:04d}.weights.h5``
(``train.checkpoints``: ``"best_and_last"`` keeps the best and the last epoch,
``"all"`` every epoch) and ``best.json`` (``{"epoch", "val_energy_mse"}``).
Epochs are counted from 1.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import logging
import math
import os
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from lrmlp import run as run_module
from lrmlp.config import ExperimentConfig
from lrmlp.dataset import (
    DescriptorDataset,
    load_dataset,
    make_batched_tf_dataset,
    make_tf_dataset,
)
from lrmlp.model import assign_legacy_weights, build_model, resolve_k1
from lrmlp.run import RunDirectory, new_run_dir
from lrmlp.schedules import learning_rate_schedule, loss_weight, loss_weight_tf

__all__ = [
    "EVAL_ATOMS_PER_BATCH",
    "Trainer",
    "TrainingResult",
    "check_model_inputs",
    "log_columns",
    "train",
    "xla_available",
]

logger = logging.getLogger(__name__)

#: Upper bound on the atoms per validation model call (molecules are never split).
EVAL_ATOMS_PER_BATCH = 4096
#: Columns of ``training_log.csv`` in every mode.
ENERGY_COLUMNS = (
    "epoch",
    "train_loss",
    "train_energy_mse",
    "val_loss",
    "val_energy_mse",
    "learning_rate",
    "seconds",
)
#: Additional columns in the experimental force mode.
FORCE_COLUMNS = ("train_force_mse", "val_force_mse", "p_energy", "p_force")


def log_columns(predict_forces: bool) -> tuple[str, ...]:
    """Columns of ``training_log.csv`` (``docs/data-format.md``).

    Parameters
    ----------
    predict_forces : bool
        Experimental force mode (``model.predict_forces``).

    Returns
    -------
    tuple of str
        ``epoch, train_loss, train_energy_mse, val_loss, val_energy_mse,
        learning_rate, seconds`` and, in the force mode, ``train_force_mse,
        val_force_mse, p_energy, p_force``. ``*_energy_mse`` are mean squared
        molecule-energy errors in (kcal/mol)² (Eq. 4.21), ``*_force_mse`` mean squared
        force-component errors in (kcal/(mol·Å))²; ``*_loss`` is the optimised loss
        (equal to the energy MSE without forces, Eq. 4.10 with forces).
        ``learning_rate`` is the Adam step size η(t) of Eq. 4.22 (dimensionless,
        same unit as ``train.learning_rate``) at the **end** of the epoch, with
        t = the number of optimizer steps taken so far: the rate of the next step,
        not the rate(s) used during the epoch (with the stepwise decay these differ
        whenever a decay boundary falls into the epoch). ``seconds`` is the wall
        time of the epoch including validation.
    """
    return ENERGY_COLUMNS + FORCE_COLUMNS if predict_forces else ENERGY_COLUMNS


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of :func:`train`.

    Attributes
    ----------
    run_dir : Path
        The run directory.
    best_epoch : int
        Epoch (1-based) with the lowest validation energy MSE (p. 52).
    best_val_energy_mse : float
        Its validation energy MSE in (kcal/mol)².
    history : list of dict
        One row of ``training_log.csv`` per epoch.
    steps : int
        Optimizer steps performed.
    """

    run_dir: Path
    best_epoch: int
    best_val_energy_mse: float
    history: list[dict[str, float]]
    steps: int


def check_model_inputs(
    model: keras.Model, dataset: DescriptorDataset, use_long_range: bool, predict_forces: bool
) -> None:
    """Check that the model fits the padded shapes of a dataset (B08).

    Parameters
    ----------
    model : keras.Model
        Model built with :func:`lrmlp.model.build_model`.
    dataset : DescriptorDataset
        The dataset; N_max comes from its ``metadata.json``.
    use_long_range : bool
        Whether the model has the long-range inputs.
    predict_forces : bool
        Whether the model has four outputs per atom (else one).

    Raises
    ------
    ValueError
        If input names or shapes, or the output width, do not match; the message
        names N_max of the dataset and the shapes of the model.
    """
    n_loc, n_lr = dataset.n_max_local, dataset.n_max_long_range
    max_z = int(dataset.metadata["max_atomic_number"])
    expected: dict[str, tuple[int, ...]] = {
        "local_matrix": (n_loc, 4),
        "local_numbers": (n_loc, 2),
    }
    if use_long_range:
        expected["long_range_matrix"] = (n_lr, 4)
        expected["long_range_features"] = (n_lr, max_z + 1)
    got = {t.name: tuple(t.shape[1:]) for t in model.inputs}
    if got != expected:
        raise ValueError(
            f"model inputs {got} do not match the dataset {dataset.path} "
            f"(metadata.json: n_max_local={n_loc}, n_max_long_range={n_lr}, "
            f"max_atomic_number={max_z}); expected {expected}"
        )
    width = 4 if predict_forces else 1
    if tuple(model.outputs[0].shape[1:]) != (width,):
        raise ValueError(
            f"model output shape {tuple(model.outputs[0].shape)} does not match "
            f"model.predict_forces={predict_forces} (expected (None, {width}))"
        )


@functools.cache
def xla_available() -> tuple[bool, str]:
    """Check once per process whether XLA can compile a function on this platform.

    Returns
    -------
    tuple of (bool, str)
        ``(True, "")`` if a small :func:`tf.function` with ``jit_compile=True``
        compiles and runs, else ``(False, reason)`` with the error message.
    """

    def probe(x: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(tf.square(x))

    try:
        tf.function(probe, jit_compile=True)(tf.ones((2,), tf.float32)).numpy()
    except Exception as err:  # every failure means "no XLA here"
        first = str(err).strip().splitlines()
        return False, f"{type(err).__name__}: {first[0] if first else ''}"
    return True, ""


def _resolve_jit_compile(config: ExperimentConfig) -> ExperimentConfig:
    """Fall back to ``train.jit_compile = false`` (with a warning) if XLA is unavailable."""
    if not config.train.jit_compile:
        return config
    ok, reason = xla_available()
    if ok:
        return config
    logger.warning(
        "train.jit_compile = true, but XLA is not available on this platform (%s); "
        "training without XLA (config.json records jit_compile = false)",
        reason,
    )
    return dataclasses.replace(config, train=dataclasses.replace(config.train, jit_compile=False))


def _open_dataset(config: ExperimentConfig, dataset_dir: str | Path | None) -> DescriptorDataset:
    path = Path(dataset_dir) if dataset_dir is not None else config.data.output_dir
    assert path is not None  # filled in by ExperimentConfig
    try:
        dataset = load_dataset(path)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"{err}; build the dataset first (lrmlp prepare / lrmlp.dataset.build_dataset)"
        ) from err
    if dataset.grid != config.grid:
        raise ValueError(
            f"the dataset {dataset.path} was built with {dataset.grid}, the configuration "
            f"has {config.grid}; rebuild the dataset or fix the configuration"
        )
    if len(dataset["train"]) == 0:
        raise ValueError(f"the dataset {dataset.path} has no training molecules")
    if len(dataset["val"]) == 0:
        raise ValueError(
            f"the dataset {dataset.path} has no validation molecules "
            f"(val_fraction={dataset.metadata.get('val_fraction')}); training needs a "
            "validation split to select the best epoch by validation MSE (thesis p. 52)"
        )
    return dataset


class Trainer:
    """Training loop of the thesis (Eq. 4.21, 4.22, 4.10) with compiled steps.

    Construction performs, in this order: the XLA check (``train.jit_compile``
    falls back to ``false`` with a warning if XLA is unavailable), seeding, loading
    and checking the dataset, resolving K1, building the model, taking over ``initial_weights``
    and creating the run directory. :meth:`fit` then trains for
    ``train.epochs`` epochs.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    dataset_dir : str or Path, optional
        Preprocessed dataset; defaults to ``config.data.output_dir``.
    run_dir : str or Path, optional
        Run directory (must be empty or absent); defaults to a new directory
        ``<train.run_root>/<name>-YYYYmmdd-HHMMSS``.
    initial_weights : mapping, optional
        Start weights in the ``weights_by_class`` format of
        :func:`lrmlp.model.assign_legacy_weights`.
    model : keras.Model, optional
        Use this model instead of building one from ``config.model``; it is checked
        against the dataset (B08).
    eval_molecules_per_batch : int, optional
        Molecules per validation call; default: as many as fit into
        :data:`EVAL_ATOMS_PER_BATCH` atoms.

    Attributes
    ----------
    config : ExperimentConfig
        The configuration with K1 resolved to integers (as in ``config.json``).
    dataset : DescriptorDataset
        The dataset.
    model : keras.Model
        The trained model.
    optimizer : keras.optimizers.Adam
        The optimizer; ``optimizer.iterations`` counts the steps.
    run : RunDirectory
        The run directory.
    train_dataset, val_dataset : tf.data.Dataset
        Training pipeline (one molecule per element, shuffled with ``train.seed``)
        and validation pipeline (batched, stored order).

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    ValueError
        If the dataset has no training or validation molecules, was built with a
        different grid, or does not fit the model.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        dataset_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        initial_weights: Mapping[str, Sequence[Sequence[np.ndarray]]] | None = None,
        *,
        model: keras.Model | None = None,
        eval_molecules_per_batch: int | None = None,
    ) -> None:
        config = _resolve_jit_compile(config)
        tc = config.train
        keras.utils.set_random_seed(tc.seed)
        if tc.deterministic:
            tf.config.experimental.enable_op_determinism()

        self.dataset = _open_dataset(config, dataset_dir)
        n_loc, n_lr = self.dataset.n_max_local, self.dataset.n_max_long_range
        mc = config.model
        model_cfg = dataclasses.replace(
            mc,
            k1_local=resolve_k1(mc.k1_local, n_loc),
            k1_long=resolve_k1(mc.k1_long, n_lr) if mc.use_long_range else mc.k1_long,
        )
        self.config = dataclasses.replace(config, model=model_cfg)
        self._use_long_range = model_cfg.use_long_range
        self._forces = model_cfg.predict_forces

        if model is None:
            model = build_model(
                model_cfg, n_loc, n_lr, int(self.dataset.metadata["max_atomic_number"])
            )
        check_model_inputs(model, self.dataset, self._use_long_range, self._forces)
        if initial_weights is not None:
            assign_legacy_weights(model, initial_weights)
        self.model = model
        self._input_keys = [t.name for t in model.inputs]

        self.lr_schedule = learning_rate_schedule(tc)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.optimizer.build(self.model.trainable_variables)
        self._train_metrics = {
            name: keras.metrics.Mean(name=f"train_{name}", dtype="float64")
            for name in self._terms()
        }
        self._val_metrics = {
            name: keras.metrics.Mean(name=f"val_{name}", dtype="float64")
            for name in ("energy_se", "force_mse")[: 2 if self._forces else 1]
        }

        train_split, val_split = self.dataset["train"], self.dataset["val"]
        self.train_dataset = make_tf_dataset(
            train_split,
            shuffle=True,
            shuffle_buffer=tc.shuffle_buffer,
            seed=tc.seed,
            include_long_range=self._use_long_range,
        )
        if eval_molecules_per_batch is None:
            max_atoms = int(np.diff(np.asarray(val_split.atom_offsets)).max())
            eval_molecules_per_batch = max(1, EVAL_ATOMS_PER_BATCH // max(max_atoms, 1))
        self.val_dataset = make_batched_tf_dataset(
            val_split, eval_molecules_per_batch, include_long_range=self._use_long_range
        )
        self._n_train = len(train_split)

        train_spec = list(self.train_dataset.element_spec)
        self._train_step_fn = tf.function(
            self._train_step,
            input_signature=train_spec,
            jit_compile=tc.jit_compile,
            reduce_retracing=True,
        )
        self._train_steps_fn = tf.function(
            self._train_steps,
            input_signature=[tf.data.IteratorSpec(self.train_dataset.element_spec)],
            reduce_retracing=True,
        )
        self._val_step_fn = tf.function(
            self._val_step,
            input_signature=list(self.val_dataset.element_spec),
            reduce_retracing=True,
        )

        if run_dir is None:
            run_dir = new_run_dir(tc.run_root, config.name)
        # TensorBoard is optional (extra "tune"): without it, tf.summary.scalar raises.
        self._tensorboard = run_module.tensorboard_available()
        if not self._tensorboard:
            logger.info(
                "TensorBoard is not installed: no tb/ directory is written "
                '(install the extra "tune" for TensorBoard logging)'
            )
        self.run = RunDirectory(run_dir).create(
            self.config,
            {**self.dataset.metadata, "path": str(self.dataset.path.resolve())},
            tensorboard=self._tensorboard,
        )

    # ------------------------------------------------------------------ steps

    def _terms(self) -> tuple[str, ...]:
        return ("loss", "energy_se", "force_mse") if self._forces else ("loss", "energy_se")

    def _model_inputs(self, inputs: Mapping[str, Any]) -> list[Any]:
        return [inputs[k] for k in self._input_keys]

    def _loss_terms(self, prediction: Any, targets: Mapping[str, Any], step: Any) -> dict[str, Any]:
        """Loss of one molecule (Eq. 4.21; Eq. 4.10 in the force mode)."""
        energy = tf.reduce_sum(prediction[:, 0])
        energy_se = tf.square(energy - targets["energy"][0])
        if not self._forces:
            return {"loss": energy_se, "energy_se": energy_se}
        tc = self.config.train
        force_mse = tf.reduce_mean(tf.square(prediction[:, 1:] - targets["forces"]))
        p_energy = tf.cast(
            loss_weight_tf(
                step,
                tc.p_energy_start,
                tc.p_energy_limit,
                tc.loss_weight_decay_steps,
                tc.loss_weight_decay_rate,
            ),
            tf.float32,
        )
        p_force = tf.cast(
            loss_weight_tf(
                step,
                tc.p_force_start,
                tc.p_force_limit,
                tc.loss_weight_decay_steps,
                tc.loss_weight_decay_rate,
            ),
            tf.float32,
        )
        loss = p_energy * energy_se + p_force * force_mse
        return {"loss": loss, "energy_se": energy_se, "force_mse": force_mse}

    def _train_step(self, inputs: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
        """One Adam update on one molecule; the loss uses the weights before the update."""
        with tf.GradientTape() as tape:
            prediction = self.model(self._model_inputs(inputs), training=True)
            terms = self._loss_terms(prediction, targets, self.optimizer.iterations)
        variables = self.model.trainable_variables
        gradients = tape.gradient(terms["loss"], variables)
        self.optimizer.apply(gradients, variables)
        for name, metric in self._train_metrics.items():
            metric.update_state(terms[name])
        return terms

    def _train_steps(self, iterator: Any) -> None:
        """Up to ``steps_per_execution`` training steps in one graph call."""
        for _ in tf.range(self.config.train.steps_per_execution):
            optional = iterator.get_next_as_optional()
            if not optional.has_value():
                break
            inputs, targets = optional.get_value()
            self._train_step_fn(inputs, targets)

    def _val_step(self, inputs: Mapping[str, Any], targets: Mapping[str, Any]) -> None:
        """Accumulate the squared energy errors of a batch of validation molecules."""
        segment_ids = inputs["segment_ids"]
        n_molecules = tf.shape(targets["energy"])[0]
        prediction = self.model(self._model_inputs(inputs), training=False)
        energy = tf.math.unsorted_segment_sum(prediction[:, 0], segment_ids, n_molecules)
        self._val_metrics["energy_se"].update_state(tf.square(energy - targets["energy"]))
        if self._forces:
            squared = tf.reduce_sum(tf.square(prediction[:, 1:] - targets["forces"]), axis=1)
            atoms = tf.math.unsorted_segment_sum(tf.ones_like(squared), segment_ids, n_molecules)
            per_molecule = tf.math.unsorted_segment_sum(squared, segment_ids, n_molecules) / (
                3.0 * atoms
            )
            self._val_metrics["force_mse"].update_state(per_molecule)

    # ------------------------------------------------------------------ public steps

    def train_step(self, inputs: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, float]:
        """Run one compiled training step on one molecule.

        Parameters
        ----------
        inputs, targets : mapping
            One element of :attr:`train_dataset`.

        Returns
        -------
        dict of str to float
            ``loss``, ``energy_se`` (squared energy error, Eq. 4.21) and, in the force
            mode, ``force_mse``, computed with the weights before the update and
            with dropout active.
        """
        terms = self._train_step_fn(inputs, targets)
        return {k: float(v) for k, v in terms.items()}

    def compute_loss(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any], *, training: bool = False
    ) -> dict[str, float]:
        """Loss terms of one molecule without updating anything.

        Parameters
        ----------
        inputs, targets : mapping
            One element of :attr:`train_dataset`.
        training : bool
            Call the model in training mode (dropout active).

        Returns
        -------
        dict of str to float
            As :meth:`train_step`, with the loss weights of the current step.
        """
        prediction = self.model(self._model_inputs(inputs), training=training)
        terms = self._loss_terms(prediction, targets, self.optimizer.iterations)
        return {k: float(v) for k, v in terms.items()}

    def tracing_counts(self) -> dict[str, int]:
        """Return the number of traces per compiled function (1 for a constant signature)."""
        return {
            "train_step": int(self._train_step_fn.experimental_get_tracing_count()),
            "train_steps": int(self._train_steps_fn.experimental_get_tracing_count()),
            "val_step": int(self._val_step_fn.experimental_get_tracing_count()),
        }

    # ------------------------------------------------------------------ epochs

    def _loss_weights(self) -> tuple[float, float]:
        tc = self.config.train
        step = int(self.optimizer.iterations)
        return (
            loss_weight(
                step,
                tc.p_energy_start,
                tc.p_energy_limit,
                tc.loss_weight_decay_steps,
                tc.loss_weight_decay_rate,
            ),
            loss_weight(
                step,
                tc.p_force_start,
                tc.p_force_limit,
                tc.loss_weight_decay_steps,
                tc.loss_weight_decay_rate,
            ),
        )

    def train_epoch(self, epoch: int, writer: Any | None = None) -> dict[str, float]:
        """Train one epoch over the (shuffled) training split.

        Parameters
        ----------
        epoch : int
            Epoch number, for logging only.
        writer : tf.summary.SummaryWriter, optional
            Receives ``step/train_loss`` (running epoch mean) at every log interval.

        Returns
        -------
        dict of str to float
            ``train_loss``, ``train_energy_mse`` and, in the force mode,
            ``train_force_mse``: means over the steps of the epoch.
        """
        for metric in self._train_metrics.values():
            metric.reset_state()
        tc = self.config.train
        k = tc.steps_per_execution
        log_every = tc.log_every_steps
        iterator = iter(self.train_dataset)
        done, next_log = 0, log_every
        for _ in range(-(-self._n_train // k)):
            self._train_steps_fn(iterator)
            done = min(done + k, self._n_train)
            if done >= next_log:
                running = float(self._train_metrics["loss"].result())
                logger.info(
                    "epoch %d, step %d/%d: running train_loss %.6g",
                    epoch,
                    done,
                    self._n_train,
                    running,
                )
                if writer is not None:
                    with writer.as_default():
                        tf.summary.scalar(
                            "step/train_loss", running, step=int(self.optimizer.iterations)
                        )
                next_log = (done // log_every + 1) * log_every
        result = {
            "train_loss": float(self._train_metrics["loss"].result()),
            "train_energy_mse": float(self._train_metrics["energy_se"].result()),
        }
        if self._forces:
            result["train_force_mse"] = float(self._train_metrics["force_mse"].result())
        return result

    def validate(self) -> dict[str, float]:
        """Mean squared errors over the validation molecules (``training=False``).

        Returns
        -------
        dict of str to float
            ``val_energy_mse`` (mean over the molecules of the squared energy
            error) and, in the force mode, ``val_force_mse`` (mean over the
            molecules of the per-molecule force-component MSE).
        """
        for metric in self._val_metrics.values():
            metric.reset_state()
        for inputs, targets in self.val_dataset:
            self._val_step_fn(inputs, targets)
        result = {"val_energy_mse": float(self._val_metrics["energy_se"].result())}
        if self._forces:
            result["val_force_mse"] = float(self._val_metrics["force_mse"].result())
        return result

    def _save_checkpoint(self, epoch: int) -> None:
        """Write the checkpoint of ``epoch`` atomically (temporary file + ``os.replace``).

        An interruption leaves either no checkpoint or a complete one, never a
        partial file under the final name.
        """
        target = self.run.checkpoint_path(epoch)
        # Keras requires the ".weights.h5" suffix; the leading dot keeps the file out
        # of RunDirectory.list_checkpoints.
        tmp = target.with_name(f".{secrets.token_hex(8)}.tmp.{target.name}")
        try:
            self.model.save_weights(str(tmp))
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _prune_checkpoints(self, epoch: int, best_epoch: int) -> None:
        if self.config.train.checkpoints == "best_and_last":
            for old in self.run.list_checkpoints():
                if old not in (epoch, best_epoch):
                    self.run.checkpoint_path(old).unlink()

    def fit(self) -> TrainingResult:
        """Train for ``train.epochs`` epochs and select the best epoch (p. 52).

        Returns
        -------
        TrainingResult
            Run directory, best epoch and its validation energy MSE, history.

        Raises
        ------
        FloatingPointError
            If a loss becomes NaN or infinite (the epoch is still logged).
        """
        tc = self.config.train
        columns = log_columns(self._forces)
        writer = tf.summary.create_file_writer(str(self.run.tb_dir)) if self._tensorboard else None
        history: list[dict[str, float]] = []
        best_epoch, best_mse = 0, math.inf
        try:
            with self.run.log_path.open("w", newline="", encoding="utf-8") as fh:
                log = csv.DictWriter(fh, fieldnames=columns)
                log.writeheader()
                for epoch in range(1, tc.epochs + 1):
                    start = time.perf_counter()
                    row: dict[str, float] = {"epoch": epoch}
                    row.update(self.train_epoch(epoch, writer))
                    row.update(self.validate())
                    if self._forces:
                        p_energy, p_force = self._loss_weights()
                        row["val_loss"] = (
                            p_energy * row["val_energy_mse"] + p_force * row["val_force_mse"]
                        )
                        row["p_energy"], row["p_force"] = p_energy, p_force
                    else:
                        row["val_loss"] = row["val_energy_mse"]
                    row["learning_rate"] = float(self.lr_schedule(self.optimizer.iterations))
                    row["seconds"] = time.perf_counter() - start
                    row = {c: row[c] for c in columns}
                    history.append(row)
                    log.writerow({c: repr(v) for c, v in row.items()})
                    fh.flush()
                    if writer is not None:
                        with writer.as_default():
                            for c in columns[1:]:
                                tf.summary.scalar(c, row[c], step=epoch)
                        writer.flush()
                    bad = [c for c in columns if not math.isfinite(row[c])]
                    if bad:
                        raise FloatingPointError(
                            f"training diverged in epoch {epoch}: non-finite {bad} "
                            f"(see {self.run.log_path})"
                        )
                    # Checkpoint first, then best.json, then pruning: best.json never
                    # points to a missing or partial checkpoint.
                    self._save_checkpoint(epoch)
                    if row["val_energy_mse"] < best_mse:
                        best_epoch, best_mse = epoch, row["val_energy_mse"]
                        self.run.write_best(best_epoch, {"val_energy_mse": best_mse})
                    self._prune_checkpoints(epoch, best_epoch)
                    logger.info(
                        "epoch %d/%d: train_loss %.6g, val_energy_mse %.6g (kcal/mol)^2, "
                        "best epoch %d, %.1f s",
                        epoch,
                        tc.epochs,
                        row["train_loss"],
                        row["val_energy_mse"],
                        best_epoch,
                        row["seconds"],
                    )
        finally:
            if writer is not None:
                writer.close()
        return TrainingResult(
            run_dir=self.run.path,
            best_epoch=best_epoch,
            best_val_energy_mse=best_mse,
            history=history,
            steps=int(self.optimizer.iterations),
        )


def train(
    config: ExperimentConfig,
    dataset_dir: Path | None = None,
    run_dir: Path | None = None,
    initial_weights: Mapping[str, Sequence[Sequence[np.ndarray]]] | None = None,
) -> TrainingResult:
    """Train a model as in the thesis (Eq. 4.21, Eq. 4.22; Eq. 4.10 in the force mode).

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    dataset_dir : Path, optional
        Preprocessed dataset (:func:`lrmlp.dataset.build_dataset`); defaults to
        ``config.data.output_dir``.
    run_dir : Path, optional
        Run directory (empty or absent); defaults to a new directory under
        ``config.train.run_root``.
    initial_weights : mapping, optional
        Start weights in the ``weights_by_class`` format of
        :func:`lrmlp.model.assign_legacy_weights`.

    Returns
    -------
    TrainingResult
        ``run_dir``, ``best_epoch``, ``best_val_energy_mse``, ``history`` and ``steps``.

    See Also
    --------
    Trainer : the training loop, for finer control.
    """
    return Trainer(config, dataset_dir, run_dir, initial_weights).fit()
