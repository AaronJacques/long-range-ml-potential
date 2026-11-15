"""Hyperparameter grid search over experiment configurations.

Replaces ``Neural-Network/HyperparameterTester.py@9413bfd``, which could not run
(B04): it passed a dictionary keyed by ``hp.HParam`` objects to a function that
expected attributes, registered the name ``lr_decay_steps`` twice (the second one
was the loss-weight schedule ``decay_steps``), ignored its ``run_name`` so that all
trials wrote into one directory, and had no entry point. The search method is
unchanged: one model is trained per point of a Cartesian grid and the trials are
compared by their validation energy MSE (the model selection criterion of p. 52,
the parameter study behind Tab. 5.2 and Sec. 7.5).

Search space (TOML)
-------------------
::

    [space]                       # one axis per dotted configuration key
    "train.learning_rate" = [1e-2, 1e-3, 1e-4]
    train.lr_decay_steps = [1e3, 1e4]          # unquoted dotted keys work as well

    [[linked]]                    # one axis that sets several keys to the same value
    name = "decay_rate"           # optional HParams name, default "key1+key2"
    keys = ["train.lr_decay_rate", "train.loss_weight_decay_rate"]
    values = [0.96, 0.97]

Every key may occur only once in the whole space. The axes are ordered as
``[space]`` (file order) followed by ``[[linked]]`` (file order); the trials are
the Cartesian product in this order with the last axis varying fastest. The
overrides of a trial are applied with :func:`lrmlp.config.load_config`, exactly
like ``lrmlp train --set``: relative paths are resolved against the directory of
the base configuration.

Output
------
``<run_root>/tune-<name>-YYYYmmdd-HHMMSS/``::

    search.json          base config, space file, axes, number of trials
    summary.csv          one row per trial: trial, status, overrides, metric, run_dir
    events.out.tfevents  TensorBoard HParams configuration
    trial-0000/          a complete run directory (lrmlp.run) plus the trial's
    trial-0001/          HParams session (hyperparameters, best_val_energy_mse)

A failing trial is recorded with status ``failed`` and its error message; the
remaining trials still run. The search needs TensorBoard (extra ``tune``); without
it, :func:`run_search` raises :class:`ModuleNotFoundError` before any trial starts,
and only ``dry_run=True`` works. The preprocessed dataset is not built here: run
``lrmlp prepare`` first (axes on ``grid.*`` or ``data.*`` need matching datasets).
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import math
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lrmlp import run as run_module
from lrmlp.config import ExperimentConfig, load_config
from lrmlp.run import new_run_dir, write_json

__all__ = [
    "METRIC",
    "SearchAxis",
    "SearchSpace",
    "TrialResult",
    "TrialResults",
    "load_space",
    "run_search",
]

logger = logging.getLogger(__name__)

#: Metric of a trial: lowest validation energy MSE over the epochs, (kcal/mol)².
METRIC = "best_val_energy_mse"
#: Error message of a search without TensorBoard (the HParams plugin is part of it).
TENSORBOARD_REQUIRED = (
    "lrmlp tune needs TensorBoard for its HParams logs; install the extra "
    '"tune" (pip install -e ".[tune]"). --dry-run works without it'
)

#: Columns of ``summary.csv`` before and after the override columns.
SUMMARY_HEAD = ("trial", "status")
SUMMARY_TAIL = (METRIC, "best_epoch", "run_dir", "message")
#: Number of combinations listed by a dry run.
DRY_RUN_LISTED = 5

TrialStatus = Literal["planned", "succeeded", "failed"]


# --------------------------------------------------------------------------- space


@dataclass(frozen=True)
class SearchAxis:
    """One axis of the grid.

    Parameters
    ----------
    name : str
        HParams name (unique in the space).
    keys : tuple of str
        Dotted configuration keys that all receive the axis value.
    values : tuple
        The values of the axis, in order.
    """

    name: str
    keys: tuple[str, ...]
    values: tuple[Any, ...]


@dataclass(frozen=True)
class SearchSpace:
    """A Cartesian grid of configuration overrides.

    Parameters
    ----------
    axes : tuple of SearchAxis
        The axes; the last one varies fastest.
    """

    axes: tuple[SearchAxis, ...]

    def __post_init__(self) -> None:
        """Check that names and keys are unique and no axis is empty."""
        names = [a.name for a in self.axes]
        dup_names = sorted({n for n in names if names.count(n) > 1})
        if dup_names:
            raise ValueError(f"duplicate hyperparameter name(s) {dup_names}")
        keys = [k for a in self.axes for k in a.keys]
        dup_keys = sorted({k for k in keys if keys.count(k) > 1})
        if dup_keys:
            raise ValueError(f"key(s) {dup_keys} occur on more than one axis")
        for axis in self.axes:
            if not axis.keys:
                raise ValueError(f"axis {axis.name!r} has no keys")
            if not axis.values:
                raise ValueError(f"axis {axis.name!r} has no values")

    def __len__(self) -> int:
        """Return the number of combinations (product of the axis lengths)."""
        return math.prod(len(a.values) for a in self.axes)

    @property
    def names(self) -> tuple[str, ...]:
        """HParams names of the axes."""
        return tuple(a.name for a in self.axes)

    @property
    def keys(self) -> tuple[str, ...]:
        """All configuration keys, in axis order."""
        return tuple(k for a in self.axes for k in a.keys)

    def combinations(self) -> Iterator[dict[str, Any]]:
        """Iterate over the combinations as override mappings.

        Yields
        ------
        dict
            Dotted configuration key → value, for every key of every axis. The order
            is that of :func:`itertools.product` over the axes.
        """
        for point in itertools.product(*(a.values for a in self.axes)):
            yield {k: v for axis, v in zip(self.axes, point, strict=True) for k in axis.keys}

    def hparams(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Map the overrides of one combination to ``{axis name: value}``."""
        return {a.name: overrides[a.keys[0]] for a in self.axes}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SearchSpace:
        """Build a space from parsed TOML (see the module docstring).

        Parameters
        ----------
        d : Mapping
            ``space`` (table of dotted key → list of values, nested tables allowed)
            and/or ``linked`` (list of tables with ``keys``, ``values`` and optionally
            ``name``).

        Returns
        -------
        SearchSpace
            The validated space.

        Raises
        ------
        ValueError
            On unknown tables, non-list values, empty axes, duplicate keys or names.
        """
        unknown = sorted(set(d) - {"space", "linked"})
        if unknown:
            raise ValueError(
                f"unknown table(s) {unknown} in the search space; allowed: space, linked"
            )
        axes: list[SearchAxis] = []
        space = d.get("space", {})
        if not isinstance(space, Mapping):
            raise ValueError("[space] must be a table")
        for key, values in _flatten(space):
            if not isinstance(values, list):
                raise ValueError(f"[space] {key!r} must be a list of values, got {values!r}")
            axes.append(SearchAxis(key, (key,), tuple(values)))
        linked = d.get("linked", [])
        if not isinstance(linked, list):
            raise ValueError("'linked' must be an array of tables ([[linked]])")
        for i, entry in enumerate(linked):
            if not isinstance(entry, Mapping):
                raise ValueError(f"[[linked]] #{i + 1} must be a table")
            extra = sorted(set(entry) - {"name", "keys", "values"})
            if extra:
                raise ValueError(f"[[linked]] #{i + 1}: unknown field(s) {extra}")
            keys, values = entry.get("keys"), entry.get("values")
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise ValueError(f"[[linked]] #{i + 1}: 'keys' must be a list of strings")
            if not isinstance(values, list):
                raise ValueError(f"[[linked]] #{i + 1}: 'values' must be a list")
            name = entry.get("name", "+".join(keys))
            if not isinstance(name, str) or not name:
                raise ValueError(f"[[linked]] #{i + 1}: 'name' must be a non-empty string")
            axes.append(SearchAxis(name, tuple(keys), tuple(values)))
        if not axes:
            raise ValueError("the search space has no axes")
        return cls(tuple(axes))


def _flatten(table: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Flatten nested TOML tables to dotted keys (configuration values are never tables)."""
    for key, value in table.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from _flatten(value, f"{dotted}.")
        else:
            yield dotted, value


def load_space(path: str | Path) -> SearchSpace:
    """Read a search space TOML file.

    Parameters
    ----------
    path : str or Path
        TOML file with ``[space]`` and/or ``[[linked]]``.

    Returns
    -------
    SearchSpace
        The space.

    Raises
    ------
    ValueError
        If the file is not a valid search space.
    """
    path = Path(path)
    with path.open("rb") as fh:
        d = tomllib.load(fh)
    try:
        return SearchSpace.from_dict(d)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------- results


@dataclass
class TrialResult:
    """Outcome of one trial.

    Attributes
    ----------
    index : int
        Position in the grid order (0-based).
    name : str
        ``trial-0000``, ``trial-0001``, ...
    overrides : dict
        Dotted configuration key → value of this trial.
    status : {"planned", "succeeded", "failed"}
        ``planned`` in a dry run.
    best_val_energy_mse : float or None
        Lowest validation energy MSE in (kcal/mol)² (p. 52), if the trial succeeded.
    best_epoch : int or None
        Epoch of that minimum.
    run_dir : Path or None
        Run directory of the trial (``None`` in a dry run).
    message : str or None
        Error message of a failed trial.
    """

    index: int
    name: str
    overrides: dict[str, Any]
    status: TrialStatus = "planned"
    best_val_energy_mse: float | None = None
    best_epoch: int | None = None
    run_dir: Path | None = None
    message: str | None = None


class TrialResults(list[TrialResult]):
    """List of :class:`TrialResult` with a one-line summary as ``str``.

    Attributes
    ----------
    search_dir : Path or None
        Directory of the search (``None`` in a dry run).
    n_combinations : int
        Size of the full grid (the list may hold fewer trials, ``max_trials``).
    """

    search_dir: Path | None = None
    n_combinations: int = 0

    def best(self) -> TrialResult | None:
        """Return the succeeded trial with the lowest metric, or ``None``."""
        done = [t for t in self if t.best_val_energy_mse is not None]
        return min(done, key=lambda t: t.best_val_energy_mse or 0.0) if done else None

    def __str__(self) -> str:
        """Summarise counts, the best trial and the search directory."""
        counts = {s: sum(t.status == s for t in self) for s in ("planned", "succeeded", "failed")}
        parts = [f"{len(self)} of {self.n_combinations} trial(s)"]
        parts += [f"{n} {s}" for s, n in counts.items() if n]
        best = self.best()
        if best is not None:
            parts.append(f"best {best.name} {METRIC}={best.best_val_energy_mse:.6g}")
        if self.search_dir is not None:
            parts.append(f"in {self.search_dir}")
        return ", ".join(parts)


# --------------------------------------------------------------------------- search


def _hparam_value(value: Any) -> bool | float | str:
    """Convert a value to a type TensorBoard HParams accepts (bool, float, str)."""
    if isinstance(value, bool | str):
        return value
    if isinstance(value, int | float):
        return float(value)
    return json.dumps(value)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool | Path):
        return str(value)
    return json.dumps(value, default=str)


def _write_summary(path: Path, space: SearchSpace, trials: Sequence[TrialResult]) -> None:
    keys = space.keys
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([*SUMMARY_HEAD, *keys, *SUMMARY_TAIL])
        for t in trials:
            writer.writerow(
                [
                    t.name,
                    t.status,
                    *(_csv_value(t.overrides[k]) for k in keys),
                    _csv_value(t.best_val_energy_mse),
                    _csv_value(t.best_epoch),
                    _csv_value(t.run_dir),
                    _csv_value(t.message),
                ]
            )
    tmp.replace(path)


def _hparam_domains(space: SearchSpace, hp: Any) -> dict[str, Any]:
    """One ``hp.HParam`` per axis with a discrete domain of its (converted) values."""
    result = {}
    for axis in space.axes:
        values = list(dict.fromkeys(_hparam_value(v) for v in axis.values))
        kinds = {type(v) for v in values}
        domain = hp.Discrete(values) if len(kinds) == 1 else None
        result[axis.name] = hp.HParam(axis.name, domain)
    return result


def run_search(
    config_path: str | Path,
    space_path: str | Path,
    max_trials: int | None = None,
    dry_run: bool = False,
    run_root: str | Path | None = None,
) -> TrialResults:
    """Train one model per grid point and compare the validation energy MSE.

    The search of ``HyperparameterTester.py`` (B04), working: each trial is a
    regular training run (:func:`lrmlp.training.train`, Eq. 4.21/4.22) with the
    trial's overrides, selected by the lowest validation MSE (p. 52; Sec. 7.5).

    Parameters
    ----------
    config_path : str or Path
        Base experiment TOML (:func:`lrmlp.config.load_config`).
    space_path : str or Path
        Search space TOML (:func:`load_space`).
    max_trials : int, optional
        Run only the first ``max_trials`` combinations in grid order.
    dry_run : bool
        Only count and list the combinations (status ``planned``); nothing is
        written and nothing is trained.
    run_root : str or Path, optional
        Parent of the search directory; defaults to ``train.run_root`` of the base
        configuration.

    Returns
    -------
    TrialResults
        One :class:`TrialResult` per selected trial, in grid order.

    Raises
    ------
    ValueError
        If the space is invalid, a key is not a configuration field, the first
        combination gives an invalid configuration or ``max_trials < 1``.
    FileNotFoundError
        If the dataset of the first trial does not exist (not raised in a dry run).
    ModuleNotFoundError
        If TensorBoard is not installed (not raised in a dry run).
    """
    if max_trials is not None and max_trials < 1:
        raise ValueError(f"max_trials must be >= 1, got {max_trials}")
    config_path, space_path = Path(config_path), Path(space_path)
    space = load_space(space_path)
    base = load_config(config_path)
    n_total = len(space)
    n = n_total if max_trials is None else min(n_total, max_trials)
    combos = list(itertools.islice(space.combinations(), n))
    # all keys occur in every combination, so this checks every key once up front
    first = load_config(config_path, combos[0])

    results = TrialResults(
        TrialResult(i, f"trial-{i:04d}", combo) for i, combo in enumerate(combos)
    )
    results.n_combinations = n_total
    logger.info(
        "search space %s: %d axis/axes, %d combination(s), %d trial(s) selected",
        space_path,
        len(space.axes),
        n_total,
        n,
    )
    if dry_run:
        for t in results[:DRY_RUN_LISTED]:
            logger.info("  %s: %s", t.name, space.hparams(t.overrides))
        if n > DRY_RUN_LISTED:
            logger.info("  ... and %d more", n - DRY_RUN_LISTED)
        return results

    if not run_module.tensorboard_available():
        raise ModuleNotFoundError(TENSORBOARD_REQUIRED, name="tensorboard")

    from lrmlp.dataset import METADATA_FILE

    dataset_dir = first.data.output_dir
    assert dataset_dir is not None  # filled in by ExperimentConfig
    if not (dataset_dir / METADATA_FILE).is_file():
        raise FileNotFoundError(
            f"no dataset at {dataset_dir}; build it first (lrmlp prepare --config {config_path})"
        )

    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp

    from lrmlp.training import train

    root = Path(run_root) if run_root is not None else base.train.run_root
    search_dir = new_run_dir(root, f"tune-{base.name}")
    results.search_dir = search_dir
    write_json(
        search_dir / "search.json",
        {
            "config": str(config_path.resolve()),
            "space": str(space_path.resolve()),
            "axes": [
                {"name": a.name, "keys": list(a.keys), "values": list(a.values)} for a in space.axes
            ],
            "n_combinations": n_total,
            "n_trials": n,
            "metric": METRIC,
        },
    )
    hparams = _hparam_domains(space, hp)
    with tf.summary.create_file_writer(str(search_dir)).as_default():
        hp.hparams_config(
            hparams=list(hparams.values()),
            metrics=[hp.Metric(METRIC, display_name="best val energy MSE (kcal/mol)^2")],
        )
    summary = search_dir / "summary.csv"
    _write_summary(summary, space, results)

    for trial in results:
        trial.run_dir = search_dir / trial.name
        logger.info("--- %s of %d: %s", trial.name, n, space.hparams(trial.overrides))
        try:
            config: ExperimentConfig = load_config(config_path, trial.overrides)
            outcome = train(config, run_dir=trial.run_dir)
        except Exception as exc:  # one failing trial must not stop the search
            trial.status = "failed"
            trial.message = f"{type(exc).__name__}: {exc}"
            logger.error("%s failed: %s", trial.name, trial.message)
            logger.debug("traceback of %s", trial.name, exc_info=True)
        else:
            trial.status = "succeeded"
            trial.best_val_energy_mse = float(outcome.best_val_energy_mse)
            trial.best_epoch = int(outcome.best_epoch)
            logger.info(
                "%s: %s = %.6g (epoch %d)",
                trial.name,
                METRIC,
                trial.best_val_energy_mse,
                trial.best_epoch,
            )
        trial.run_dir.mkdir(parents=True, exist_ok=True)
        values = {
            hparams[name]: _hparam_value(v) for name, v in space.hparams(trial.overrides).items()
        }
        with tf.summary.create_file_writer(str(trial.run_dir)).as_default():
            hp.hparams(values, trial_id=trial.name)
            if trial.best_val_energy_mse is not None:
                tf.summary.scalar(METRIC, trial.best_val_energy_mse, step=trial.best_epoch or 0)
        _write_summary(summary, space, results)

    return results
