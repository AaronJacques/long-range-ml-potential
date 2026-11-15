"""Typed experiment configuration loaded from TOML.

This module replaces the class attributes of the legacy ``Neural-Network/Constants.py``
with frozen dataclasses. An :class:`ExperimentConfig` bundles the grid, data, model
and training settings of one experiment. Symbols in the docstrings refer to the
thesis (chapter 4 and Tab. 5.1/5.2).

Relative paths in a configuration file (``data.sources``, ``data.output_dir`` and
``train.run_root``) are resolved relative to the directory of that file, so a
configuration works independently of the current working directory.

The module uses the standard library only; it never imports TensorFlow.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "GridConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
    "parse_overrides",
]

#: Allowed values of :attr:`GridConfig.assignment`.
ASSIGNMENTS = ("thesis", "legacy")
#: Allowed values of :attr:`TrainConfig.checkpoints`.
CHECKPOINT_POLICIES = ("all", "best_and_last")
#: Allowed values of :attr:`ModelConfig.activation`.
ACTIVATIONS = ("elu", "relu")
#: Names of the configuration sections, in file order.
SECTIONS = ("grid", "data", "model", "train")


# --------------------------------------------------------------------------- validation helpers


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive(owner: str, name: str, value: float) -> None:
    _require(value > 0, f"{owner}.{name} must be positive, got {value!r}")


def _non_negative(owner: str, name: str, value: float) -> None:
    _require(value >= 0, f"{owner}.{name} must be non-negative, got {value!r}")


def _k1(owner: str, name: str, value: int | str) -> None:
    _require(
        value == "auto" or (isinstance(value, int) and value > 0),
        f"{owner}.{name} must be a positive integer or 'auto', got {value!r}",
    )


# --------------------------------------------------------------------------- sections


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Hierarchical cell grid and cutoff (thesis Sec. 4.2 and 4.3.2, Tab. 5.1).

    Parameters
    ----------
    grid_size : float
        Edge length of the finest grid cells in Å, Δg (Tab. 5.1).
        Legacy: ``Dataset.GRID_SIZE``.
    max_local_level : int
        Highest cell level that belongs to the local environment, l_max (Eq. 4.13).
        Legacy: ``Dataset.MAX_LOCAL_LEVEL``.
    inner_cutoff_offset : float
        Distance in Å between cutoff and inner cutoff, r_c − r_cs = 0.2 Å (Tab. 5.1).
    max_atomic_number : int
        Largest atomic number of the one-hot species encoding, m_max (p. 42).
        Legacy: ``Dataset.MAX_ATOM_ELEMENTS``.
    assignment : {"thesis", "legacy"}
        ``"thesis"`` puts every atom with r ≤ r_c into the local environment and the
        rest into the reduced grid (p. 43). ``"legacy"`` reproduces the cell-based
        assignment of the legacy code (``Dataset.py:310-356``, deviation A01/A02).
    sort_neighbours : bool
        Sort the local rows (with Z_i) and the long-range rows (with F_i) of each
        atom by descending first column (s resp. 1/r), as ``Dataset.sort_matrix``
        does (``Dataset.py:362-376``). Not part of the thesis.
    """

    grid_size: float = 1.0
    max_local_level: int = 6
    inner_cutoff_offset: float = 0.2
    max_atomic_number: int = 100
    assignment: Literal["thesis", "legacy"] = "thesis"
    sort_neighbours: bool = False

    def __post_init__(self) -> None:
        """Validate the grid parameters."""
        _positive("grid", "grid_size", self.grid_size)
        _positive("grid", "max_local_level", self.max_local_level)
        _positive("grid", "inner_cutoff_offset", self.inner_cutoff_offset)
        _positive("grid", "max_atomic_number", self.max_atomic_number)
        _require(
            self.assignment in ASSIGNMENTS,
            f"grid.assignment must be one of {ASSIGNMENTS}, got {self.assignment!r}",
        )
        _require(
            self.inner_cutoff_offset < self.cutoff,
            f"grid.inner_cutoff_offset ({self.inner_cutoff_offset}) must be smaller than "
            f"the cutoff ({self.cutoff})",
        )

    @property
    def cutoff(self) -> float:
        """Cutoff radius r_c in Å (Tab. 5.1, Eq. 4.14, p. 42).

        The thesis states that all cells of G_i lie within r_c; in 3D this is false
        (distances in G_i reach up to √3·(l_max+1)·Δg), see deviation A02 and
        ``assignment``. The expression is exactly the one of the legacy code
        (``Constants.py:34``).
        """
        return (self.max_local_level + 1 + 0.5**0.5) * self.grid_size

    @property
    def inner_cutoff(self) -> float:
        """Inner cutoff radius r_cs = r_c − 0.2 Å of the smooth switch (Tab. 5.1).

        Same expression as the legacy code (``Constants.py:36``).
        """
        return self.cutoff - self.inner_cutoff_offset


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Raw data sources, sample selection and split (thesis Sec. 4.3.1, p. 40).

    Parameters
    ----------
    sources : tuple of Path
        ``.npz`` trajectories of the sGDML datasets [16] with ``E``, ``F``, ``R``
        and ``z``. Several sources are concatenated.
    n_samples_per_source : int or "all"
        Upper bound on the number of leading frames taken from each source: at most
        65 000 samples in the thesis (p. 40). A source with fewer frames contributes
        all of them (the legacy code slices ``[:n]``; the clipping to the available
        frames is done by the dataset builder). ``"all"`` uses every frame of each
        source (Stachyose: all 27 272).
    val_fraction : float
        Fraction of the samples in the validation split (80/10/10, p. 40).
    test_fraction : float
        Fraction of the samples in the test split (80/10/10, p. 40).
    split_seed : int
        Seed of the random split.
    output_dir : Path or None
        Directory of the preprocessed dataset. ``None`` means
        ``data/processed/<name>`` and is filled in by :class:`ExperimentConfig`.
    n_workers : int
        Worker processes of the preprocessing; 0 uses all CPUs.
    """

    sources: tuple[Path, ...]
    n_samples_per_source: int | Literal["all"] = 65000
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 42
    output_dir: Path | None = None
    n_workers: int = 0

    def __post_init__(self) -> None:
        """Validate the data parameters."""
        _require(len(self.sources) > 0, "data.sources must contain at least one path")
        _require(
            self.n_samples_per_source == "all"
            or (isinstance(self.n_samples_per_source, int) and self.n_samples_per_source > 0),
            "data.n_samples_per_source must be a positive integer or 'all', "
            f"got {self.n_samples_per_source!r}",
        )
        _non_negative("data", "val_fraction", self.val_fraction)
        _non_negative("data", "test_fraction", self.test_fraction)
        _require(
            self.val_fraction + self.test_fraction < 1,
            "data.val_fraction + data.test_fraction must be smaller than 1, got "
            f"{self.val_fraction} + {self.test_fraction}",
        )
        _non_negative("data", "n_workers", self.n_workers)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Network architecture (thesis Sec. 4.3.3, Fig. 4.2-4.6, Tab. 5.2).

    Parameters
    ----------
    activation : {"elu", "relu"}
        Activation function; ELU in the thesis (Fig. 4.2/4.4).
    embedding_dims : tuple of int
        Layer widths of the embedding network (Fig. 4.3). The legacy code uses
        (64, 128, 128, 128), deviation F3.
    fitting_width : int
        Width of the fitting network (Fig. 4.4).
    fitting_res_blocks : tuple of int
        Widths of the residual blocks of the fitting network (Fig. 4.4). The legacy
        code uses (128, 128), deviation F3.
    dropout_rate : float
        Dropout rate, 0.2 in the thesis (Fig. 4.2/4.4). Never active in the legacy
        code (B02), hence 0.0 in the legacy profiles.
    k1_local, k1_long : int or "auto"
        K1 of the local and long-range descriptor (Tab. 5.2). ``"auto"`` sets
        K1 = N_max of the dataset ("N_max = K1 always chosen", p. 52).
    k2_local, k2_long : int
        K2 of the local and long-range descriptor (Tab. 5.2).
    use_long_range : bool
        Use the long-range part (Fig. 4.5) or only the local one (Fig. 4.6).
    predict_forces : bool
        Also predict forces. The thesis trains on energies only (p. 40); force
        prediction is experimental (F4).
    """

    activation: Literal["elu", "relu"] = "elu"
    embedding_dims: tuple[int, ...] = (64, 128, 128, 256)
    fitting_width: int = 128
    fitting_res_blocks: tuple[int, ...] = (128, 128, 128)
    dropout_rate: float = 0.2
    k1_local: int | Literal["auto"] = "auto"
    k1_long: int | Literal["auto"] = "auto"
    k2_local: int = 4
    k2_long: int = 1
    use_long_range: bool = True
    predict_forces: bool = False

    def __post_init__(self) -> None:
        """Validate the model parameters."""
        _require(
            self.activation in ACTIVATIONS,
            f"model.activation must be one of {ACTIVATIONS}, got {self.activation!r}",
        )
        _require(len(self.embedding_dims) > 0, "model.embedding_dims must not be empty")
        for i, width in enumerate(self.embedding_dims):
            _positive("model", f"embedding_dims[{i}]", width)
        _positive("model", "fitting_width", self.fitting_width)
        for i, width in enumerate(self.fitting_res_blocks):
            _positive("model", f"fitting_res_blocks[{i}]", width)
        _require(
            0 <= self.dropout_rate < 1,
            f"model.dropout_rate must satisfy 0 <= rate < 1, got {self.dropout_rate!r}",
        )
        _k1("model", "k1_local", self.k1_local)
        _k1("model", "k1_long", self.k1_long)
        _positive("model", "k2_local", self.k2_local)
        _positive("model", "k2_long", self.k2_long)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation, loss weighting and bookkeeping (thesis Sec. 4.3.4, Tab. 5.2).

    Parameters
    ----------
    epochs : int
        Number of epochs (Tab. 5.2).
    learning_rate : float
        Initial learning rate η_init (Tab. 5.2).
    lr_decay_steps : float
        Decay steps s of the exponential learning-rate schedule (Eq. 4.22, Tab. 5.2).
    lr_decay_rate : float
        Decay rate of the learning-rate schedule, 0.97 (Eq. 4.22).
    lr_staircase : bool
        Decay the learning rate in discrete steps.
    p_energy_start, p_energy_limit : float
        Start and limit of the energy loss weight p_E (Eq. 4.10).
    p_force_start, p_force_limit : float
        Start and limit of the force loss weight p_F (Eq. 4.10).
    loss_weight_decay_steps : float
        Decay steps of the loss weights. Legacy: ``Hyperparameters.decay_steps``.
    loss_weight_decay_rate : float
        Decay rate of the loss weights. Legacy: ``Hyperparameters.decay_rate``.
    shuffle_buffer : int
        Shuffle buffer size of the training pipeline; 1 disables shuffling.
    seed : int
        Global random seed (not given in the thesis).
    deterministic : bool
        Request deterministic TensorFlow ops.
    steps_per_execution : int
        Training steps per compiled call (purely technical). Default 8: the result
        is bit-identical to one step per call (also with dropout and with XLA) and
        training is faster (``docs/benchmarks.md``).
    jit_compile : bool
        Compile the training step with XLA (purely technical). Default ``False``;
        the thesis presets (``configs/{paracetamol,aspirin,ac-ala3-nhme,
        stachyose}.toml``) set ``true`` (about 2× faster, ``docs/benchmarks.md``).
        Without dropout, XLA and non-XLA runs agree to rounding. **With dropout
        (``model.dropout_rate > 0``), XLA draws other dropout masks** than the plain
        step: statistically equivalent, but not bit-identical to a non-XLA run with
        the same seed. If XLA is not available on the platform, the trainer logs a
        warning and trains without XLA (recorded as ``false`` in ``config.json``).
    log_every_steps : int
        Interval of the step-level progress log. The log is written at the end of
        the first compiled call (``steps_per_execution`` steps) that reaches the
        next multiple of the interval.
    checkpoints : {"best_and_last", "all"}
        Keep only the best and the last checkpoint, or one per epoch. The best epoch
        is the one with the lowest validation MSE (p. 52).
    run_root : Path
        Directory in which run directories are created.
    """

    epochs: int = 200
    learning_rate: float = 1e-4
    lr_decay_steps: float = 2.08e5
    lr_decay_rate: float = 0.97
    lr_staircase: bool = True
    p_energy_start: float = 1.0
    p_energy_limit: float = 1.0
    p_force_start: float = 1.0
    p_force_limit: float = 0.0
    loss_weight_decay_steps: float = 7.5e3
    loss_weight_decay_rate: float = 0.97
    shuffle_buffer: int = 1000
    seed: int = 42
    deterministic: bool = False
    steps_per_execution: int = 8
    jit_compile: bool = False
    log_every_steps: int = 1000
    checkpoints: Literal["all", "best_and_last"] = "best_and_last"
    run_root: Path = field(default_factory=lambda: Path("runs"))

    def __post_init__(self) -> None:
        """Validate the training parameters."""
        _positive("train", "epochs", self.epochs)
        _positive("train", "learning_rate", self.learning_rate)
        _positive("train", "lr_decay_steps", self.lr_decay_steps)
        _positive("train", "lr_decay_rate", self.lr_decay_rate)
        for name in ("p_energy_start", "p_energy_limit", "p_force_start", "p_force_limit"):
            _non_negative("train", name, getattr(self, name))
        _positive("train", "loss_weight_decay_steps", self.loss_weight_decay_steps)
        _positive("train", "loss_weight_decay_rate", self.loss_weight_decay_rate)
        _positive("train", "shuffle_buffer", self.shuffle_buffer)
        _require(
            self.steps_per_execution >= 1,
            f"train.steps_per_execution must be >= 1, got {self.steps_per_execution!r}",
        )
        _positive("train", "log_every_steps", self.log_every_steps)
        _require(
            self.checkpoints in CHECKPOINT_POLICIES,
            f"train.checkpoints must be one of {CHECKPOINT_POLICIES}, got {self.checkpoints!r}",
        )


_SECTION_TYPES: dict[str, type] = {
    "grid": GridConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
}


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration of one experiment.

    Parameters
    ----------
    name : str
        Experiment name; prefix of the run directory and default dataset name.
    grid : GridConfig
        Grid and cutoff.
    data : DataConfig
        Data sources and split.
    model : ModelConfig
        Architecture.
    train : TrainConfig
        Training settings.
    """

    name: str
    grid: GridConfig
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        """Validate the name and fill in the default dataset directory."""
        _require(
            isinstance(self.name, str) and self.name.strip() != "",
            "name must be a non-empty string",
        )
        _require(
            "/" not in self.name and "\\" not in self.name and self.name not in (".", ".."),
            f"name must be usable as a directory name, got {self.name!r}",
        )
        if self.data.output_dir is None:
            data = dataclasses.replace(self.data, output_dir=Path("data") / "processed" / self.name)
            object.__setattr__(self, "data", data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain JSON/TOML types.

        Returns
        -------
        dict
            ``name``, one dictionary per section and a ``derived`` section with
            ``cutoff`` and ``inner_cutoff``. Tuples become lists, paths strings.
        """
        result: dict[str, Any] = {"name": self.name}
        for section in SECTIONS:
            obj = getattr(self, section)
            result[section] = {
                f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)
            }
        result["derived"] = {"cutoff": self.grid.cutoff, "inner_cutoff": self.grid.inner_cutoff}
        return result

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        """Build a configuration from a dictionary such as :meth:`to_dict` output.

        Parameters
        ----------
        d : Mapping
            ``name`` and the sections ``grid``, ``data``, ``model`` and ``train``.
            A ``derived`` section is checked against the recomputed values.

        Returns
        -------
        ExperimentConfig
            The validated configuration. Paths are taken as given.

        Raises
        ------
        ValueError
            On unknown or missing keys, wrong types, invalid values or a ``derived``
            section that does not match the grid.
        """
        allowed = {"name", *SECTIONS, "derived"}
        unknown = sorted(set(d) - allowed)
        _require(not unknown, f"unknown top-level key(s) {unknown}; allowed: {sorted(allowed)}")
        _require("name" in d, "missing required key 'name'")
        _require(isinstance(d["name"], str), f"name must be a string, got {d['name']!r}")
        _require("data" in d, "missing required section [data]")
        sections: dict[str, Any] = {}
        for section in SECTIONS:
            raw = d.get(section, {})
            _require(
                isinstance(raw, Mapping), f"[{section}] must be a table, got {type(raw).__name__}"
            )
            sections[section] = _build_section(section, raw)
        config = cls(name=d["name"], **sections)
        derived = d.get("derived")
        if derived is not None:
            expected = config.to_dict()["derived"]
            _require(
                isinstance(derived, Mapping) and dict(derived) == expected,
                f"derived section {derived!r} does not match the grid ({expected!r})",
            )
        return config

    def with_overrides(self, overrides: Mapping[str, Any] | Sequence[str]) -> Self:
        """Return a copy with some values replaced.

        Parameters
        ----------
        overrides : Mapping or sequence of str
            Dotted keys (``"train.epochs"``) mapped to new values, or strings of the
            form ``"train.epochs=2"``; see :func:`parse_overrides`. Relative paths are
            kept as given (not resolved). Only :func:`load_config` resolves relative
            paths, also those from overrides, against the config file's directory.

        Returns
        -------
        ExperimentConfig
            The new, validated configuration.
        """
        d = self.to_dict()
        del d["derived"]
        _apply_overrides(d, parse_overrides(overrides))
        return type(self).from_dict(d)


# --------------------------------------------------------------------------- conversion


def _to_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_to_plain(v) for v in value]
    return value


def _type_name(hint: Any) -> str:
    return getattr(hint, "__name__", None) or str(hint).replace("typing.", "")


def _coerce(value: Any, hint: Any, key: str) -> Any:
    """Convert ``value`` to the field type ``hint`` or raise ``ValueError``."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is Literal:
        if value in args and isinstance(value, str):
            return value
        raise ValueError(f"{key} must be one of {args}, got {value!r}")
    if origin in (types.UnionType, typing.Union):
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, key)
            except ValueError:
                continue
        raise ValueError(f"{key}: {value!r} does not match {_type_name(hint)}")
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ValueError(f"{key} must be a list, got {value!r}")
        return tuple(_coerce(v, args[0], f"{key}[{i}]") for i, v in enumerate(value))
    if hint is bool:
        if isinstance(value, bool):
            return value
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif hint is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif hint is str:
        if isinstance(value, str):
            return value
    elif hint is Path:
        if isinstance(value, str | Path):
            return Path(value)
    else:  # pragma: no cover - guards against new field types
        raise TypeError(f"unsupported field type {hint!r} for {key}")
    raise ValueError(f"{key} must be of type {_type_name(hint)}, got {value!r}")


def _build_section(section: str, raw: Mapping[str, Any]) -> Any:
    cls = _SECTION_TYPES[section]
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(raw) - set(names))
    _require(not unknown, f"unknown key(s) in [{section}]: {unknown}; allowed: {sorted(names)}")
    kwargs = {k: _coerce(v, hints[k], f"{section}.{k}") for k, v in raw.items()}
    try:
        return cls(**kwargs)
    except TypeError as exc:  # missing required field
        raise ValueError(f"[{section}]: {exc}") from exc


# --------------------------------------------------------------------------- overrides


def _parse_value(text: str) -> Any:
    """Parse ``text`` as a TOML value; fall back to the plain string."""
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def parse_overrides(overrides: Mapping[str, Any] | Sequence[str] | None) -> dict[str, Any]:
    """Normalise overrides to a mapping of dotted keys to values.

    Parameters
    ----------
    overrides : Mapping, sequence of str or None
        A mapping is taken as is. Each string must look like ``"section.key=value"``
        (or ``"name=value"``); the value is parsed as a TOML literal
        (``2``, ``1e-3``, ``true``, ``[16, 32]``, ``"x"``) and otherwise kept as a
        string.

    Returns
    -------
    dict
        Dotted keys mapped to values.

    Raises
    ------
    ValueError
        If a string has no ``=`` or an empty key.
    """
    if overrides is None:
        return {}
    if isinstance(overrides, Mapping):
        return dict(overrides)
    if isinstance(overrides, str):
        raise ValueError("overrides must be a mapping or a sequence of 'key=value' strings")
    result: dict[str, Any] = {}
    for item in overrides:
        key, sep, text = item.partition("=")
        key = key.strip()
        _require(sep == "=" and key != "", f"override {item!r} is not of the form key=value")
        result[key] = _parse_value(text.strip())
    return result


def _apply_overrides(d: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        parts = key.split(".")
        if parts == ["name"]:
            d["name"] = value
            continue
        _require(
            len(parts) == 2 and parts[0] in SECTIONS,
            f"override key {key!r} must be 'name' or '<section>.<field>' with section in "
            f"{SECTIONS}",
        )
        section, name = parts
        fields = {f.name for f in dataclasses.fields(_SECTION_TYPES[section])}
        _require(name in fields, f"unknown override key {key!r}; [{section}] has {sorted(fields)}")
        d.setdefault(section, {})[name] = value


# --------------------------------------------------------------------------- loading


def _resolve(path: Any, base: Path) -> Any:
    if not isinstance(path, str | Path):
        return path  # left to type validation
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(os.path.normpath(base / p))


def load_config(
    path: str | Path, overrides: Mapping[str, Any] | Sequence[str] | None = None
) -> ExperimentConfig:
    """Load and validate an experiment configuration from a TOML file.

    Parameters
    ----------
    path : str or Path
        TOML file with ``name`` and the tables ``[grid]``, ``[data]``, ``[model]``
        and ``[train]``.
    overrides : Mapping, sequence of str or None
        Values that replace those of the file before validation; see
        :func:`parse_overrides`.

    Returns
    -------
    ExperimentConfig
        The configuration. Relative paths in ``data.sources``, ``data.output_dir``
        and ``train.run_root`` (also those given as overrides) are resolved relative
        to the directory of ``path``, not to the current working directory: with
        ``configs/a.toml`` the override ``train.run_root=runs`` gives
        ``configs/runs``. Callers that want CWD-relative override paths must make
        them absolute before passing them. Without ``data.output_dir`` the dataset goes
        to ``data/processed/<name>`` below that directory.

    Raises
    ------
    ValueError
        On unknown keys, wrong types or invalid values.
    """
    path = Path(path)
    with path.open("rb") as fh:
        d = tomllib.load(fh)
    _require("derived" not in d, f"{path}: 'derived' is computed and must not be set")
    _apply_overrides(d, parse_overrides(overrides))
    base = path.resolve().parent
    data = d.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("sources"), list):
            data["sources"] = [_resolve(s, base) for s in data["sources"]]
        if "output_dir" in data:
            data["output_dir"] = _resolve(data["output_dir"], base)
        elif isinstance(d.get("name"), str):
            data["output_dir"] = base / "data" / "processed" / d["name"]
    train = d.get("train")
    if isinstance(train, dict):
        train["run_root"] = _resolve(train.get("run_root", "runs"), base)
    else:
        d["train"] = {"run_root": base / "runs"}
    try:
        return ExperimentConfig.from_dict(d)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
