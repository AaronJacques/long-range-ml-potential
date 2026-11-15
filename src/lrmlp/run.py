"""Run directory contract.

A training run writes everything into one directory
``<run_root>/<name>-YYYYmmdd-HHMMSS[-n]/``::

    config.json                      resolved ExperimentConfig (to_dict)
    dataset.json                     copy of the dataset metadata plus its path
    training_log.csv                 one row per epoch
    tb/                              TensorBoard event files (only if tensorboard is installed)
    checkpoints/epoch_0001.weights.h5
    best.json                        {"epoch": ..., "val_energy_mse": ...}
    evaluation_<split>_<epoch>.json  evaluation results

Existing run directories are never overwritten. JSON files are UTF-8 encoded with
sorted keys. The module uses the standard library only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import secrets
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from lrmlp.config import ExperimentConfig

__all__ = ["RunDirectory", "new_run_dir", "read_json", "tensorboard_available", "write_json"]

_CHECKPOINT_RE = re.compile(r"^epoch_(\d{4,})\.weights\.h5$")


def tensorboard_available() -> bool:
    """Return whether the optional ``tensorboard`` package is installed.

    TensorBoard comes with the extra ``tune``. Without it, training writes no
    ``tb/`` directory and ``lrmlp tune`` refuses to run (``--dry-run`` still works).
    """
    try:
        return importlib.util.find_spec("tensorboard") is not None
    except (ImportError, ValueError):
        return False


def write_json(path: str | Path, data: Any) -> Path:
    """Write ``data`` as UTF-8 JSON with sorted keys, atomically.

    The temporary file is created with mode ``0o666``, so the operating system
    applies the current umask and the result gets the same permissions as a
    plainly created file (not the ``0o600`` of :func:`tempfile.mkstemp`).

    Parameters
    ----------
    path : str or Path
        Target file; it is replaced via a temporary file in the same directory.
    data : Any
        JSON-serialisable object.

    Returns
    -------
    Path
        The written file.
    """
    path = Path(path)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file.

    Parameters
    ----------
    path : str or Path
        File to read.

    Returns
    -------
    Any
        The decoded object.
    """
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def new_run_dir(run_root: str | Path, name: str, *, now: datetime | None = None) -> Path:
    """Create a new, unique run directory.

    Parameters
    ----------
    run_root : str or Path
        Parent directory; created if missing.
    name : str
        Experiment name, the prefix of the directory name.
    now : datetime, optional
        Timestamp to use instead of the current local time.

    Returns
    -------
    Path
        The created, empty directory ``<run_root>/<name>-YYYYmmdd-HHMMSS``. If that
        already exists, ``-1``, ``-2``, ... is appended. Creation is atomic, so two
        runs started in the same second never share a directory.

    Raises
    ------
    ValueError
        If ``name`` is empty or contains a path separator.
    """
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"run name must be usable as a directory name, got {name!r}")
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{name}-{(now or datetime.now()).strftime('%Y%m%d-%H%M%S')}"
    n = 0
    while True:
        candidate = root / (stem if n == 0 else f"{stem}-{n}")
        try:
            candidate.mkdir()
        except FileExistsError:
            n += 1
            continue
        return candidate


class RunDirectory:
    """Paths and metadata files of one run directory.

    Parameters
    ----------
    path : str or Path
        The run directory. It need not exist until :meth:`create` is called.
    """

    CONFIG_FILE = "config.json"
    DATASET_FILE = "dataset.json"
    LOG_FILE = "training_log.csv"
    BEST_FILE = "best.json"
    TB_DIR = "tb"
    CHECKPOINT_DIR = "checkpoints"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        """Return ``RunDirectory('<path>')``."""
        return f"{type(self).__name__}({str(self.path)!r})"

    # ------------------------------------------------------------------ paths

    @property
    def config_path(self) -> Path:
        """Resolved experiment configuration, ``config.json``."""
        return self.path / self.CONFIG_FILE

    @property
    def dataset_path(self) -> Path:
        """Dataset metadata, ``dataset.json``."""
        return self.path / self.DATASET_FILE

    @property
    def log_path(self) -> Path:
        """Per-epoch training log, ``training_log.csv``."""
        return self.path / self.LOG_FILE

    @property
    def best_path(self) -> Path:
        """Best epoch by validation MSE, ``best.json``."""
        return self.path / self.BEST_FILE

    @property
    def tb_dir(self) -> Path:
        """TensorBoard directory, ``tb/``."""
        return self.path / self.TB_DIR

    @property
    def checkpoint_dir(self) -> Path:
        """Checkpoint directory, ``checkpoints/``."""
        return self.path / self.CHECKPOINT_DIR

    def checkpoint_path(self, epoch: int) -> Path:
        """Return the weights file ``checkpoints/epoch_{epoch:04d}.weights.h5``."""
        return self.checkpoint_dir / f"epoch_{_check_epoch(epoch):04d}.weights.h5"

    def evaluation_path(self, split: str, epoch: int) -> Path:
        """Return the results file ``evaluation_<split>_<epoch>.json``."""
        if not split or "/" in split or "\\" in split:
            raise ValueError(f"invalid split name {split!r}")
        return self.path / f"evaluation_{split}_{_check_epoch(epoch)}.json"

    # ------------------------------------------------------------------ files

    def create(
        self,
        config: ExperimentConfig | Mapping[str, Any],
        dataset_metadata: Mapping[str, Any],
        *,
        tensorboard: bool = True,
    ) -> RunDirectory:
        """Create the layout and write ``config.json`` and ``dataset.json``.

        Parameters
        ----------
        config : ExperimentConfig or Mapping
            Resolved configuration; an :class:`ExperimentConfig` is stored via
            :meth:`ExperimentConfig.to_dict`.
        dataset_metadata : Mapping
            Copy of the dataset metadata including the dataset path.
        tensorboard : bool, default True
            Create ``tb/``. ``False`` when TensorBoard is not installed.

        Returns
        -------
        RunDirectory
            ``self``.

        Raises
        ------
        FileExistsError
            If the directory exists and is not empty; runs are never overwritten.
        """
        if self.path.exists() and (not self.path.is_dir() or any(self.path.iterdir())):
            raise FileExistsError(f"run directory {self.path} already exists and is not empty")
        self.path.mkdir(parents=True, exist_ok=True)
        if tensorboard:
            self.tb_dir.mkdir()
        self.checkpoint_dir.mkdir()
        data = config.to_dict() if isinstance(config, ExperimentConfig) else dict(config)
        write_json(self.config_path, data)
        write_json(self.dataset_path, dict(dataset_metadata))
        return self

    def read_config(self) -> ExperimentConfig:
        """Read ``config.json`` back into an :class:`ExperimentConfig`."""
        return ExperimentConfig.from_dict(read_json(self._existing(self.config_path)))

    def read_dataset_metadata(self) -> dict[str, Any]:
        """Read ``dataset.json``."""
        data: dict[str, Any] = read_json(self._existing(self.dataset_path))
        return data

    def write_best(self, epoch: int, metrics: Mapping[str, float]) -> Path:
        """Write ``best.json`` with ``epoch`` and ``metrics`` (e.g. ``val_energy_mse``)."""
        if "epoch" in metrics:
            raise ValueError("metrics must not contain the key 'epoch'")
        return write_json(self.best_path, {"epoch": _check_epoch(epoch), **metrics})

    def read_best(self) -> dict[str, Any]:
        """Read ``best.json``."""
        data: dict[str, Any] = read_json(self._existing(self.best_path))
        return data

    def list_checkpoints(self) -> list[int]:
        """Epochs with a weights file in ``checkpoints/``, in ascending order."""
        if not self.checkpoint_dir.is_dir():
            return []
        epochs = []
        for p in self.checkpoint_dir.iterdir():
            match = _CHECKPOINT_RE.match(p.name)
            if match and p.is_file():
                epochs.append(int(match.group(1)))
        return sorted(epochs)

    def resolve_epoch(self, epoch: str | int) -> int:
        """Turn ``"best"``, ``"last"`` or an epoch number into an existing epoch.

        Parameters
        ----------
        epoch : {"best", "last"} or int
            ``"best"`` reads ``best.json``, ``"last"`` takes the newest checkpoint.
            Integers (also as digit strings) are checked against the checkpoints.

        Returns
        -------
        int
            An epoch whose checkpoint exists.

        Raises
        ------
        FileNotFoundError
            If ``best.json`` or the requested checkpoint is missing.
        ValueError
            If ``epoch`` is neither ``"best"``, ``"last"`` nor an integer.
        """
        available = self.list_checkpoints()
        if epoch == "best":
            if not self.best_path.is_file():
                raise FileNotFoundError(
                    f"no best.json in run directory {self.path}; the run has not finished an "
                    "epoch yet"
                )
            value = self.read_best().get("epoch")
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{self.best_path} has no integer 'epoch': {value!r}")
        elif epoch == "last":
            if not available:
                raise FileNotFoundError(f"no checkpoints in {self.checkpoint_dir}")
            return available[-1]
        elif isinstance(epoch, int) and not isinstance(epoch, bool):
            value = epoch
        elif isinstance(epoch, str) and epoch.isdigit():
            value = int(epoch)
        else:
            raise ValueError(f"epoch must be 'best', 'last' or an integer, got {epoch!r}")
        if value not in available:
            raise FileNotFoundError(
                f"no checkpoint for epoch {value} in {self.checkpoint_dir}; "
                f"available epochs: {available or 'none'}"
            )
        return value

    def _existing(self, path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} not found in run directory {self.path}")
        return path


def _check_epoch(epoch: int) -> int:
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}")
    return epoch
