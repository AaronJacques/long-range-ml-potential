"""Learning curves of a training run (thesis Fig. 5.1, 5.2 and A.1-A.6).

Replaces the functions ``plot_loss`` and ``plot_energy_loss`` of
``Neural-Network/testing_notebook.ipynb@9413bfd`` (cells 2-3). The figure style is
kept (energy MSE of training and validation over the epochs, font size 16, legend
upper right, the first ``skip`` epochs left out, one PDF per figure); what changes:

* both log formats are read with :mod:`csv` (no pandas): the current columns of
  :func:`lrmlp.training.log_columns` and the legacy ones (``Epoch``, ``Energy Loss``,
  ``Val Energy Loss``, ...), see :data:`LEGACY_COLUMNS`;
* the title states "with" or "without Long-Range" according to
  ``model.use_long_range`` of the run (the notebook always wrote "without", B18) or
  is given explicitly;
* force, total-loss and loss-weight figures are drawn only if the log has them;
* paths are portable, the default output directory is ``<run>/plots/``.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lrmlp.run import RunDirectory

__all__ = [
    "FONT_SIZE",
    "LEGACY_COLUMNS",
    "TrainingLog",
    "energy_title",
    "new_figure",
    "plot_training_log",
    "read_training_log",
]

logger = logging.getLogger(__name__)

#: Font size of the thesis figures (notebook cell 3).
FONT_SIZE = 16
#: Legacy column of ``training_log.csv`` (``Constants.Logging``) → current column.
LEGACY_COLUMNS: dict[str, str] = {
    "Epoch": "epoch",
    "Train Total Loss": "train_loss",
    "Energy Loss": "train_energy_mse",
    "Force Loss": "train_force_mse",
    "Val Total Loss": "val_loss",
    "Val Energy Loss": "val_energy_mse",
    "Val Force Loss": "val_force_mse",
    "P Energy": "p_energy",
    "P Force": "p_force",
}
#: Axis label of the squared energy errors.
ENERGY_LABEL = "MSE (kcal/mol)²"
#: Axis label of the squared force errors.
FORCE_LABEL = "MSE (kcal/(mol·Å))²"


@dataclass(frozen=True)
class TrainingLog:
    """Columns of a ``training_log.csv`` under their current names.

    Attributes
    ----------
    columns : dict of str to ndarray
        Column name (:func:`lrmlp.training.log_columns`) → float values, one per
        epoch. Columns without any finite value (the NaN columns of legacy logs,
        B07) are left out.
    legacy : bool
        Whether the file had the legacy header.
    """

    columns: dict[str, np.ndarray]
    legacy: bool

    def __contains__(self, name: object) -> bool:
        """Whether the log has column ``name``."""
        return name in self.columns

    def __getitem__(self, name: str) -> np.ndarray:
        """Values of column ``name``."""
        return self.columns[name]

    def __len__(self) -> int:
        """Return the number of epochs (rows)."""
        return len(self.columns["epoch"])


def _float(text: str) -> float:
    text = text.strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def read_training_log(path: str | Path, skip: int = 0) -> TrainingLog:
    """Read a training log in the current or the legacy format.

    Parameters
    ----------
    path : str or Path
        ``training_log.csv``.
    skip : int
        Leave out the first ``skip`` epochs (rows), as ``log[skip:]`` in the notebook.

    Returns
    -------
    TrainingLog
        The columns under the current names.

    Raises
    ------
    ValueError
        If there is no epoch column, ``skip`` is negative or no row remains.
    """
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader, [])]
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    legacy = "Epoch" in header and "epoch" not in header
    names = [LEGACY_COLUMNS.get(h, h) if legacy else h for h in header]
    if "epoch" not in names:
        raise ValueError(f"{path}: no epoch column in the header {header}")
    rows = rows[skip:]
    if not rows:
        raise ValueError(f"{path}: no epochs left after skipping {skip}")
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        values = np.array([_float(row[i]) if i < len(row) else math.nan for row in rows])
        if np.isfinite(values).any():
            columns[name] = values
    if "epoch" not in columns:
        raise ValueError(f"{path}: the epoch column has no numbers")
    return TrainingLog(columns, legacy)


def energy_title(name: str | None, use_long_range: bool | None) -> str:
    """Title of the energy figure, as in thesis Fig. 5.1/5.2.

    Parameters
    ----------
    name : str or None
        Molecule / experiment name.
    use_long_range : bool or None
        ``model.use_long_range`` of the run; ``None`` if unknown.

    Returns
    -------
    str
        ``"Energy Loss <name> with Long-Range"`` or ``"... without Long-Range"``
        (B18: the notebook wrote "without" regardless of the flag).
    """
    parts = ["Energy Loss"]
    if name:
        parts.append(name)
    if use_long_range is not None:
        parts.append("with Long-Range" if use_long_range else "without Long-Range")
    return " ".join(parts)


def new_figure(size: tuple[float, float]) -> tuple[Any, Any]:
    """Create a figure on the Agg canvas, without pyplot and its global state.

    Parameters
    ----------
    size : (float, float)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        Its single axes.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot()


def _curve_figure(
    log: TrainingLog,
    series: Sequence[tuple[str, str]],
    ylabel: str,
    title: str | None,
    path: Path,
    *,
    thesis_style: bool = False,
) -> Path:
    fig, ax = new_figure((10, 5))
    epoch = log["epoch"]
    for column, label in series:
        if column in log:
            ax.plot(epoch, log[column], label=label)
    size = FONT_SIZE if thesis_style else None
    ax.set_xlabel("Epoch", fontsize=size)
    ax.set_ylabel(ylabel, fontsize=size)
    if title:
        ax.set_title(title, fontsize=size)
    if thesis_style:
        ax.tick_params(labelsize=FONT_SIZE - 1)
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=4)
        ax.legend(loc="upper right", fontsize=FONT_SIZE)
    else:
        ax.legend()
    fig.savefig(path, bbox_inches="tight")
    return path


def _run_info(source: Path) -> tuple[Path, Path, Mapping[str, Any] | None]:
    """Return (log file, default output directory, config dict or None)."""
    if source.is_dir():
        rd = RunDirectory(source)
        log_path, out = rd.log_path, rd.path / "plots"
        config_dir = rd.path
    else:
        log_path, out = source, source.parent / "plots"
        config_dir = source.parent
    if not log_path.is_file():
        raise FileNotFoundError(f"training log {log_path} not found")
    config_path = config_dir / RunDirectory.CONFIG_FILE
    config = None
    if config_path.is_file():
        from lrmlp.run import read_json

        data = read_json(config_path)
        config = data if isinstance(data, Mapping) else None
    return log_path, out, config


def plot_training_log(
    source: str | Path,
    output_dir: str | Path | None = None,
    skip: int = 0,
    title: str | None = None,
) -> list[Path]:
    """Plot the learning curves of a run (thesis Fig. 5.1, 5.2 and A.1-A.6).

    Parameters
    ----------
    source : str or Path
        Run directory or a ``training_log.csv`` (current or legacy header). The
        title and file names use ``config.json`` of the run directory (or next to
        the CSV) if present.
    output_dir : str or Path, optional
        Directory of the PDFs; defaults to ``<run>/plots/`` (next to the CSV).
    skip : int
        Leave out the first ``skip`` epochs, which dominate the scale (the thesis
        figures skip 20 resp. 30).
    title : str, optional
        Title of the energy figure; defaults to :func:`energy_title` of the run's
        ``name`` and ``model.use_long_range`` (B18).

    Returns
    -------
    list of Path
        The written PDFs: always ``<prefix>energy_mse.pdf`` (training and validation
        energy MSE in (kcal/mol)², Eq. 4.21); ``total_loss.pdf`` if the log has a
        total loss that differs from the energy MSE (force mode, Eq. 4.10);
        ``force_mse.pdf`` and ``loss_weights.pdf`` (p_E, p_F) if the log has them.
        ``<prefix>`` is ``<name>_long_range_`` / ``<name>_no_long_range_`` when the
        configuration is known, otherwise empty.

    Raises
    ------
    FileNotFoundError
        If the log does not exist.
    ValueError
        If the log has no energy MSE column or no epoch remains after ``skip``.
    """
    log_path, default_out, config = _run_info(Path(source))
    log = read_training_log(log_path, skip)
    if "train_energy_mse" not in log and "val_energy_mse" not in log:
        raise ValueError(f"{log_path}: no energy MSE column")
    name: str | None = None
    use_long_range: bool | None = None
    prefix = ""
    if config is not None:
        name = config.get("name") if isinstance(config.get("name"), str) else None
        model = config.get("model")
        if isinstance(model, Mapping) and isinstance(model.get("use_long_range"), bool):
            use_long_range = model["use_long_range"]
        if name is not None and use_long_range is not None:
            prefix = f"{name}_{'long_range' if use_long_range else 'no_long_range'}_"
    out = Path(output_dir) if output_dir is not None else default_out
    out.mkdir(parents=True, exist_ok=True)

    written = [
        _curve_figure(
            log,
            [("train_energy_mse", "Train MSE Loss"), ("val_energy_mse", "Validation MSE Loss")],
            ENERGY_LABEL,
            title if title is not None else energy_title(name, use_long_range),
            out / f"{prefix}energy_mse.pdf",
            thesis_style=True,
        )
    ]
    has_total = "train_loss" in log and not (
        "train_energy_mse" in log and np.array_equal(log["train_loss"], log["train_energy_mse"])
    )
    if has_total:
        written.append(
            _curve_figure(
                log,
                [("train_loss", "Train Total Loss"), ("val_loss", "Validation Total Loss")],
                "Loss",
                None,
                out / f"{prefix}total_loss.pdf",
            )
        )
    if "train_force_mse" in log or "val_force_mse" in log:
        written.append(
            _curve_figure(
                log,
                [("train_force_mse", "Train Force MSE"), ("val_force_mse", "Validation Force MSE")],
                FORCE_LABEL,
                None,
                out / f"{prefix}force_mse.pdf",
            )
        )
    if "p_energy" in log or "p_force" in log:
        written.append(
            _curve_figure(
                log,
                [("p_energy", "P Energy"), ("p_force", "P Force")],
                "Weight",
                None,
                out / f"{prefix}loss_weights.pdf",
            )
        )
    for path in written:
        logger.debug("wrote %s", path)
    return written
