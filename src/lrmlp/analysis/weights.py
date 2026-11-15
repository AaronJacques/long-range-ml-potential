"""Weight distributions of two trained runs (thesis Fig. 5.3, p. 55).

Replaces ``plot_stacked_weight_distribution`` of
``Neural-Network/weigtht_distribution.py@9413bfd`` (B17). The figure is the same:
two step histograms ("With" / "Without" long range) of all weights of each model,
150 common bins, ``density=True``, only weights strictly inside the boundary
(−1, 1), font size 16, one PDF. What changes:

* the models are rebuilt from their run directories (:class:`lrmlp.run.RunDirectory`,
  :func:`lrmlp.model.build_model`, ``load_weights``) instead of loading ``.h5``
  files from hard-coded Windows paths, and each model is loaded once;
* the weights are flattened with one ``np.concatenate`` and filtered with a boolean
  mask instead of Python loops over every value.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lrmlp.analysis.plots import FONT_SIZE, new_figure
from lrmlp.run import RunDirectory

__all__ = [
    "flatten_weights",
    "load_run_weights",
    "plot_weight_distribution",
    "weight_histograms",
]

logger = logging.getLogger(__name__)


def flatten_weights(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate all weight arrays of a model into one flat float64 vector.

    Parameters
    ----------
    arrays : sequence of ndarray
        E.g. ``model.get_weights()`` (all layers, trainable and not, as the legacy
        ``layer.get_weights()`` loop).

    Returns
    -------
    ndarray, shape (n,)
        The values in layer order.
    """
    if not arrays:
        return np.zeros(0)
    return np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w in arrays])


def load_run_weights(run_dir: str | Path, epoch: str | int = "best") -> np.ndarray:
    """All weights of one checkpoint of a run, flattened.

    Parameters
    ----------
    run_dir : str or Path
        Run directory (layout in :mod:`lrmlp.run`).
    epoch : {"best", "last"} or int
        Checkpoint (:meth:`lrmlp.run.RunDirectory.resolve_epoch`); ``"best"`` is the
        epoch with the lowest validation MSE (p. 52), the model of Fig. 5.3.

    Returns
    -------
    ndarray, shape (n,)
        :func:`flatten_weights` of the model.
    """
    from lrmlp.model import build_model

    rd = RunDirectory(run_dir)
    config = rd.read_config()
    meta = rd.read_dataset_metadata()
    try:
        n_max_local, n_max_long = int(meta["n_max_local"]), int(meta["n_max_long_range"])
    except KeyError as err:
        raise ValueError(f"{rd.dataset_path} has no {err.args[0]!r}") from err
    resolved = rd.resolve_epoch(epoch)
    model = build_model(config.model, n_max_local, n_max_long, config.grid.max_atomic_number)
    model.load_weights(rd.checkpoint_path(resolved))
    weights = flatten_weights(model.get_weights())
    logger.info("%s epoch %d: %d weights", rd.path, resolved, weights.size)
    return weights


def _inside(weights: np.ndarray, boundary: tuple[float, float] | None) -> np.ndarray:
    if boundary is None:
        return weights
    lo, hi = boundary
    return weights[(weights > lo) & (weights < hi)]


def weight_histograms(
    weights: Sequence[np.ndarray],
    bins: int = 150,
    boundary: tuple[float, float] | None = (-1.0, 1.0),
    density: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Histograms of several weight vectors on common bins, as in Fig. 5.3.

    Same binning as ``matplotlib.pyplot.hist`` with a list of datasets: ``bins``
    equal-width bins over the range of all values that lie strictly inside
    ``boundary``.

    Parameters
    ----------
    weights : sequence of ndarray
        One flat weight vector per model.
    bins : int
        Number of bins (150 in Fig. 5.3).
    boundary : (float, float) or None
        Keep only weights with ``lo < w < hi``; ``None`` keeps all.
    density : bool
        Normalise each histogram to unit area.

    Returns
    -------
    counts : list of ndarray, shape (bins,)
        One histogram per input.
    edges : ndarray, shape (bins + 1,)
        The common bin edges.
    """
    kept = [_inside(np.asarray(w, dtype=np.float64).ravel(), boundary) for w in weights]
    edges = np.histogram_bin_edges(np.concatenate(kept) if kept else np.zeros(0), bins=bins)
    counts = [np.histogram(w, bins=edges, density=density)[0] for w in kept]
    return counts, edges


def plot_weight_distribution(
    run_a: str | Path,
    run_b: str | Path,
    output: str | Path,
    epoch_a: str | int = "best",
    epoch_b: str | int = "best",
    boundary: tuple[float, float] | None = (-1.0, 1.0),
    bins: int = 150,
    labels: tuple[str, str] = ("With", "Without"),
    title: str | None = None,
) -> Path:
    """Plot the weight distributions of two runs (thesis Fig. 5.3, p. 55).

    Parameters
    ----------
    run_a, run_b : str or Path
        Run directories, in Fig. 5.3 the model with and the one without long range.
    output : str or Path
        Output figure (PDF); parent directories are created.
    epoch_a, epoch_b : {"best", "last"} or int
        Checkpoints of the two runs.
    boundary : (float, float) or None
        Only weights strictly inside are shown, and the x axis is limited to it.
    bins : int
        Number of common bins.
    labels : (str, str)
        Legend entries of the two runs.
    title : str, optional
        Defaults to ``"Weight Distribution <name of run_a>"``.

    Returns
    -------
    Path
        ``output``.
    """
    weights = [load_run_weights(run_a, epoch_a), load_run_weights(run_b, epoch_b)]
    if title is None:
        title = f"Weight Distribution {RunDirectory(run_a).read_config().name}"
    _, edges = weight_histograms(weights, bins, boundary)
    kept = [_inside(w, boundary) for w in weights]

    fig, ax = new_figure((6, 5.5))
    ax.hist(kept, bins=edges, density=True, histtype="step", label=list(labels))
    if boundary is not None:
        ax.set_xlim(boundary)
    ax.legend(loc="upper right", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.set_xlabel("Weight Value", fontsize=FONT_SIZE)
    ax.set_ylabel("Density", fontsize=FONT_SIZE)
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=4)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    return out
