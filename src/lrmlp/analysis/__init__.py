"""Figures of trained runs (needs the ``plot`` extra, matplotlib).

* :mod:`lrmlp.analysis.plots`: learning curves from ``training_log.csv``
  (thesis Fig. 5.1, 5.2 and A.1-A.6), :func:`~lrmlp.analysis.plots.plot_training_log`.
* :mod:`lrmlp.analysis.weights`: weight distributions of two runs, e.g. with and
  without long range (thesis Fig. 5.3, p. 55),
  :func:`~lrmlp.analysis.weights.plot_weight_distribution`.

The figures are drawn on :class:`matplotlib.figure.Figure` objects with the Agg
canvas, without :mod:`matplotlib.pyplot`, so the global backend is never changed
and nothing is shown on screen. The submodules are imported on first use.
"""

from __future__ import annotations

__all__ = ["plots", "weights"]
