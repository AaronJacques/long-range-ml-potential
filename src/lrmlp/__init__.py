"""Machine learning potentials with long-range interaction.

``lrmlp`` is the code accompanying the master's thesis *Machine Learning Potentials
with Long Range Interaction*. It predicts the potential energy of a molecule from
per-atom descriptors built on a hierarchical cell grid: a local descriptor matrix
for neighbouring atoms within the cutoff and a long-range part made of cell
centroids and atom counts of the remaining cells. The descriptors are passed
through an embedding network and summed over atoms.

The public API is exported lazily, so ``import lrmlp`` does not import TensorFlow.
The submodule that defines a name is only imported on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

#: Maps every lazily exported public name to the submodule that defines it.
_LAZY_EXPORTS: dict[str, str] = {
    "ExperimentConfig": "lrmlp.config",
    "GridConfig": "lrmlp.config",
    "DataConfig": "lrmlp.config",
    "ModelConfig": "lrmlp.config",
    "TrainConfig": "lrmlp.config",
    "load_config": "lrmlp.config",
    "compute_descriptors": "lrmlp.descriptors",
    "build_dataset": "lrmlp.dataset",
    "load_dataset": "lrmlp.dataset",
    "build_model": "lrmlp.model",
    "train": "lrmlp.training",
    "evaluate": "lrmlp.evaluation",
    "Potential": "lrmlp.potential",
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    """Resolve a lazily exported public name.

    Parameters
    ----------
    name : str
        Attribute name requested on the package.

    Returns
    -------
    Any
        The object ``name`` from its defining submodule. It is cached in the
        package namespace, so later lookups bypass this function.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the public API.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List module attributes including the lazily exported names.

    Returns
    -------
    list of str
        Sorted names of the eager module globals and the public API.
    """
    return sorted(set(globals()) | set(__all__))
