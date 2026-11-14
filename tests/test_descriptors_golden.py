"""Golden regression of ``lrmlp.descriptors`` against the legacy ``create_matrices``.

The fixtures (T02) hold the legacy rows per frame in legacy atom order; ``order``
maps each legacy position to the input atom index (B01).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lrmlp.config import GridConfig
from lrmlp.descriptors import compute_descriptors, max_neighbour_counts, pad_descriptors

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

pytestmark = pytest.mark.golden

CASES = ("GA_aspirin", "GB_aspirin", "GB_aspirin_sorted", "GA_edge", "GB_edge", "GC_stachyose")


def _frames(case: str) -> list[tuple[GridConfig, dict[str, np.ndarray]]]:
    golden = fx.load_descriptor_golden(case)
    grid = GridConfig(
        grid_size=float(golden["grid_size"]),
        max_local_level=int(golden["max_local_level"]),
        assignment="legacy",
        sort_neighbours=bool(golden["sort_matrix"]),
    )
    assert grid.cutoff == float(golden["cut_off"])
    assert grid.inner_cutoff == float(golden["inner_cut_off"])
    return [(grid, fx.frame(golden, f)) for f in range(int(golden["n_frames"]))]


def _legacy_pad(per_atom: list[np.ndarray], n_max: int, width: int) -> np.ndarray:
    """``pad_df_entry`` (``Dataset.py:391-404``) for a non-empty molecule."""
    padded = np.zeros((len(per_atom), n_max, width), dtype=np.float32)
    for i, x in enumerate(per_atom):
        if len(x) > 0:
            padded[i, : len(x), :] = x.astype(np.float32)
    return padded


@pytest.mark.parametrize("case", CASES)
def test_rows_match_golden(case: str) -> None:
    for grid, fr in _frames(case):
        desc = compute_descriptors(fr["positions"], fr["numbers"], grid)
        local = fx.split_ragged(fr["local_rows"], fr["local_counts"])
        local_z = fx.split_ragged(fr["local_numbers"], fr["local_counts"])
        long_range = fx.split_ragged(fr["lr_rows"], fr["lr_counts"])
        features = fx.split_ragged(fr["lr_features"], fr["lr_counts"])
        for k, i in enumerate(fr["order"]):
            atom = desc.atom(int(i))
            msg = f"{case} atom {i} (legacy position {k})"
            np.testing.assert_allclose(
                atom.local_matrix, local[k].reshape(-1, 4), rtol=1e-12, atol=1e-12, err_msg=msg
            )
            np.testing.assert_array_equal(atom.local_numbers, local_z[k].reshape(-1, 2), msg)
            np.testing.assert_allclose(
                atom.long_range_matrix,
                long_range[k].reshape(-1, 4),
                rtol=1e-12,
                atol=1e-12,
                err_msg=msg,
            )
            np.testing.assert_array_equal(
                atom.long_range_features, features[k].reshape(-1, 101), msg
            )
        assert max_neighbour_counts(desc) == tuple(int(v) for v in fr["n_max"])


@pytest.mark.parametrize("case", CASES)
def test_padded_arrays_match_legacy_padding(case: str) -> None:
    for grid, fr in _frames(case):
        desc = compute_descriptors(fr["positions"], fr["numbers"], grid)
        n_loc, n_lr = (int(v) for v in fr["n_max"])
        padded = pad_descriptors(desc, n_loc, n_lr)
        order = fr["order"]
        legacy = {
            "local_matrix": ("local_rows", "local_counts", n_loc, 4),
            "local_numbers": ("local_numbers", "local_counts", n_loc, 2),
            "long_range_matrix": ("lr_rows", "lr_counts", n_lr, 4),
            "long_range_features": ("lr_features", "lr_counts", n_lr, 101),
        }
        for key, (rows, counts, n_max, width) in legacy.items():
            per_atom = fx.split_ragged(fr[rows], fr[counts])
            want = _legacy_pad([x.reshape(-1, width) for x in per_atom], n_max, width)
            got = padded[key][order]  # input order → legacy atom order
            np.testing.assert_allclose(got.astype(np.float32), want, rtol=0, atol=1e-6, err_msg=key)


def test_gb_edge_has_atom_without_rows() -> None:
    """GB_edge/gb_assignment contains an atom with 0 local and 0 long-range rows (B13)."""
    found = False
    for grid, fr in _frames("GB_edge"):
        desc = compute_descriptors(fr["positions"], fr["numbers"], grid)
        empty = np.flatnonzero((desc.n_local == 0) & (desc.n_long == 0))
        if empty.size:
            found = True
            padded = pad_descriptors(desc, *max_neighbour_counts(desc))
            for value in padded.values():
                assert not value[empty].any()
    assert found
