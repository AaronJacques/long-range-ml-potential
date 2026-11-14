"""Integrity of the committed data golden fixtures (runs without the legacy env)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from golden import fixtures as fx
    from golden import synthetic
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx
    from golden import synthetic

MANIFEST = fx.load_manifest()
DESCRIPTOR_CASES = sorted(MANIFEST["descriptors"])
DATASET_CASES = sorted(MANIFEST["datasets"])
SPLITS = ("train", "val", "test")


def _frames(golden: dict) -> list[dict]:
    return [fx.frame(golden, f) for f in range(int(golden["n_frames"]))]


# --------------------------------------------------------------------------- files


@pytest.mark.parametrize("rel", sorted(MANIFEST["files"]))
def test_file_present_with_manifest_sha256(rel: str) -> None:
    path = fx.fixtures_dir() / rel
    assert path.is_file(), rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST["files"][rel]["sha256"]


def test_manifest_lists_all_golden_files() -> None:
    on_disk = {f"golden/{p.name}" for p in fx.golden_dir().glob("data_*.npz")}
    assert on_disk == {k for k in MANIFEST["files"] if k.startswith("golden/")}
    assert {"synthetic_small.npz", "stachyose_like.npz", "edge_cases.npz"} <= set(MANIFEST["files"])
    assert len(MANIFEST["legacy_commit"]) == 40


def test_fixture_size_below_limit() -> None:
    total = sum(p.stat().st_size for p in fx.fixtures_dir().rglob("*") if p.is_file())
    assert total < 5 * 1024 * 1024


def test_golden_configs() -> None:
    cfg = MANIFEST["configs"]
    assert (cfg["G-A"]["GRID_SIZE"], cfg["G-A"]["MAX_LOCAL_LEVEL"]) == (1.0, 6)
    assert (cfg["G-B"]["GRID_SIZE"], cfg["G-B"]["MAX_LOCAL_LEVEL"]) == (1.0, 2)
    assert (cfg["G-C"]["GRID_SIZE"], cfg["G-C"]["MAX_LOCAL_LEVEL"]) == (2.0, 4)
    for c in cfg.values():
        assert c["CUT_OFF"] == (c["MAX_LOCAL_LEVEL"] + 1 + 0.5**0.5) * c["GRID_SIZE"]
        assert c["INNER_CUT_OFF"] == c["CUT_OFF"] - 0.2
        assert c["MAX_ATOM_ELEMENTS"] == 100 and c["use_charge_number"] is True


# --------------------------------------------------------------------------- synthetic inputs


def test_synthetic_inputs_shapes_and_constraints() -> None:
    for name, n_frames, n_atoms in (("synthetic_small", 60, 21), ("stachyose_like", 5, 87)):
        d = fx.load_synthetic(name)
        assert d["E"].shape == (n_frames,)
        assert d["R"].shape == d["F"].shape == (n_frames, n_atoms, 3)
        assert d["z"].shape == (n_atoms,)
        assert np.all(d["R"] == np.round(d["R"] * 1024) / 1024)
        for R in d["R"]:
            dist = np.linalg.norm(R[:, None] - R[None], axis=-1)
            assert dist[np.triu_indices(n_atoms, 1)].min() >= 1.0
    z = fx.load_synthetic("synthetic_small")["z"]
    assert list(z) == [6] * 9 + [1] * 8 + [8] * 4


def test_synthetic_inputs_reproducible_from_generator() -> None:
    small = synthetic.aspirin_like(60, seed=0)
    stored = fx.load_synthetic("synthetic_small")
    for key in ("E", "F", "R", "z"):
        assert np.array_equal(small[key], stored[key]), key
    edge = fx.load_edge_cases()
    for name, case in synthetic.edge_cases().items():
        assert np.array_equal(edge[name]["R"], case["R"])
        assert np.array_equal(edge[name]["z"], case["z"])


def test_lennard_jones_forces_are_negative_gradient() -> None:
    R = fx.load_synthetic("synthetic_small")["R"][0]
    _, F = synthetic.lennard_jones(R)
    h = 1e-6
    for i, k in ((0, 0), (5, 1), (20, 2)):
        Rp, Rm = R.copy(), R.copy()
        Rp[i, k] += h
        Rm[i, k] -= h
        num = -(synthetic.lennard_jones(Rp)[0] - synthetic.lennard_jones(Rm)[0]) / (2 * h)
        assert num == pytest.approx(F[i, k], rel=1e-5, abs=1e-7)


def test_edge_cases_required() -> None:
    required = {
        "level_lmax2_within_rc",
        "dropped_pairs",
        "partial_cell",
        "single_cell",
        "no_long_range",
        "multi_element",
    }
    assert required <= set(fx.load_edge_cases())


# --------------------------------------------------------------------------- descriptor golden


@pytest.mark.parametrize("case", DESCRIPTOR_CASES)
def test_descriptor_golden_consistency(case: str) -> None:
    g = fx.load_descriptor_golden(case)
    n_max = np.zeros(2, dtype=np.int64)
    for fr in _frames(g):
        n = len(fr["positions"])
        assert np.array_equal(np.sort(fr["order"]), np.arange(n)), "order is a permutation"
        assert fr["local_counts"].shape == fr["lr_counts"].shape == (n,)
        assert fr["local_counts"].sum() == len(fr["local_rows"]) == len(fr["local_numbers"])
        assert fr["lr_counts"].sum() == len(fr["lr_rows"]) == len(fr["lr_features"])
        assert fr["local_rows"].shape[1:] == (4,) and fr["local_rows"].dtype == np.float64
        assert fr["lr_rows"].shape[1:] == (4,) and fr["lr_rows"].dtype == np.float64
        assert fr["local_numbers"].shape[1:] == (2,)
        assert fr["lr_features"].shape[1:] == (101,)
        assert list(fr["n_max"]) == [fr["local_counts"].max(), fr["lr_counts"].max()]
        # first column of Z / F is the central atom's atomic number (legacy row order)
        z_legacy = fr["numbers"][fr["order"]]
        assert np.array_equal(fr["local_numbers"][:, 0], np.repeat(z_legacy, fr["local_counts"]))
        assert np.array_equal(fr["lr_features"][:, 0], np.repeat(z_legacy, fr["lr_counts"]))
        n_max = np.maximum(n_max, fr["n_max"])
    assert list(n_max) == MANIFEST["descriptors"][case]["n_max"]


@pytest.mark.parametrize("case", DESCRIPTOR_CASES)
def test_legacy_row_order_is_cell_traversal(case: str) -> None:
    g = fx.load_descriptor_golden(case)
    for fr in _frames(g):
        expected = fx.legacy_traversal_order(fr["positions"], float(g["grid_size"]))
        assert np.array_equal(fr["order"], expected)


@pytest.mark.parametrize("case", DESCRIPTOR_CASES)
def test_counts_match_independent_legacy_partition(case: str) -> None:
    g = fx.load_descriptor_golden(case)
    args = (float(g["grid_size"]), int(g["max_local_level"]), float(g["cut_off"]))
    for fr in _frames(g):
        local, lr = fx.legacy_expected_counts(fr["positions"], *args)
        assert fx.to_input_order(list(fr["local_counts"]), fr["order"]) == list(local)
        assert fx.to_input_order(list(fr["lr_counts"]), fr["order"]) == list(lr)


def test_sorted_variant_is_permutation_of_unsorted() -> None:
    a = fx.load_descriptor_golden("GB_aspirin")
    b = fx.load_descriptor_golden("GB_aspirin_sorted")
    for fa, fb in zip(_frames(a), _frames(b), strict=True):
        for rows, counts in (("local_rows", "local_counts"), ("lr_rows", "lr_counts")):
            assert np.array_equal(fa[counts], fb[counts])
            for ra, rb in zip(
                fx.split_ragged(fa[rows], fa[counts]),
                fx.split_ragged(fb[rows], fb[counts]),
                strict=True,
            ):
                assert np.array_equal(np.sort(ra, axis=0), np.sort(rb, axis=0))
                assert np.all(np.diff(rb[:, 0]) <= 0), "sorted descending by first column"


@pytest.mark.parametrize("case", ["GA_edge", "GB_edge"])
def test_edge_cases_show_a01_and_a02(case: str) -> None:
    g = fx.load_descriptor_golden(case)
    args = (float(g["grid_size"]), int(g["max_local_level"]), float(g["cut_off"]))
    a01 = a02 = 0
    zero_local_with_neighbour = False
    for fr in _frames(g):
        cls = fx.legacy_pair_classes(fr["positions"], *args)
        a01 += int(cls["centroid_in_rc"].sum())
        a02 += int(cls["dropped"].sum())
        local = np.array(fx.to_input_order(list(fr["local_counts"]), fr["order"]))
        zero_local_with_neighbour |= bool(np.any((local == 0) & cls["centroid_in_rc"].any(axis=1)))
        # dropped pairs leave no trace: local + reduced + centroid partners exclude them
        assert np.all(
            local
            + cls["dropped"].sum(axis=1)
            + cls["reduced"].sum(axis=1)
            + cls["centroid"].sum(axis=1)
            == len(local) - 1
        )
    assert a01 > 0, "A01: an atom inside r_c reaches only via a cell centroid"
    assert a02 > 0, "A02: pairs beyond r_c in cells with level <= l_max are dropped"
    assert zero_local_with_neighbour


def test_ga_edge_level_lmax2_case() -> None:
    g = fx.load_descriptor_golden("GA_edge")
    f = list(g["frame_names"]).index("level_lmax2_within_rc")
    fr = fx.frame(g, f)
    local = fx.to_input_order(list(fr["local_counts"]), fr["order"])
    assert local == [1, 1, 0]  # the atom at x = 8.0 has no local row despite r = 7.001 < r_c


# --------------------------------------------------------------------------- dataset golden


@pytest.mark.parametrize("name", DATASET_CASES)
def test_dataset_golden_consistency(name: str) -> None:
    g = fx.load_dataset_golden(name)
    src = fx.load_synthetic("synthetic_small")
    n_frames, n_atoms = src["R"].shape[:2]
    indices = np.concatenate([g[f"{s}__index"] for s in SPLITS])
    assert np.array_equal(np.sort(indices), np.arange(n_frames)), "splits partition the frames"
    assert [len(g[f"{s}__index"]) for s in SPLITS] == [48, 6, 6]
    for s in SPLITS:
        idx, off = g[f"{s}__index"], g[f"{s}__atom_offsets"]
        a = int(off[-1])
        assert off.shape == (len(idx) + 1,) and np.all(np.diff(off) == n_atoms)
        assert np.array_equal(g[f"{s}__energy"], src["E"][idx])
        assert np.array_equal(g[f"{s}__forces"], src["F"][idx].reshape(-1, 3))
        assert g[f"{s}__local_matrix"].shape == (a, int(g["n_max_local"]), 4)
        assert g[f"{s}__local_numbers"].shape == (a, int(g["n_max_local"]), 2)
        assert g[f"{s}__long_range_matrix"].shape == (a, int(g["n_max_long_range"]), 4)
        assert g[f"{s}__long_range_features"].shape == (a, int(g["n_max_long_range"]), 101)
        for key in ("local_matrix", "local_numbers", "long_range_matrix", "long_range_features"):
            assert g[f"{s}__{key}"].dtype == np.float32
        for m in range(len(idx)):
            order = g[f"{s}__order"][off[m] : off[m + 1]]
            assert np.array_equal(np.sort(order), np.arange(n_atoms))
            assert np.array_equal(
                order, fx.legacy_traversal_order(src["R"][idx[m]], float(g["grid_size"]))
            )
    n_max = np.concatenate([g[f"{s}__n_max"] for s in SPLITS])
    assert [n_max[:, 0].max(), n_max[:, 1].max()] == [g["n_max_local"], g["n_max_long_range"]]
    assert MANIFEST["datasets"][name]["n_max_local"] == int(g["n_max_local"])


@pytest.mark.parametrize(("name", "case"), [("GA", "GA_aspirin"), ("GB", "GB_aspirin")])
def test_dataset_padding_matches_descriptor_golden(name: str, case: str) -> None:
    """Padded dataset rows equal the float32 descriptor rows of the same frame."""
    ds = fx.load_dataset_golden(name)
    desc = fx.load_descriptor_golden(case)
    for s in SPLITS:
        idx, off = ds[f"{s}__index"], ds[f"{s}__atom_offsets"]
        for m, frame_id in enumerate(idx):
            if frame_id >= int(desc["n_frames"]):
                continue
            fr = fx.frame(desc, int(frame_id))
            rows = fx.split_ragged(fr["local_rows"], fr["local_counts"])
            padded = ds[f"{s}__local_matrix"][off[m] : off[m + 1]]
            for k, r in enumerate(rows):
                assert np.array_equal(padded[k, : len(r)], r.astype(np.float32))
                assert not padded[k, len(r) :].any()
