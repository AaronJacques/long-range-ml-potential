"""Generate the data golden fixtures from the unchanged legacy code.

Run in the golden env (legacy code needs TF 2.21 + tf_keras)::

    TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 \\
        .overhaul/env/venv-py312-tf221/bin/python tests/golden/generate_data_golden.py [--check]

Writes into ``--out`` (default ``tests/fixtures/golden``) and the synthetic input
data into its parent (``tests/fixtures``). ``--check`` regenerates everything in a
temp dir and compares it bit for bit with the committed files.

``data_split.npz`` freezes the pandas split rule of ``create_dataset`` for many
frame counts (the dataset goldens alone use n = 60, where every split size is an
integer and the rounding of ``df.sample(frac=...)`` stays unchecked).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golden import fixtures, legacy, synthetic  # noqa: E402

GENERATOR_VERSION = 1
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
MANIFEST_NAME = "manifest_data.json"

# Golden configurations (binding for T02-T08): grid size [Angstrom], l_max.
CONFIGS: dict[str, dict[str, Any]] = {
    "G-A": {"grid_size": 1.0, "max_local_level": 6},
    "G-B": {"grid_size": 1.0, "max_local_level": 2},
    "G-C": {"grid_size": 2.0, "max_local_level": 4},
}

SYNTHETIC_SMALL_FRAMES = 60
STACHYOSE_FRAMES = 5

# name -> (config, source, frames or None for all, sort_matrix)
DESCRIPTOR_CASES: dict[str, tuple[str, str, int | None, bool]] = {
    "GA_aspirin": ("G-A", "synthetic_small", 10, False),
    "GB_aspirin": ("G-B", "synthetic_small", 10, False),
    "GB_aspirin_sorted": ("G-B", "synthetic_small", 10, True),
    "GA_edge": ("G-A", "edge_cases", None, False),
    "GB_edge": ("G-B", "edge_cases", None, False),
    "GC_stachyose": ("G-C", "stachyose_like", 3, False),
}

# name -> config; source is always synthetic_small.npz
DATASET_CASES: dict[str, str] = {"GA": "G-A", "GB": "G-B"}
SPLITS = ("train", "val", "test")
SPLIT_FILES = {"train": "train.pkl.gzip", "val": "val.pkl.gzip", "test": "test.pkl.gzip"}

# Split golden: the legacy split expressions (verbatim, ``Dataset.py:475-477`` in
# ``create_dataset``) run on a dummy DataFrame with a RangeIndex of n rows.
# ``generate`` verifies that these lines occur unchanged in the legacy source.
LEGACY_SPLIT_CODE = (
    "train_df = df.sample(frac=1 - (val_split + test_split), random_state=42)\n"
    "val_df = df.drop(train_df.index).sample("
    "frac=val_split / (val_split + test_split), random_state=42)\n"
    "test_df = df.drop(train_df.index).drop(val_df.index)\n"
)
# n = 27272: Stachyose (int() truncation gives a different train split); 63, 65:
# x.5 cases (round half to even differs from half up); 7, 13, 61, 101: small odd n.
SPLIT_SIZES = (7, 13, 60, 61, 63, 65, 101, 27272, 65000)
SPLIT_FRACTIONS = ((0.1, 0.1), (0.05, 0.15))
SPLIT_MAX_STORED = 1000  # n above this: sizes and sha256 only, no index arrays


# --------------------------------------------------------------------------- inputs


def write_inputs(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    small = synthetic.aspirin_like(SYNTHETIC_SMALL_FRAMES, seed=0)
    synthetic.write_npz(data_dir / "synthetic_small.npz", **small)
    stach = synthetic.stachyose_like(STACHYOSE_FRAMES, seed=1)
    synthetic.write_npz(data_dir / "stachyose_like.npz", **stach)
    edge: dict[str, np.ndarray] = {}
    for name, case in synthetic.edge_cases().items():
        edge[f"{name}__R"] = case["R"]
        edge[f"{name}__z"] = case["z"]
    fixtures.save_npz(data_dir / "edge_cases.npz", edge)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def source_frames(
    data_dir: Path, source: str, n: int | None
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """[(name, source frame index, R (N, 3), z (N,))]."""
    if source == "edge_cases":
        cases = synthetic.edge_cases()
        return [(name, i, c["R"], c["z"]) for i, (name, c) in enumerate(cases.items())]
    data = _load_npz(data_dir / f"{source}.npz")
    count = len(data["R"]) if n is None else n
    return [(f"frame_{f}", f, data["R"][f], data["z"]) for f in range(count)]


# --------------------------------------------------------------------------- descriptors


def _stack(parts: list, width: int, dtype: np.dtype) -> np.ndarray:
    """Concatenate per-atom legacy arrays; empty entries (shape (0,)) become (0, width)."""
    out = []
    for p in parts:
        a = np.asarray(p)
        if a.size == 0:
            out.append(np.zeros((0, width), dtype=dtype))
            continue
        if a.ndim != 2 or a.shape[1] != width or a.dtype != dtype:
            raise TypeError(
                f"unexpected legacy array {a.shape} {a.dtype}, want (k, {width}) {dtype}"
            )
        out.append(a)
    return np.concatenate(out, axis=0) if out else np.zeros((0, width), dtype=dtype)


def descriptor_golden(
    ns: Any, frames: list, cfg: dict[str, Any], sort_matrix: bool
) -> tuple[dict, dict]:
    consts = legacy.configure(ns, sort_matrix=sort_matrix, **cfg)
    width_lr = consts["MAX_ATOM_ELEMENTS"] + 1
    arrays: dict[str, np.ndarray] = {
        "n_frames": np.int64(len(frames)),
        "grid_size": np.float64(consts["GRID_SIZE"]),
        "max_local_level": np.int64(consts["MAX_LOCAL_LEVEL"]),
        "cut_off": np.float64(consts["CUT_OFF"]),
        "inner_cut_off": np.float64(consts["INNER_CUT_OFF"]),
        "sort_matrix": np.bool_(sort_matrix),
        "frame_names": np.array([name for name, _, _, _ in frames]),
        "frame_ids": np.array([fid for _, fid, _, _ in frames], dtype=np.int64),
    }
    stats = {
        "n_frames": len(frames),
        "n_atoms": 0,
        "local_rows": 0,
        "long_range_rows": 0,
        "n_max": [0, 0],
        "dropped_pairs": 0,
        "centroid_in_rc_pairs": 0,
        "reduced_pairs": 0,
    }
    for f, (_, _, R, z) in enumerate(frames):
        grid = ns.Dataset.create_grid(R.tolist(), z.tolist(), consts["GRID_SIZE"])
        loc, loc_num, lr, lr_feat, n_loc, n_lr = ns.Dataset.create_matrices(grid)
        p = f"f{f}__"
        arrays[p + "order"] = legacy.legacy_atom_order(grid, R)
        arrays[p + "positions"] = np.asarray(R, dtype=np.float64)
        arrays[p + "numbers"] = np.asarray(z, dtype=np.int64)
        arrays[p + "local_rows"] = _stack(loc, 4, np.dtype(np.float64))
        arrays[p + "local_counts"] = np.array([len(x) for x in loc], dtype=np.int64)
        arrays[p + "local_numbers"] = _stack(loc_num, 2, np.dtype(np.int64))
        arrays[p + "lr_rows"] = _stack(lr, 4, np.dtype(np.float64))
        arrays[p + "lr_counts"] = np.array([len(x) for x in lr], dtype=np.int64)
        arrays[p + "lr_features"] = _stack(lr_feat, width_lr, np.dtype(np.int64))
        arrays[p + "n_max"] = np.array([n_loc, n_lr], dtype=np.int64)

        cls = fixtures.legacy_pair_classes(
            R, consts["GRID_SIZE"], consts["MAX_LOCAL_LEVEL"], consts["CUT_OFF"]
        )
        stats["n_atoms"] += len(R)
        stats["local_rows"] += int(arrays[p + "local_counts"].sum())
        stats["long_range_rows"] += int(arrays[p + "lr_counts"].sum())
        stats["n_max"] = [max(stats["n_max"][0], int(n_loc)), max(stats["n_max"][1], int(n_lr))]
        stats["dropped_pairs"] += int(cls["dropped"].sum()) // 2
        stats["centroid_in_rc_pairs"] += int(cls["centroid_in_rc"].sum()) // 2
        stats["reduced_pairs"] += int(cls["reduced"].sum()) // 2
    return arrays, stats


# --------------------------------------------------------------------------- dataset


def dataset_golden(ns: Any, data_dir: Path, cfg: dict[str, Any]) -> tuple[dict, dict]:
    import pandas as pd

    consts = legacy.configure(ns, **cfg)
    keys = ns.Constants.Keys
    src = data_dir / "synthetic_small.npz"
    data = _load_npz(src)
    with tempfile.TemporaryDirectory(prefix="lrmlp_ds_") as td:
        ns.Dataset.create_dataset(
            [str(src)], consts["GRID_SIZE"], save_folder=str(Path(td) / "ds"), n_samples_per=None
        )
        folders = sorted(Path(td).glob("ds_*"))
        if len(folders) != 1:
            raise RuntimeError(f"expected one legacy dataset folder, got {folders}")
        dfs = {s: pd.read_pickle(folders[0] / SPLIT_FILES[s], compression="gzip") for s in SPLITS}

    arrays: dict[str, np.ndarray] = {
        "grid_size": np.float64(consts["GRID_SIZE"]),
        "max_local_level": np.int64(consts["MAX_LOCAL_LEVEL"]),
        "cut_off": np.float64(consts["CUT_OFF"]),
        "inner_cut_off": np.float64(consts["INNER_CUT_OFF"]),
    }
    stats: dict[str, Any] = {"splits": {}}
    n_max = [0, 0]
    for s in SPLITS:
        df = dfs[s]
        p = f"{s}__"
        index = df.index.to_numpy().astype(np.int64)
        local = list(df[keys.LOCAL_DISTANCE_MATRIX_KEY])
        for col, arr in ((keys.LOCAL_DISTANCE_MATRIX_KEY, local),):
            if any(a.dtype != np.float32 for a in arr):
                raise TypeError(f"{col}: legacy padding is not float32")
        counts = np.array([a.shape[0] for a in local], dtype=np.int64)
        arrays[p + "index"] = index
        arrays[p + "energy"] = df[keys.ENERGY_KEY].to_numpy().astype(np.float64)
        arrays[p + "forces"] = np.concatenate(
            [np.asarray(f, dtype=np.float64) for f in df[keys.FORCE_KEY]]
        )
        arrays[p + "atom_offsets"] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        arrays[p + "n_max"] = np.stack(
            [df[keys.N_MAX_LOCAL_KEY].to_numpy(), df[keys.N_MAX_LONG_RANGE_KEY].to_numpy()], axis=1
        ).astype(np.int64)
        arrays[p + "local_matrix"] = np.concatenate(local)
        arrays[p + "local_numbers"] = np.concatenate(list(df[keys.LOCAL_ATOMIC_NUMBERS_KEY]))
        arrays[p + "long_range_matrix"] = np.concatenate(
            list(df[keys.LONG_RANGE_DISTANCE_MATRIX_KEY])
        )
        arrays[p + "long_range_features"] = np.concatenate(
            list(df[keys.LONG_RANGE_ATOMIC_FEATURES_KEY])
        )
        orders = []
        for i in index:
            R = data["R"][i]
            grid = ns.Dataset.create_grid(R.tolist(), data["z"].tolist(), consts["GRID_SIZE"])
            orders.append(legacy.legacy_atom_order(grid, R))
        arrays[p + "order"] = np.concatenate(orders)
        n_max = [
            max(n_max[0], int(arrays[p + "n_max"][:, 0].max())),
            max(n_max[1], int(arrays[p + "n_max"][:, 1].max())),
        ]
        stats["splits"][s] = len(index)
    if (
        arrays["train__local_matrix"].shape[1] != n_max[0]
        or arrays["train__long_range_matrix"].shape[1] != n_max[1]
    ):
        raise RuntimeError("padding width differs from the maximum per-molecule n_max")
    arrays["n_max_local"] = np.int64(n_max[0])
    arrays["n_max_long_range"] = np.int64(n_max[1])
    stats["n_max_local"], stats["n_max_long_range"] = n_max
    return arrays, stats


# --------------------------------------------------------------------------- split


def split_case_name(n: int, val_split: float, test_split: float) -> str:
    return f"n{n}_v{round(val_split * 100):02d}_t{round(test_split * 100):02d}"


def index_sha256(index: np.ndarray) -> str:
    """sha256 of an index sequence as little-endian int64 bytes."""
    return hashlib.sha256(np.asarray(index, dtype="<i8").tobytes()).hexdigest()


def legacy_split_code(src: Path) -> str:
    """``LEGACY_SPLIT_CODE`` after checking it against the legacy ``Dataset.py``."""
    lines = [line.strip() for line in (src / "Dataset.py").read_text().splitlines()]
    want = LEGACY_SPLIT_CODE.splitlines()
    hits = [i for i in range(len(lines)) if lines[i : i + len(want)] == want]
    if len(hits) != 1:
        raise RuntimeError(f"legacy split expressions not found exactly once (hits: {hits})")
    return LEGACY_SPLIT_CODE


def split_golden(code: str) -> tuple[dict, dict]:
    import pandas as pd

    arrays: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {}
    names = []
    for n in SPLIT_SIZES:
        for val_split, test_split in SPLIT_FRACTIONS:
            scope: dict[str, Any] = {
                "df": pd.DataFrame({"frame": np.arange(n)}),
                "val_split": val_split,
                "test_split": test_split,
            }
            exec(code, {}, scope)  # verbatim legacy expressions
            parts = {s: scope[f"{s}_df"].index.to_numpy().astype(np.int64) for s in SPLITS}
            name = split_case_name(n, val_split, test_split)
            names.append(name)
            stored = n <= SPLIT_MAX_STORED
            p = f"{name}__"
            arrays[p + "n"] = np.int64(n)
            arrays[p + "fractions"] = np.array([val_split, test_split], dtype=np.float64)
            arrays[p + "sizes"] = np.array([len(parts[s]) for s in SPLITS], dtype=np.int64)
            arrays[p + "sha256"] = np.array([index_sha256(parts[s]) for s in SPLITS])
            if stored:
                for s in SPLITS:
                    arrays[p + s] = parts[s].astype(np.int32)
            stats[name] = {
                "n": n,
                "val_split": val_split,
                "test_split": test_split,
                "sizes": [len(parts[s]) for s in SPLITS],
                "indices_stored": stored,
            }
    arrays["cases"] = np.array(names)
    return arrays, stats


# --------------------------------------------------------------------------- manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions(ns: Any) -> dict[str, str]:
    import pandas as pd

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tensorflow": ns.Dataset.tf.__version__,
        "tf_keras": importlib.metadata.version("tf_keras"),
    }


def generate(out: Path, ref: str) -> dict[str, Any]:
    out = Path(out)
    data_dir = out.parent
    out.mkdir(parents=True, exist_ok=True)
    write_inputs(data_dir)

    src = legacy.extract_legacy(ref)
    try:
        ns = legacy.load_legacy(src)
        manifest: dict[str, Any] = {
            "generator": "tests/golden/generate_data_golden.py",
            "generator_version": GENERATOR_VERSION,
            "legacy_ref": ref,
            "legacy_commit": legacy.commit_sha(ref),
            "versions": versions(ns),
            "configs": {},
            "inputs": {
                "synthetic_small.npz": {
                    "generator": "aspirin_like",
                    "n_frames": SYNTHETIC_SMALL_FRAMES,
                    "seed": 0,
                },
                "stachyose_like.npz": {
                    "generator": "stachyose_like",
                    "n_frames": STACHYOSE_FRAMES,
                    "seed": 1,
                },
                "edge_cases.npz": {
                    "generator": "edge_cases",
                    "cases": list(synthetic.edge_cases()),
                },
            },
            "descriptors": {},
            "datasets": {},
            "splits": {},
        }
        for name, cfg in CONFIGS.items():
            manifest["configs"][name] = legacy.configure(ns, **cfg)

        for case, (cfg_name, source, n, sort) in DESCRIPTOR_CASES.items():
            frames = source_frames(data_dir, source, n)
            arrays, stats = descriptor_golden(ns, frames, CONFIGS[cfg_name], sort)
            fixtures.save_npz(out / f"data_descriptors_{case}.npz", arrays)
            manifest["descriptors"][case] = {
                "config": cfg_name,
                "source": f"{source}.npz",
                "sort_matrix": sort,
                **stats,
            }
            print(
                f"descriptors {case}: n_max={stats['n_max']} dropped={stats['dropped_pairs']} "
                f"centroid_in_rc={stats['centroid_in_rc_pairs']}",
                flush=True,
            )

        for name, cfg_name in DATASET_CASES.items():
            arrays, stats = dataset_golden(ns, data_dir, CONFIGS[cfg_name])
            fixtures.save_npz(out / f"data_dataset_{name}.npz", arrays)
            manifest["datasets"][name] = {
                "config": cfg_name,
                "source": "synthetic_small.npz",
                "val_split": 0.1,
                "test_split": 0.1,
                "random_state": 42,
                **stats,
            }
            print(
                f"dataset {name}: n_max=({stats['n_max_local']}, {stats['n_max_long_range']}) "
                f"splits={stats['splits']}",
                flush=True,
            )

        arrays, stats = split_golden(legacy_split_code(src))
        fixtures.save_npz(out / "data_split.npz", arrays)
        manifest["splits"] = {
            "code": LEGACY_SPLIT_CODE.splitlines(),
            "legacy_lines": "Dataset.py:475-477 (create_dataset)",
            "index": "RangeIndex(n)",
            "sha256": "sha256 of each index sequence as little-endian int64 bytes",
            "cases": stats,
        }
        print(f"split: {len(stats)} cases, n in {list(SPLIT_SIZES)}", flush=True)
    finally:
        legacy.cleanup_legacy(src)

    files = sorted(
        [
            data_dir / "synthetic_small.npz",
            data_dir / "stachyose_like.npz",
            data_dir / "edge_cases.npz",
        ]
        + [out / f"data_descriptors_{c}.npz" for c in DESCRIPTOR_CASES]
        + [out / f"data_dataset_{d}.npz" for d in DATASET_CASES]
        + [out / "data_split.npz"]
    )
    manifest["files"] = {
        str(p.relative_to(data_dir)): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in files
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return manifest


# --------------------------------------------------------------------------- check


def compare_arrays(a_path: Path, b_path: Path) -> list[str]:
    diffs = []
    a, b = _load_npz(a_path), _load_npz(b_path)
    if list(a) != list(b):
        diffs.append(
            f"{b_path.name}: keys differ: only new {sorted(set(a) - set(b))}, "
            f"only committed {sorted(set(b) - set(a))}"
        )
    for key in a:
        if key not in b:
            continue
        x, y = a[key], b[key]
        if x.dtype != y.dtype or x.shape != y.shape:
            diffs.append(f"{b_path.name}:{key}: {x.dtype}{x.shape} != committed {y.dtype}{y.shape}")
        elif x.tobytes() != y.tobytes():
            diffs.append(f"{b_path.name}:{key}: values differ")
    return diffs


def _strip(manifest: dict[str, Any]) -> dict[str, Any]:
    """Manifest fields that describe behaviour (no environment versions, no file encodings)."""
    m = dict(manifest)
    m.pop("versions", None)
    m.pop("generated_at", None)
    m["files"] = sorted(manifest.get("files", {}))
    return m


def _dict_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(
                    f"manifest{path}.{k}: present only in {'committed' if k in b else 'new'}"
                )
            else:
                out.extend(_dict_diff(a[k], b[k], f"{path}.{k}"))
        return out
    return [] if a == b else [f"manifest{path}: {a!r} != committed {b!r}"]


def check(committed_out: Path, ref: str) -> int:
    committed_out = Path(committed_out)
    committed_data = committed_out.parent
    committed_manifest_path = committed_out / MANIFEST_NAME
    if not committed_manifest_path.is_file():
        print(f"DIFF: {committed_manifest_path} missing")
        return 1
    committed = json.loads(committed_manifest_path.read_text())
    with tempfile.TemporaryDirectory(prefix="lrmlp_golden_check_") as td:
        new_out = Path(td) / "golden"
        new = generate(new_out, ref)
        diffs = _dict_diff(_strip(new), _strip(committed))
        notes = []
        for rel in sorted(set(new["files"]) & set(committed["files"])):
            committed_file = committed_data / rel
            if not committed_file.is_file():
                diffs.append(f"{rel}: missing in {committed_data}")
                continue
            diffs.extend(compare_arrays(Path(td) / rel, committed_file))
            if new["files"][rel]["sha256"] != sha256(committed_file):
                notes.append(f"note: {rel}: file bytes differ (encoding), arrays compared above")
        if new.get("versions") != committed.get("versions"):
            notes.append(
                f"note: versions differ: new {new.get('versions')} "
                f"vs committed {committed.get('versions')}"
            )
    for line in notes:
        print(line)
    if diffs:
        print("DIFF:")
        for line in diffs:
            print("  " + line)
        return 1
    print("OK: identical")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="golden dir (inputs go to its parent)"
    )
    parser.add_argument(
        "--check", action="store_true", help="regenerate in a temp dir and compare bit for bit"
    )
    parser.add_argument(
        "--ref", default=legacy.LEGACY_REF, help="legacy commit (default %(default)s)"
    )
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    if args.check:
        code = check(args.out, args.ref)
    else:
        generate(args.out, args.ref)
        print(f"wrote golden fixtures to {args.out}")
        code = 0
    print(f"runtime: {time.perf_counter() - t0:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
