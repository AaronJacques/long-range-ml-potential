"""Benchmarks of the legacy code (``Neural-Network/@9413bfd``) against ``lrmlp``.

Both implementations run on the same TensorFlow runtime of ``.venv``: the legacy
code with ``TF_USE_LEGACY_KERAS=1`` (tf_keras, extra ``legacy``), ``lrmlp`` with
Keras 3. One process measures one implementation; ``benchmarks/compare.py``
combines the two result files.

Usage::

    python benchmarks/run_benchmarks.py --impl new --suite all --out new.json
    TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 \\
        python benchmarks/run_benchmarks.py --impl legacy --suite all --out legacy.json

Equivalence mode (normally started by ``compare.py --check-equivalence``)::

    TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 \\
        python benchmarks/run_benchmarks.py --impl legacy --check --check-dir DIR
    python benchmarks/run_benchmarks.py --impl new --check --check-dir DIR --out eq.json

The legacy process exports its descriptors, model weights and energies into
``DIR``; the new process recomputes everything with ``lrmlp`` and compares. The
exit code is 1 if a tolerance is violated.

Suites (workloads are deterministic, see ``WORKLOADS``):

``prep``   descriptor preprocessing (legacy ``create_dataset``, new ``build_dataset``),
           molecules/s with one process and with ``--workers``, bytes per molecule on
           disk and in RAM.
``input``  one epoch of the training input pipeline without a model, molecules/s.
``train``  molecule steps/s over 2 epochs after a warm-up epoch, energy only, |B| = 1;
           new: ``steps_per_execution`` in {1, 8} x ``jit_compile`` in {false, true}.
``infer``  molecules/s for 1000 molecules; legacy: one call per molecule as
           ``evaluate.py``; new: ``Potential.predict(batch_size=256)`` with and
           without the descriptor computation.

The legacy code is never imported from the working tree: ``tests/golden/legacy.py``
exports it with ``git archive`` into a temporary directory.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from golden import legacy as golden_legacy  # noqa: E402
from golden import synthetic  # noqa: E402

SCHEMA_VERSION = 1
PROFILE = REPO_ROOT / "configs" / "legacy-head.toml"
THESIS_PRESETS = ("paracetamol", "aspirin", "ac-ala3-nhme", "stachyose")
SUITES = ("prep", "input", "train", "infer")

# name -> generator, seed, frames (full / --quick), grid (G-A / G-C of the goldens)
WORKLOADS: dict[str, dict[str, Any]] = {
    "aspirin": {
        "generator": "aspirin_like",
        "seed": 0,
        "n_frames": 2000,
        "n_frames_quick": 200,
        "grid": "G-A",
        "grid_size": 1.0,
        "max_local_level": 6,
    },
    "stachyose": {
        "generator": "stachyose_like",
        "seed": 1,
        "n_frames": 200,
        "n_frames_quick": 20,
        "grid": "G-C",
        "grid_size": 2.0,
        "max_local_level": 4,
    },
}
VAL_FRACTION, TEST_FRACTION, SPLIT_SEED = 0.1, 0.1, 42
SHUFFLE_BUFFER = 1000  # legacy Dataset.SHUFFLE_BUFFER_SIZE
TRAIN_SEED = 42
LEGACY_MODEL_SEED = 1234
INPUT_WORKLOAD = TRAIN_WORKLOAD = INFER_WORKLOAD = "aspirin"
N_INFER, N_INFER_QUICK = 1000, 200
INFER_BATCH = 256
WARMUP_EPOCHS, TIMED_EPOCHS = 1, 2
TRAIN_VARIANTS = ((1, False), (1, True), (8, False), (8, True))  # (steps_per_execution, xla)

# Equivalence tolerances (plan section 4.6).
DESC_ATOL = DESC_RTOL = 1e-6
ENERGY_RTOL, ENERGY_ATOL_PER_ATOM = 1e-5, 1e-5
XLA_STEPS, XLA_RTOL, XLA_ATOL = 100, 1e-4, 1e-6
SPE_RTOL, SPE_ATOL = 1e-4, 1e-6

INPUT_KEYS = ("local_matrix", "local_numbers", "long_range_matrix", "long_range_features")
INTEGER_KEYS = ("local_numbers", "long_range_features")
LEGACY_CHECK_FILE = "legacy_check.npz"
LEGACY_CHECK_META = "legacy_check.json"


# =========================================================================== helpers


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Silence the prints and progress bars of the legacy code."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def summarize(times: list[float], n: int | None = None, unit: str = "mol_per_s") -> dict:
    """Median and all repetitions; ``n`` items per repetition gives a rate."""
    med = statistics.median(times)
    out: dict[str, Any] = {"median_s": med, "times_s": times}
    if n is not None:
        out["n"] = n
        out[unit] = n / med if med > 0 else None
    return out


def timed(fn: Callable[[], Any], repeats: int, warmup: int = 0) -> tuple[list[float], Any]:
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def profile() -> dict[str, Any]:
    """The legacy-head profile (configs/legacy-head.toml) as plain TOML tables."""
    with PROFILE.open("rb") as fh:
        return tomllib.load(fh)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def environment() -> dict[str, Any]:
    cpu = ""
    mem = None
    if sys.platform == "darwin":
        cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        raw = _run(["sysctl", "-n", "hw.memsize"])
        mem = int(raw) if raw.isdigit() else None
    elif Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        with contextlib.suppress(ValueError, OSError, AttributeError):
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "src", "benchmarks"])
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": cpu or platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": mem,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_sha": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "git_dirty_src_or_benchmarks": bool(status),
        "legacy_ref": golden_legacy.LEGACY_REF,
        "legacy_commit": golden_legacy.commit_sha(),
    }


def loadavg() -> list[float] | None:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except OSError:
        return None


# =========================================================================== workloads


@dataclasses.dataclass
class Workload:
    name: str
    spec: dict[str, Any]
    data: dict[str, np.ndarray]
    npz: Path

    @property
    def n_frames(self) -> int:
        return int(self.data["R"].shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.data["R"].shape[1])

    @property
    def grid_size(self) -> float:
        return float(self.spec["grid_size"])

    @property
    def max_local_level(self) -> int:
        return int(self.spec["max_local_level"])

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for key in ("E", "F", "R", "z"):
            h.update(np.ascontiguousarray(self.data[key]).tobytes())
        return h.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "generator": f"tests/golden/synthetic.py:{self.spec['generator']}",
            "seed": self.spec["seed"],
            "n_frames": self.n_frames,
            "n_atoms": self.n_atoms,
            "grid": self.spec["grid"],
            "grid_size": self.grid_size,
            "max_local_level": self.max_local_level,
            "sha256": self.fingerprint(),
        }


class Workspace:
    """Temporary directory, workloads and cached datasets of one process."""

    def __init__(self, quick: bool, keep: bool = False) -> None:
        self.quick = quick
        self.keep = keep
        self.root = Path(tempfile.mkdtemp(prefix="lrmlp_bench_"))
        self._workloads: dict[str, Workload] = {}
        self._counter = 0
        self.datasets: dict[str, Path] = {}

    def close(self) -> None:
        if not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)
        else:
            log(f"kept work directory {self.root}")

    def fresh(self, prefix: str) -> Path:
        self._counter += 1
        path = self.root / f"{prefix}_{self._counter:03d}"
        path.mkdir(parents=True)
        return path

    def workload(self, name: str) -> Workload:
        if name not in self._workloads:
            spec = WORKLOADS[name]
            n = spec["n_frames_quick"] if self.quick else spec["n_frames"]
            data = getattr(synthetic, spec["generator"])(n, seed=spec["seed"])
            npz = self.root / f"{name}.npz"
            synthetic.write_npz(npz, **data)
            self._workloads[name] = Workload(name, spec, data, npz)
        return self._workloads[name]

    def subset(self, name: str, n: int) -> Workload:
        key = f"{name}[:{n}]"
        if key not in self._workloads:
            wl = self.workload(name)
            data = {k: (v[:n] if k != "z" else v) for k, v in wl.data.items()}
            npz = self.root / f"{name}_{n}.npz"
            synthetic.write_npz(npz, **data)
            self._workloads[key] = Workload(name, wl.spec, data, npz)
        return self._workloads[key]

    def described(self) -> dict[str, Any]:
        return {k: w.describe() for k, w in self._workloads.items()}


def n_infer(quick: bool) -> int:
    return N_INFER_QUICK if quick else N_INFER


# =========================================================================== legacy


class TimedIterable:
    """Wraps a dataset; records the time of every request for an element.

    ``passes[p]`` holds the ``perf_counter`` values before each ``next`` of pass
    ``p`` plus the time of the final ``StopIteration``. The loop body of the
    consumer runs between two entries, so the pass time includes the work done on
    each element (an observation hook, the legacy code is unchanged).
    """

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.passes: list[list[float]] = []

    def __iter__(self) -> Iterator[Any]:
        stamps: list[float] = []
        self.passes.append(stamps)
        stamps.append(time.perf_counter())
        for element in self.dataset:
            yield element
            stamps.append(time.perf_counter())

    def pass_times(self) -> list[float]:
        return [p[-1] - p[0] for p in self.passes]

    def steady_times(self) -> list[float]:
        """Pass time without the first element (excludes tracing on the first call)."""
        return [p[-1] - p[1] for p in self.passes if len(p) > 2]


def legacy_hyperparameters(epochs: int = 1) -> SimpleNamespace:
    """``Constants.Hyperparameters`` of the legacy-head profile (attribute object)."""
    prof = profile()
    m, t = prof["model"], prof["train"]
    return SimpleNamespace(
        initial_learning_rate=t["learning_rate"],
        lr_decay_steps=t["lr_decay_steps"],
        decay_steps=t["loss_weight_decay_steps"],
        decay_rate=t["lr_decay_rate"],
        epochs=epochs,
        p_energy_start=t["p_energy_start"],
        p_energy_limit=t["p_energy_limit"],
        p_force_start=t["p_force_start"],
        p_force_limit=t["p_force_limit"],
        M1_local=m["k1_local"],
        M2_local=m["k2_local"],
        M1_long=m["k1_long"],
        M2_long=m["k2_long"],
        embedding_dims=list(m["embedding_dims"]),
    )


class LegacyBench:
    def __init__(self, ws: Workspace, repeats: int) -> None:
        if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
            raise SystemExit("--impl legacy needs TF_USE_LEGACY_KERAS=1 (Keras 2 / tf_keras)")
        self.ws = ws
        self.repeats = repeats
        self.src = golden_legacy.extract_legacy()
        self.ns = golden_legacy.load_legacy(self.src)
        self.tf = self.ns.Training.tf
        self.keras = self.tf.keras

    def close(self) -> None:
        golden_legacy.cleanup_legacy(self.src)

    def versions(self) -> dict[str, Any]:
        import pandas as pd

        return {
            "tensorflow": self.tf.__version__,
            "keras": f"tf_keras {importlib.metadata.version('tf_keras')}",
            "pandas": pd.__version__,
            "intra_op_threads": self.tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op_threads": self.tf.config.threading.get_inter_op_parallelism_threads(),
        }

    # ------------------------------------------------------------------ helpers

    def configure(self, wl: Workload, n_max: tuple[int, int] | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if n_max is not None:
            kwargs = {"n_max_local": n_max[0], "n_max_long_range": n_max[1]}
        consts = golden_legacy.configure(
            self.ns,
            grid_size=wl.grid_size,
            max_local_level=wl.max_local_level,
            shuffle_buffer=SHUFFLE_BUFFER,
            predict_only_energy=True,
            use_long_range=True,
            **kwargs,
        )
        self.ns.Constants.Model.activation = profile()["model"]["activation"]
        return consts

    def create_dataset(
        self, wl: Workload, val_split: float = VAL_FRACTION, test_split: float = TEST_FRACTION
    ) -> Path:
        self.configure(wl)
        work = self.ws.fresh(f"legacy_ds_{wl.name}")
        with quiet():
            self.ns.Dataset.create_dataset(
                [str(wl.npz)],
                wl.grid_size,
                save_folder=str(work / "ds"),
                val_split=val_split,
                test_split=test_split,
                n_samples_per=None,
            )
        (folder,) = work.glob("ds_*")
        return folder

    def dataset(self, wl: Workload) -> Path:
        if wl.name not in self.ws.datasets:
            self.ws.datasets[wl.name] = self.create_dataset(wl)
        return self.ws.datasets[wl.name]

    def read_splits(self, folder: Path) -> dict[str, Any]:
        import pandas as pd

        return {
            s: pd.read_pickle(folder / f"{s}.pkl.gzip", compression="gzip")
            for s in ("train", "val", "test")
        }

    def n_max(self, folder: Path) -> tuple[int, int]:
        keys = self.ns.Constants.Keys
        df = self.read_splits(folder)["train"]
        return (
            int(df[keys.LOCAL_DISTANCE_MATRIX_KEY].iloc[0].shape[1]),
            int(df[keys.LONG_RANGE_DISTANCE_MATRIX_KEY].iloc[0].shape[1]),
        )

    def build_model(self, hp: SimpleNamespace) -> Any:
        self.keras.backend.clear_session()
        self.keras.utils.set_random_seed(LEGACY_MODEL_SEED)
        return self.ns.Model.get_model(hp)

    # ------------------------------------------------------------------ suites

    def prep(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in WORKLOADS:
            wl = self.ws.workload(name)
            log(f"legacy prep {name}: {wl.n_frames} frames x {self.repeats}")
            folders: list[Path] = []
            times, _ = timed(
                lambda wl=wl, acc=folders: acc.append(self.create_dataset(wl)), self.repeats
            )
            folder = folders[-1]
            self.ws.datasets[name] = folder
            splits = self.read_splits(folder)
            ram = 0
            for df in splits.values():
                for col in df.columns:
                    ram += sum(np.asarray(v).nbytes for v in df[col])
            disk = sum((folder / f"{s}.pkl.gzip").stat().st_size for s in splits)
            out[name] = {
                "serial": summarize(times, wl.n_frames),
                "parallel": None,
                "bytes_per_molecule": {"disk": disk / wl.n_frames, "ram": ram / wl.n_frames},
                "n_max": list(self.n_max(folder)),
                "format": "pandas DataFrame, .pkl.gzip, padded float32 arrays per molecule",
            }
        return out

    def input(self) -> dict[str, Any]:
        wl = self.ws.workload(INPUT_WORKLOAD)
        folder = self.dataset(wl)
        self.configure(wl, self.n_max(folder))
        path = str(folder / "train.pkl.gzip")
        log(f"legacy input {wl.name}")
        load_times = []
        ds = size = None
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            with quiet():
                ds, size = self.ns.Dataset.create_tf_dataset(path)
            load_times.append(time.perf_counter() - t0)

        def epoch() -> None:
            for _ in ds:
                pass

        times, _ = timed(epoch, self.repeats, warmup=1)
        return {
            wl.name: {
                "split": "train",
                "load": summarize(load_times),
                "epoch": summarize(times, size),
                "pipeline": "from_generator over df.iterrows(), shuffle 1000, prefetch",
            }
        }

    def train(self) -> dict[str, Any]:
        wl = self.ws.workload(TRAIN_WORKLOAD)
        folder = self.dataset(wl)
        n_max = self.configure(wl, self.n_max(folder))
        n_max = (n_max["n_max_local"], n_max["n_max_long_range"])
        self.ns.Constants.Dataset.FOLDER = str(folder)
        epochs = WARMUP_EPOCHS + TIMED_EPOCHS * self.repeats
        hp = legacy_hyperparameters(epochs)
        wrapped: dict[str, TimedIterable] = {}
        original_ds = self.ns.Training.create_tf_dataset
        original_model = self.ns.Training.get_model

        def create_tf_dataset(path: str) -> tuple[Any, int]:
            ds, size = original_ds(path)
            wrapped[Path(path).name] = TimedIterable(ds)
            return wrapped[Path(path).name], size

        work = self.ws.fresh("legacy_train")
        (work / "cwd").mkdir()
        cwd = Path.cwd()
        log(f"legacy train {wl.name}: {epochs} epochs")
        self.ns.Training.create_tf_dataset = create_tf_dataset
        self.ns.Training.get_model = self.build_model
        t0 = time.perf_counter()
        try:
            os.chdir(work / "cwd")
            with quiet():
                self.ns.Training.start_training(hp)
        finally:
            os.chdir(cwd)
            self.ns.Training.create_tf_dataset = original_ds
            self.ns.Training.get_model = original_model
        total = time.perf_counter() - t0
        train_passes = wrapped["train.pkl.gzip"].pass_times()
        val_passes = wrapped["val.pkl.gzip"].pass_times()
        n_train = len(wrapped["train.pkl.gzip"].passes[0]) - 1
        n_val = len(wrapped["val.pkl.gzip"].passes[0]) - 1
        windows = [
            sum(
                train_passes[
                    WARMUP_EPOCHS + TIMED_EPOCHS * r : WARMUP_EPOCHS + TIMED_EPOCHS * (r + 1)
                ]
            )
            for r in range(self.repeats)
        ]
        return {
            wl.name: {
                "n_train": n_train,
                "n_val": n_val,
                "n_max": list(n_max),
                "epochs_per_window": TIMED_EPOCHS,
                "start_training_total_s": total,
                "variants": {
                    "legacy": {
                        "steps_per_execution": 1,
                        "jit_compile": False,
                        "warmup_s": train_passes[:WARMUP_EPOCHS],
                        "epoch_times_s": train_passes,
                        "window": summarize(windows, TIMED_EPOCHS * n_train, "steps_per_s"),
                    }
                },
                "validation": summarize(val_passes[1:], n_val),
            }
        }

    def infer(self) -> dict[str, Any]:
        n = n_infer(self.ws.quick)
        wl = self.ws.subset(INFER_WORKLOAD, n)
        log(f"legacy infer {wl.name}: {n} molecules")
        folders: list[Path] = []
        prep_times, _ = timed(
            lambda: folders.append(self.create_dataset(wl, val_split=0.0, test_split=1.0)),
            self.repeats,
        )
        folder = folders[-1]
        test_pkl = folder / "test.pkl.gzip"
        n_max = self.n_max_from(test_pkl)
        self.configure(wl, n_max)
        model = self.build_model(legacy_hyperparameters())
        h5 = self.ws.fresh("legacy_infer") / "model.h5"
        with quiet():
            model.save(str(h5))
        wrapped: list[TimedIterable] = []
        original = self.ns.evaluate.create_tf_dataset

        def create_tf_dataset(path: str) -> tuple[Any, int]:
            ds, size = original(path)
            wrapped.append(TimedIterable(ds))
            return wrapped[-1], size

        self.ns.evaluate.create_tf_dataset = create_tf_dataset
        totals = []
        try:
            for _ in range(self.repeats):
                t0 = time.perf_counter()
                with quiet():
                    self.ns.evaluate.evaluate_model(str(h5), str(test_pkl), use_long_range=True)
                totals.append(time.perf_counter() - t0)
        finally:
            self.ns.evaluate.create_tf_dataset = original
        steady = [w.steady_times()[0] for w in wrapped]
        evaluate_total = summarize(totals, n)
        prep = summarize(prep_times, n)
        e2e = prep["median_s"] + evaluate_total["median_s"]
        return {
            wl.name: {
                "n_molecules": n,
                "n_max": list(n_max),
                "batch_size": 1,
                "without_descriptors": summarize(steady, n - 1),
                "evaluate_model_total": evaluate_total,
                "create_dataset": prep,
                "with_descriptors": {
                    "median_s": e2e,
                    "n": n,
                    "mol_per_s": n / e2e,
                    "composition": "median create_dataset + median evaluate_model (incl. load)",
                },
                "note": "without_descriptors: evaluate.py test_step loop, first molecule "
                "(tf.function tracing) excluded",
            }
        }

    def n_max_from(self, pkl: Path) -> tuple[int, int]:
        import pandas as pd

        keys = self.ns.Constants.Keys
        df = pd.read_pickle(pkl, compression="gzip")
        return (
            int(df[keys.LOCAL_DISTANCE_MATRIX_KEY].iloc[0].shape[1]),
            int(df[keys.LONG_RANGE_DISTANCE_MATRIX_KEY].iloc[0].shape[1]),
        )

    # ------------------------------------------------------------------ equivalence export

    def export_check(self, check_dir: Path) -> None:
        import pandas as pd
        from golden.generate_model_golden import record_construction, weight_arrays

        keys = self.ns.Constants.Keys
        cols = {
            "local_matrix": keys.LOCAL_DISTANCE_MATRIX_KEY,
            "local_numbers": keys.LOCAL_ATOMIC_NUMBERS_KEY,
            "long_range_matrix": keys.LONG_RANGE_DISTANCE_MATRIX_KEY,
            "long_range_features": keys.LONG_RANGE_ATOMIC_FEATURES_KEY,
        }
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {"workloads": {}}
        for name in WORKLOADS:
            wl = self.ws.workload(name)
            log(f"legacy check export {name}: {wl.n_frames} frames")
            folder = self.create_dataset(wl)
            df = pd.concat(list(self.read_splits(folder).values())).sort_index()
            if not np.array_equal(df.index.to_numpy(), np.arange(wl.n_frames)):
                raise RuntimeError("legacy dataset index is not the frame index")
            p = f"{name}__"
            for key, col in cols.items():
                block = np.concatenate(list(df[col]))
                if key in INTEGER_KEYS:
                    compact = block.astype(np.uint8)
                    if not np.array_equal(compact.astype(block.dtype), block):
                        raise RuntimeError(f"{key}: legacy values do not fit uint8")
                    block = compact
                arrays[p + key] = block
            orders = []
            consts = self.configure(wl)
            for f in range(wl.n_frames):
                R = wl.data["R"][f]
                grid = self.ns.Dataset.create_grid(
                    R.tolist(), wl.data["z"].tolist(), consts["GRID_SIZE"]
                )
                orders.append(golden_legacy.legacy_atom_order(grid, R))
            arrays[p + "order"] = np.stack(orders).astype(np.int64)
            n_max = (arrays[p + "local_matrix"].shape[1], arrays[p + "long_range_matrix"].shape[1])

            # Model with the legacy-head architecture, weights in construction order.
            self.configure(wl, n_max)
            self.keras.backend.clear_session()
            self.keras.utils.set_random_seed(LEGACY_MODEL_SEED)
            with record_construction(self.ns) as created:
                model = self.ns.Model.get_model(legacy_hyperparameters())
            in_model = {id(layer) for layer in model.layers}
            layers = [layer for layer in created if id(layer) in in_model]
            for key, w in weight_arrays(layers).items():
                arrays[f"{p}w__{key}"] = w
            forward = self.tf.function(lambda x, model=model: model(x, training=False))
            energies = np.empty(wl.n_frames, dtype=np.float64)
            n = wl.n_atoms
            for f in range(wl.n_frames):
                x = [
                    self.tf.constant(arrays[p + k][f * n : (f + 1) * n], self.tf.float32)
                    for k in INPUT_KEYS
                ]
                pred = forward(x)
                energies[f] = float(self.tf.math.reduce_sum(pred[:, 0]).numpy())
            arrays[p + "energy"] = energies
            meta["workloads"][name] = {**wl.describe(), "n_max": list(n_max)}
        np.savez(check_dir / LEGACY_CHECK_FILE, **arrays)
        meta["versions"] = self.versions()
        (check_dir / LEGACY_CHECK_META).write_text(json.dumps(meta, indent=2) + "\n")


# =========================================================================== new


class NewBench:
    def __init__(self, ws: Workspace, repeats: int, workers: int) -> None:
        if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
            raise SystemExit("--impl new needs Keras 3: unset TF_USE_LEGACY_KERAS")
        self.ws = ws
        self.repeats = repeats
        self.workers = workers

    def close(self) -> None:
        pass

    def versions(self) -> dict[str, Any]:
        import keras
        import tensorflow as tf

        import lrmlp
        from lrmlp.config import TrainConfig, load_config

        def execution(tc: Any) -> dict[str, Any]:
            return {"steps_per_execution": tc.steps_per_execution, "jit_compile": tc.jit_compile}

        return {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "lrmlp": lrmlp.__version__,
            "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            # Execution settings of the measured code: the TrainConfig defaults and the
            # four thesis presets (labels of the training rows in compare.py).
            "train_defaults": {
                "TrainConfig": execution(TrainConfig()),
                "presets": {
                    name: execution(load_config(REPO_ROOT / "configs" / f"{name}.toml").train)
                    for name in THESIS_PRESETS
                },
            },
        }

    # ------------------------------------------------------------------ helpers

    def config(self, wl: Workload, output_dir: Path, **train: Any) -> Any:
        from lrmlp.config import load_config

        base = load_config(PROFILE)
        grid = dataclasses.replace(
            base.grid, grid_size=wl.grid_size, max_local_level=wl.max_local_level
        )
        data = dataclasses.replace(
            base.data,
            sources=(wl.npz,),
            n_samples_per_source="all",
            val_fraction=VAL_FRACTION,
            test_fraction=TEST_FRACTION,
            split_seed=SPLIT_SEED,
            output_dir=output_dir,
            n_workers=1,
        )
        tc = dataclasses.replace(
            base.train,
            shuffle_buffer=SHUFFLE_BUFFER,
            seed=TRAIN_SEED,
            run_root=self.ws.root / "runs",
            **train,
        )
        return dataclasses.replace(base, name=f"bench-{wl.name}", grid=grid, data=data, train=tc)

    def build(self, wl: Workload, workers: int) -> Path:
        from lrmlp.dataset import build_dataset

        out = self.ws.fresh(f"new_ds_{wl.name}") / "dataset"
        return build_dataset(self.config(wl, out), n_workers=workers)

    def dataset(self, wl: Workload) -> Path:
        if wl.name not in self.ws.datasets:
            self.ws.datasets[wl.name] = self.build(wl, self.workers)
        return self.ws.datasets[wl.name]

    def model(self, wl: Workload, n_max: tuple[int, int]) -> tuple[Any, Any]:
        import keras

        from lrmlp.model import build_model

        cfg = self.config(wl, self.ws.root / "unused")
        keras.utils.set_random_seed(LEGACY_MODEL_SEED)
        return cfg, build_model(cfg.model, n_max[0], n_max[1], cfg.grid.max_atomic_number)

    # ------------------------------------------------------------------ suites

    def prep(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in WORKLOADS:
            wl = self.ws.workload(name)
            log(
                f"new prep {name}: {wl.n_frames} frames x {self.repeats} "
                f"(1 and {self.workers} workers)"
            )
            paths: list[Path] = []
            serial, _ = timed(lambda wl=wl, acc=paths: acc.append(self.build(wl, 1)), self.repeats)
            parallel, _ = timed(
                lambda wl=wl, acc=paths: acc.append(self.build(wl, self.workers)), self.repeats
            )
            path = paths[-1]
            self.ws.datasets[name] = path
            ram = 0
            for f in path.rglob("*.npy"):
                try:
                    ram += np.load(f, mmap_mode="r").nbytes
                except ValueError:  # empty arrays cannot be memory-mapped
                    ram += np.load(f).nbytes
            meta = json.loads((path / "metadata.json").read_text())
            out[name] = {
                "serial": summarize(serial, wl.n_frames),
                "parallel": {**summarize(parallel, wl.n_frames), "n_workers": self.workers},
                "bytes_per_molecule": {
                    "disk": dir_bytes(path) / wl.n_frames,
                    "ram": ram / wl.n_frames,
                },
                "n_max": [meta["n_max_local"], meta["n_max_long_range"]],
                "format": "directory of .npy arrays (mmap), compact species counts",
            }
        return out

    def input(self) -> dict[str, Any]:
        from lrmlp.dataset import load_dataset, make_tf_dataset

        wl = self.ws.workload(INPUT_WORKLOAD)
        path = self.dataset(wl)
        log(f"new input {wl.name}")
        holder: dict[str, Any] = {}

        def load() -> None:
            split = load_dataset(path)["train"]
            holder["n"] = len(split)
            holder["ds"] = make_tf_dataset(
                split, shuffle=True, shuffle_buffer=SHUFFLE_BUFFER, seed=TRAIN_SEED
            )

        load_times, _ = timed(load, self.repeats)

        def epoch() -> None:
            for _ in holder["ds"]:
                pass

        times, _ = timed(epoch, self.repeats, warmup=1)
        return {
            wl.name: {
                "split": "train",
                "load": summarize(load_times),
                "epoch": summarize(times, holder["n"]),
                "pipeline": "Dataset.range + numpy_function over mmap arrays, shuffle 1000, "
                "prefetch",
            }
        }

    def train(self) -> dict[str, Any]:
        import keras

        from lrmlp.training import Trainer

        wl = self.ws.workload(TRAIN_WORKLOAD)
        path = self.dataset(wl)
        variants: dict[str, Any] = {}
        validation: dict[str, Any] | None = None
        n_train = n_val = 0
        n_max: list[int] = []
        for k, xla in TRAIN_VARIANTS:
            key = f"spe{k}_xla{int(xla)}"
            log(f"new train {wl.name}: {key}")
            keras.backend.clear_session()
            gc.collect()
            cfg = self.config(wl, path, steps_per_execution=k, jit_compile=xla)
            trainer = Trainer(cfg, path, run_dir=self.ws.fresh(f"new_run_{key}") / "run")
            n_train, n_val = len(trainer.dataset["train"]), len(trainer.dataset["val"])
            n_max = [trainer.dataset.n_max_local, trainer.dataset.n_max_long_range]
            epoch = 0
            warm = []
            for _ in range(WARMUP_EPOCHS):
                epoch += 1
                t0 = time.perf_counter()
                trainer.train_epoch(epoch)
                warm.append(time.perf_counter() - t0)
            windows, epoch_times = [], []
            for _ in range(self.repeats):
                t0 = time.perf_counter()
                for _ in range(TIMED_EPOCHS):
                    epoch += 1
                    t1 = time.perf_counter()
                    trainer.train_epoch(epoch)
                    epoch_times.append(time.perf_counter() - t1)
                windows.append(time.perf_counter() - t0)
            variants[key] = {
                "steps_per_execution": k,
                "jit_compile": xla,
                "warmup_s": warm,
                "epoch_times_s": warm + epoch_times,
                "window": summarize(windows, TIMED_EPOCHS * n_train, "steps_per_s"),
                "tracing_counts": trainer.tracing_counts(),
            }
            if validation is None:
                times, _ = timed(trainer.validate, self.repeats, warmup=1)
                validation = summarize(times, n_val)
            del trainer
        return {
            wl.name: {
                "n_train": n_train,
                "n_val": n_val,
                "n_max": n_max,
                "epochs_per_window": TIMED_EPOCHS,
                "variants": variants,
                "validation": validation,
            }
        }

    def infer(self) -> dict[str, Any]:
        from lrmlp.descriptors import compute_descriptors, max_neighbour_counts
        from lrmlp.potential import Potential

        n = n_infer(self.ws.quick)
        wl = self.ws.subset(INFER_WORKLOAD, n)
        log(f"new infer {wl.name}: {n} molecules")
        R, z = wl.data["R"], wl.data["z"]
        cfg = self.config(wl, self.ws.root / "unused")
        n_loc = n_lr = 0
        for s in range(n):
            a, b = max_neighbour_counts(compute_descriptors(R[s], z, cfg.grid))
            n_loc, n_lr = max(n_loc, a), max(n_lr, b)
        cfg, model = self.model(wl, (n_loc, n_lr))
        pot = Potential(model, cfg, n_loc, n_lr)
        pot.predict(R[:2], z, batch_size=INFER_BATCH)  # trace once

        holder: dict[str, Any] = {}

        def descriptors() -> None:
            holder["desc"] = [pot.descriptors(R[s], z) for s in range(n)]

        desc_times, _ = timed(descriptors, self.repeats)
        model_times, _ = timed(
            lambda: pot.predict_descriptors(holder["desc"], batch_size=INFER_BATCH), self.repeats
        )
        full_times, _ = timed(lambda: pot.predict(R, z, batch_size=INFER_BATCH), self.repeats)
        par_times, _ = timed(
            lambda: pot.predict(R, z, batch_size=INFER_BATCH, n_workers=self.workers),
            self.repeats,
        )
        return {
            wl.name: {
                "n_molecules": n,
                "n_max": [n_loc, n_lr],
                "batch_size": INFER_BATCH,
                "without_descriptors": summarize(model_times, n),
                "descriptors_only": summarize(desc_times, n),
                "with_descriptors": summarize(full_times, n),
                "with_descriptors_parallel": {
                    **summarize(par_times, n),
                    "n_workers": self.workers,
                },
                "tracing_count": pot.tracing_count,
            }
        }

    # ------------------------------------------------------------------ equivalence

    def check(self, check_dir: Path) -> dict[str, Any]:
        legacy_meta = json.loads((check_dir / LEGACY_CHECK_META).read_text())
        with np.load(check_dir / LEGACY_CHECK_FILE) as raw:
            legacy = {k: raw[k] for k in raw.files}
        report: dict[str, Any] = {
            "tolerances": {
                "descriptors": {"atol": DESC_ATOL, "rtol": DESC_RTOL, "integers": "exact"},
                "energies": {"rtol": ENERGY_RTOL, "atol_per_atom": ENERGY_ATOL_PER_ATOM},
                "xla": {"steps": XLA_STEPS, "rtol": XLA_RTOL, "atol": XLA_ATOL},
                "steps_per_execution": {"rtol": SPE_RTOL, "atol": SPE_ATOL},
            },
            "legacy_versions": legacy_meta["versions"],
            "workloads": {},
        }
        for name in WORKLOADS:
            wl = self.ws.workload(name)
            want = legacy_meta["workloads"][name]["sha256"]
            if wl.fingerprint() != want:
                raise RuntimeError(f"{name}: workload differs from the legacy process")
            report["workloads"][name] = {
                "n_frames": wl.n_frames,
                "descriptors": self._check_descriptors(wl, legacy, legacy_meta),
                "energies": self._check_energies(wl, legacy, legacy_meta),
            }
        report["xla"] = self._check_xla()
        report["steps_per_execution"] = self._check_spe()
        checks = [report["xla"], report["steps_per_execution"]]
        for w in report["workloads"].values():
            checks += [w["descriptors"], w["energies"]]
        report["passed"] = all(c["passed"] for c in checks)
        return report

    def _check_descriptors(self, wl: Workload, legacy: dict, meta: dict) -> dict[str, Any]:
        from lrmlp.descriptors import compute_descriptors, max_neighbour_counts, pad_descriptors

        log(f"check descriptors {wl.name}")
        p = f"{wl.name}__"
        n_max = tuple(meta["workloads"][wl.name]["n_max"])
        cfg = self.config(wl, self.ws.root / "unused")
        n = wl.n_atoms
        stats = {k: {"max_abs_diff": 0.0, "violations": 0} for k in INPUT_KEYS}
        new_n_max = [0, 0]
        for f in range(wl.n_frames):
            desc = compute_descriptors(wl.data["R"][f], wl.data["z"], cfg.grid)
            a, b = max_neighbour_counts(desc)
            new_n_max = [max(new_n_max[0], a), max(new_n_max[1], b)]
            padded = pad_descriptors(desc, n_max[0], n_max[1])
            order = legacy[p + "order"][f]
            for key in INPUT_KEYS:
                got = np.asarray(padded[key], dtype=np.float64)[order]
                ref = legacy[p + key][f * n : (f + 1) * n].astype(np.float64)
                diff = np.abs(got - ref)
                stats[key]["max_abs_diff"] = max(stats[key]["max_abs_diff"], float(diff.max()))
                if key in INTEGER_KEYS:
                    bad = int(np.count_nonzero(diff))
                else:
                    bad = int(np.count_nonzero(diff > DESC_ATOL + DESC_RTOL * np.abs(ref)))
                stats[key]["violations"] += bad
        n_max_equal = tuple(new_n_max) == n_max
        return {
            "n_max_legacy": list(n_max),
            "n_max_new": new_n_max,
            "n_max_equal": n_max_equal,
            "per_key": stats,
            "passed": n_max_equal and all(s["violations"] == 0 for s in stats.values()),
        }

    def _check_energies(self, wl: Workload, legacy: dict, meta: dict) -> dict[str, Any]:
        from lrmlp.model import assign_legacy_weights
        from lrmlp.potential import Potential

        log(f"check energies {wl.name}")
        p = f"{wl.name}__"
        n_max = tuple(meta["workloads"][wl.name]["n_max"])
        grouped: dict[str, dict[int, dict[int, np.ndarray]]] = {}
        for key, value in legacy.items():
            if key.startswith(p + "w__"):
                cls, k, j = key[len(p) + 3 :].split("__")
                grouped.setdefault(cls, {}).setdefault(int(k), {})[int(j)] = value
        weights = {
            cls: [[layers[k][j] for j in sorted(layers[k])] for k in sorted(layers)]
            for cls, layers in grouped.items()
        }
        cfg, model = self.model(wl, n_max)
        assign_legacy_weights(model, weights)
        pot = Potential(model, cfg, n_max[0], n_max[1])
        got = pot.predict(wl.data["R"], wl.data["z"], batch_size=INFER_BATCH).energies
        ref = legacy[p + "energy"]
        diff = np.abs(got - ref)
        tol = ENERGY_ATOL_PER_ATOM * wl.n_atoms + ENERGY_RTOL * np.abs(ref)
        rel = diff / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
        return {
            "n_molecules": int(ref.shape[0]),
            "max_abs_diff": float(diff.max()),
            "max_rel_diff": float(rel.max()),
            "max_abs_energy": float(np.abs(ref).max()),
            "violations": int(np.count_nonzero(diff > tol)),
            "passed": bool(np.all(diff <= tol)),
        }

    def _trainer_pair(self, name: str, a: dict[str, Any], b: dict[str, Any]) -> tuple[Any, Any]:
        import keras

        from lrmlp.model import layers_in_construction_order
        from lrmlp.training import Trainer

        wl = self.ws.workload(TRAIN_WORKLOAD)
        path = self.dataset(wl)
        keras.backend.clear_session()
        first = Trainer(self.config(wl, path, **a), path, run_dir=self.ws.fresh(name) / "run")
        start: dict[str, list[list[np.ndarray]]] = {}
        for layer in layers_in_construction_order(first.model):
            start.setdefault(type(layer).__name__, []).append(layer.get_weights())
        second = Trainer(
            self.config(wl, path, **b),
            path,
            run_dir=self.ws.fresh(name) / "run",
            initial_weights=start,
        )
        return first, second

    @staticmethod
    def _compare_weights(a: Any, b: Any, rtol: float, atol: float) -> dict[str, Any]:
        max_abs, max_rel, bad = 0.0, 0.0, 0
        for x, y in zip(a.model.get_weights(), b.model.get_weights(), strict=True):
            d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
            ref = np.abs(np.asarray(x, np.float64))
            max_abs = max(max_abs, float(d.max()))
            big = ref > 1e-3
            if big.any():
                max_rel = max(max_rel, float((d[big] / ref[big]).max()))
            bad += int(np.count_nonzero(d > atol + rtol * ref))
        return {
            "max_abs_diff": max_abs,
            "max_rel_diff_where_abs_gt_1e-3": max_rel,
            "violations": bad,
        }

    def _check_xla(self) -> dict[str, Any]:
        log(f"check xla: {XLA_STEPS} steps")
        plain, xla = self._trainer_pair(
            "check_xla",
            {"jit_compile": False, "steps_per_execution": 1},
            {"jit_compile": True, "steps_per_execution": 1},
        )
        # One pass only: the dataset reshuffles on every iteration.
        elements = list(plain.train_dataset.take(XLA_STEPS))
        losses = []
        for trainer in (plain, xla):
            values = [trainer.train_step(x, y)["loss"] for x, y in elements]
            losses.append(np.asarray(values))
        step_rel = np.abs(losses[0] - losses[1]) / np.maximum(np.abs(losses[0]), 1e-30)
        mean_rel = abs(losses[0].mean() - losses[1].mean()) / max(abs(losses[0].mean()), 1e-30)
        weights = self._compare_weights(plain, xla, XLA_RTOL, XLA_ATOL)
        steps = [int(plain.optimizer.iterations), int(xla.optimizer.iterations)]
        # Criterion as for training in plan section 4.6: weights rtol/atol and the mean
        # loss rtol. The per-step relative loss difference is reported only: a single
        # squared energy error can be close to 0, where a relative error is meaningless.
        return {
            "steps": steps,
            "mean_loss_rel_diff": float(mean_rel),
            "step_loss_max_rel_diff": float(step_rel.max()),
            "weights": weights,
            "passed": steps == [XLA_STEPS, XLA_STEPS]
            and weights["violations"] == 0
            and bool(mean_rel <= XLA_RTOL),
        }

    def _check_spe(self) -> dict[str, Any]:
        log("check steps_per_execution 8 vs 1: one epoch")
        one, eight = self._trainer_pair(
            "check_spe",
            {"jit_compile": False, "steps_per_execution": 1},
            {"jit_compile": False, "steps_per_execution": 8},
        )
        r1, r8 = one.train_epoch(1), eight.train_epoch(1)
        weights = self._compare_weights(one, eight, SPE_RTOL, SPE_ATOL)
        loss_rel = abs(r1["train_loss"] - r8["train_loss"]) / max(abs(r1["train_loss"]), 1e-30)
        return {
            "steps": [int(one.optimizer.iterations), int(eight.optimizer.iterations)],
            "train_loss_rel_diff": loss_rel,
            "weights": weights,
            "passed": weights["violations"] == 0 and loss_rel <= SPE_RTOL,
        }


# =========================================================================== main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--impl", choices=("legacy", "new"), required=True)
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    parser.add_argument("--quick", action="store_true", help="200 aspirin / 20 stachyose frames")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1, help="processes (default: all CPUs)"
    )
    parser.add_argument("--repeats", type=int, default=3, help="repetitions per measurement")
    parser.add_argument("--out", type=Path, help="result JSON")
    parser.add_argument("--note", default="", help="free text stored in the metadata")
    parser.add_argument("--check", action="store_true", help="equivalence mode")
    parser.add_argument("--check-dir", type=Path, help="exchange directory of --check")
    parser.add_argument("--keep", action="store_true", help="keep the temporary work directory")
    args = parser.parse_args(argv)
    if args.repeats < 3 and not args.check:
        parser.error("--repeats must be at least 3 (median of >= 3 repetitions)")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.check and args.check_dir is None:
        parser.error("--check needs --check-dir")
    if not args.check and args.out is None:
        parser.error("--out is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    ws = Workspace(args.quick, keep=args.keep)
    bench: LegacyBench | NewBench | None = None
    try:
        if args.impl == "legacy":
            bench = LegacyBench(ws, args.repeats)
        else:
            bench = NewBench(ws, args.repeats, args.workers)
        if args.check:
            args.check_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(bench, LegacyBench):
                bench.export_check(args.check_dir)
                log(f"wrote legacy check data to {args.check_dir}")
                return 0
            report = bench.check(args.check_dir)
            report["quick"] = args.quick
            text = json.dumps(report, indent=2) + "\n"
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(text)
            print(text)
            log("equivalence: " + ("PASSED" if report["passed"] else "FAILED"))
            return 0 if report["passed"] else 1

        suites = SUITES if args.suite == "all" else (args.suite,)
        started = datetime.now(UTC).isoformat(timespec="seconds")
        load_start = loadavg()
        t0 = time.perf_counter()
        results = {suite: getattr(bench, suite)() for suite in suites}
        result = {
            "schema_version": SCHEMA_VERSION,
            "impl": args.impl,
            "metadata": {
                "started": started,
                "runtime_s": time.perf_counter() - t0,
                "note": args.note,
                "loadavg_start": load_start,
                "loadavg_end": loadavg(),
                "command": [Path(sys.executable).name, *sys.argv],
                **environment(),
                **bench.versions(),
            },
            "params": {
                "quick": args.quick,
                "repeats": args.repeats,
                "workers": args.workers,
                "suites": list(suites),
                "profile": str(PROFILE.relative_to(REPO_ROOT)),
                "split": [VAL_FRACTION, TEST_FRACTION, SPLIT_SEED],
                "shuffle_buffer": SHUFFLE_BUFFER,
                "warmup_epochs": WARMUP_EPOCHS,
                "timed_epochs": TIMED_EPOCHS,
                "infer_batch_size": INFER_BATCH,
                "train_variants": [list(v) for v in TRAIN_VARIANTS],
                "legacy_model_seed": LEGACY_MODEL_SEED,
            },
            "workloads": ws.described(),
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        log(f"wrote {args.out} ({result['metadata']['runtime_s']:.0f} s)")
        return 0
    finally:
        if bench is not None:
            bench.close()
        ws.close()


if __name__ == "__main__":
    sys.exit(main())
