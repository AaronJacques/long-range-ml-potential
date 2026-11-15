"""Compare a legacy and a new benchmark result and write ``docs/benchmarks.md``.

Usage::

    python benchmarks/compare.py LEGACY.json NEW.json [--check-equivalence]
        [--markdown docs/benchmarks.md]

The two files are recognised by their ``impl`` field, so the order does not
matter. If several files of one implementation are given (e.g. a shell glob over
``benchmarks/results/`` with reference runs of several days), the most recent run
(``metadata.started``) is used.

``--check-equivalence`` runs ``run_benchmarks.py --check`` twice in subprocesses,
first with ``TF_USE_LEGACY_KERAS=1`` (legacy export) and then with Keras 3 (the
``lrmlp`` side compares), on the workload of the result files (``--quick`` if they
are quick runs). The exit code is 1 if an equivalence tolerance is violated, 2 on
input errors. Missed performance targets do not change the exit code; they are
reported.

The Markdown file is generated completely, except for the block between
``<!-- BEGIN MANUAL ANALYSIS -->`` and ``<!-- END MANUAL ANALYSIS -->``, which is
taken over from an existing file so that hand-written analysis survives a rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RUNNER = HERE / "run_benchmarks.py"

BEGIN_MANUAL = "<!-- BEGIN MANUAL ANALYSIS -->"
END_MANUAL = "<!-- END MANUAL ANALYSIS -->"
DEFAULT_MANUAL = "_No manual analysis yet._"

# Default-change rule for train.jit_compile and train.steps_per_execution (plan section 4.4).
MIN_GAIN = 0.15

WORKLOAD_LABEL = {"aspirin": "aspirin-like", "stachyose": "stachyose-like"}


# =========================================================================== loading


def load_results(paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_impl: dict[str, list[dict[str, Any]]] = {"legacy": [], "new": []}
    for path in paths:
        data = json.loads(Path(path).read_text())
        impl = data.get("impl")
        if impl not in by_impl:
            raise SystemExit(f"{path}: unknown impl {impl!r}")
        data["_path"] = str(path)
        by_impl[impl].append(data)
    picked = {}
    for impl, runs in by_impl.items():
        if not runs:
            raise SystemExit(f"no {impl} result among {[str(p) for p in paths]}")
        runs.sort(key=lambda d: d["metadata"]["started"])
        if len(runs) > 1:
            print(
                f"note: {len(runs)} {impl} results, using the latest: {runs[-1]['_path']}",
                file=sys.stderr,
            )
        picked[impl] = runs[-1]
    legacy, new = picked["legacy"], picked["new"]
    if legacy["params"]["quick"] != new["params"]["quick"]:
        raise SystemExit("legacy and new results differ in --quick")
    for name, w in legacy["workloads"].items():
        other = new["workloads"].get(name)
        if other is not None and other["sha256"] != w["sha256"]:
            raise SystemExit(f"workload {name} differs between the two runs")
    return legacy, new


def get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


# =========================================================================== equivalence


def run_equivalence(quick: bool, workers: int | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lrmlp_equiv_") as td:
        exchange = Path(td) / "exchange"
        report = Path(td) / "equivalence.json"
        common = ["--check", "--check-dir", str(exchange)] + (["--quick"] if quick else [])
        if workers:
            common += ["--workers", str(workers)]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TF_USE_LEGACY_KERAS": "1"}
        print("equivalence: legacy export ...", file=sys.stderr, flush=True)
        subprocess.run(
            [sys.executable, str(RUNNER), "--impl", "legacy", *common], env=env, check=True
        )
        env.pop("TF_USE_LEGACY_KERAS")
        print("equivalence: lrmlp comparison ...", file=sys.stderr, flush=True)
        proc = subprocess.run(
            [sys.executable, str(RUNNER), "--impl", "new", *common, "--out", str(report)],
            env=env,
            stdout=subprocess.DEVNULL,
        )
        if not report.is_file():
            raise SystemExit(f"equivalence check failed without a report (exit {proc.returncode})")
        return json.loads(report.read_text())


# =========================================================================== metrics


def fmt_rate(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f}" if v >= 100 else f"{v:,.1f}"


def fmt_bytes(v: float | None) -> str:
    if v is None:
        return "—"
    for unit, scale in (("MB", 1e6), ("kB", 1e3)):
        if v >= scale:
            return f"{v / scale:.1f} {unit}"
    return f"{v:.0f} B"


def fmt_factor(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}×"


def summary_rows(legacy: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per metric: legacy, new, factor (> 1 = new better), target."""
    L, N = legacy["results"], new["results"]
    rows: list[dict[str, Any]] = []

    def rate_row(category: str, workload: str, lv: Any, nv: Any, target: float | None) -> None:
        factor = nv / lv if lv and nv else None
        rows.append(
            {
                "category": category,
                "workload": workload,
                "legacy": fmt_rate(lv),
                "new": fmt_rate(nv),
                "factor": factor,
                "target": target,
            }
        )

    def size_row(category: str, workload: str, lv: Any, nv: Any, target: float | None) -> None:
        factor = lv / nv if lv and nv else None
        rows.append(
            {
                "category": category,
                "workload": workload,
                "legacy": fmt_bytes(lv),
                "new": fmt_bytes(nv),
                "factor": factor,
                "target": target,
            }
        )

    for name in ("aspirin", "stachyose"):
        wl = new["workloads"].get(name) or legacy["workloads"].get(name)
        if not wl or get(L, "prep", name) is None or get(N, "prep", name) is None:
            continue
        label = f"{wl['n_frames']}× {WORKLOAD_LABEL[name]} ({wl['grid']})"
        lp, np_ = L["prep"][name], N["prep"][name]
        rate_row(
            "Preprocessing, mol/s, 1 process",
            label,
            get(lp, "serial", "mol_per_s"),
            get(np_, "serial", "mol_per_s"),
            20,
        )
        workers = get(np_, "parallel", "n_workers")
        rate_row(
            f"Preprocessing, mol/s, {workers} processes (legacy: 1)",
            label,
            get(lp, "serial", "mol_per_s"),
            get(np_, "parallel", "mol_per_s"),
            None,
        )
        size_row(
            "Bytes per molecule on disk (factor = smaller)",
            label,
            get(lp, "bytes_per_molecule", "disk"),
            get(np_, "bytes_per_molecule", "disk"),
            3,
        )
        size_row(
            "Bytes per molecule in RAM (factor = smaller)",
            label,
            get(lp, "bytes_per_molecule", "ram"),
            get(np_, "bytes_per_molecule", "ram"),
            3,
        )
    li, ni = get(L, "input", "aspirin"), get(N, "input", "aspirin")
    if li and ni:
        rate_row(
            "Input pipeline, mol/s (1 epoch, no model)",
            f"{ni['epoch']['n']} train molecules",
            get(li, "epoch", "mol_per_s"),
            get(ni, "epoch", "mol_per_s"),
            10,
        )
    lt, nt = get(L, "train", "aspirin"), get(N, "train", "aspirin")
    if lt and nt:
        lrate = get(lt, "variants", "legacy", "window", "steps_per_s")
        label = f"{nt['n_train']} train molecules, |B| = 1"
        defaults = train_defaults(new)
        for v in nt["variants"].values():
            setting = (v["steps_per_execution"], bool(v["jit_compile"]))
            tags = [tag for tag, value in defaults.items() if value == setting]
            name = (
                f"Training, steps/s, steps_per_execution={v['steps_per_execution']}, "
                f"jit_compile={str(v['jit_compile']).lower()}"
            )
            rate_row(
                name + (f" ({', '.join(tags)})" if tags else ""),
                label,
                lrate,
                get(v, "window", "steps_per_s"),
                1.5 if tags else None,
            )
        rate_row(
            "Validation, mol/s (no gradients)",
            f"{nt['n_val']} val molecules",
            get(lt, "validation", "mol_per_s"),
            get(nt, "validation", "mol_per_s"),
            None,
        )
    lf, nf = get(L, "infer", "aspirin"), get(N, "infer", "aspirin")
    if lf and nf:
        label = f"{nf['n_molecules']} molecules"
        rate_row(
            "Inference, mol/s, precomputed descriptors (legacy: 1 molecule per call; "
            f"new: batch {nf['batch_size']})",
            label,
            get(lf, "without_descriptors", "mol_per_s"),
            get(nf, "without_descriptors", "mol_per_s"),
            10,
        )
        rate_row(
            "Inference, mol/s, from positions (legacy: create_dataset + evaluate_model)",
            label,
            get(lf, "with_descriptors", "mol_per_s"),
            get(nf, "with_descriptors", "mol_per_s"),
            None,
        )
        rate_row(
            f"Inference, mol/s, from positions, "
            f"{get(nf, 'with_descriptors_parallel', 'n_workers')} processes",
            label,
            get(lf, "with_descriptors", "mol_per_s"),
            get(nf, "with_descriptors_parallel", "mol_per_s"),
            None,
        )
    for row in rows:
        f, t = row["factor"], row["target"]
        row["met"] = None if t is None or f is None else f >= t
    return rows


def train_defaults(new: dict[str, Any]) -> dict[str, tuple[int, bool]]:
    """Execution settings of the measured code: label -> (steps_per_execution, jit_compile).

    Result files without ``metadata.train_defaults`` (older runs) had K = 1 without XLA.
    """
    recorded = get(new, "metadata", "train_defaults")
    if not recorded:
        return {"default": (1, False)}

    def pair(d: dict[str, Any]) -> tuple[int, bool]:
        return int(d["steps_per_execution"]), bool(d["jit_compile"])

    out = {"TrainConfig default": pair(recorded["TrainConfig"])}
    presets = {pair(d) for d in recorded.get("presets", {}).values()}
    if len(presets) == 1:
        out["thesis presets"] = presets.pop()
    else:
        for name, d in recorded.get("presets", {}).items():
            out[f"preset {name}"] = pair(d)
    return out


def recommendation(new: dict[str, Any], eq: dict[str, Any] | None) -> dict[str, Any]:
    variants = get(new, "results", "train", "aspirin", "variants") or {}

    def rate(k: int, xla: bool) -> float | None:
        return get(variants, f"spe{k}_xla{int(xla)}", "window", "steps_per_s")

    out: dict[str, Any] = {"gains": {}}
    for k in (1, 8):
        a, b = rate(k, False), rate(k, True)
        out["gains"][f"xla_at_spe{k}"] = b / a - 1 if a and b else None
    for xla in (False, True):
        a, b = rate(1, xla), rate(8, xla)
        out["gains"][f"spe8_at_xla{int(xla)}"] = b / a - 1 if a and b else None
    xla_eq = get(eq, "xla", "passed") if eq else None
    spe_eq = get(eq, "steps_per_execution", "passed") if eq else None
    xla_gains = [out["gains"]["xla_at_spe1"], out["gains"]["xla_at_spe8"]]
    spe_gain = out["gains"]["spe8_at_xla0"]
    out["xla_fast"] = all(g is not None and g >= MIN_GAIN for g in xla_gains)
    out["spe_fast"] = spe_gain is not None and spe_gain >= MIN_GAIN
    out["xla_equivalent"] = xla_eq
    out["spe_equivalent"] = spe_eq
    out["recommend_xla"] = bool(out["xla_fast"] and xla_eq)
    out["recommend_spe8"] = bool(out["spe_fast"] and spe_eq)
    return out


# =========================================================================== text output


def print_table(rows: list[dict[str, Any]]) -> None:
    width = max(len(r["category"]) + len(r["workload"]) + 3 for r in rows)
    print(f"{'metric':<{width}}  {'legacy':>12}  {'new':>12}  {'factor':>8}  target  met")
    for r in rows:
        target = f">= {r['target']}" if r["target"] is not None else "-"
        met = {None: "-", True: "yes", False: "NO"}[r["met"]]
        name = f"{r['category']} [{r['workload']}]"
        print(
            f"{name:<{width}}  {r['legacy']:>12}  {r['new']:>12}  "
            f"{fmt_factor(r['factor']):>8}  {target:>6}  {met}"
        )


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def status_lines(legacy: dict[str, Any], new: dict[str, Any]) -> list[str]:
    lines = []
    notes = {d["metadata"].get("note") for d in (legacy, new)} - {""}
    if notes:
        lines += [f"> **Status: {'; '.join(sorted(notes))}.**", ">"]
    if legacy["params"]["quick"]:
        lines += [
            "> **Quick run (`--quick`): reduced workloads, not a reference measurement.**",
            ">",
        ]
    for d in (legacy, new):
        m = d["metadata"]
        lines.append(
            f"> - {d['impl']}: started {m['started']}, load average at start / end "
            f"{m.get('loadavg_start')} / {m.get('loadavg_end')} ({m.get('cpu_count')} CPUs), "
            f"runtime {m['runtime_s']:.0f} s, file `{Path(d['_path']).as_posix()}`."
        )
    return lines


def env_table(legacy: dict[str, Any], new: dict[str, Any]) -> str:
    keys = [
        ("cpu", "CPU"),
        ("cpu_count", "logical CPUs"),
        ("memory_bytes", "memory"),
        ("platform", "OS"),
        ("python", "Python"),
        ("numpy", "NumPy"),
        ("tensorflow", "TensorFlow"),
        ("keras", "Keras"),
        ("pandas", "pandas"),
        ("lrmlp", "lrmlp"),
        ("intra_op_threads", "TF intra-op threads (0 = default)"),
        ("git_sha", "git SHA (working tree)"),
        ("git_dirty_src_or_benchmarks", "uncommitted changes in src/ or benchmarks/"),
        ("legacy_commit", "legacy commit"),
    ]
    rows = []
    for key, label in keys:
        lv, nv = legacy["metadata"].get(key, "—"), new["metadata"].get(key, "—")
        if key == "memory_bytes":
            lv = f"{lv / 2**30:.0f} GiB" if isinstance(lv, int) else lv
            nv = f"{nv / 2**30:.0f} GiB" if isinstance(nv, int) else nv
        rows.append([label, f"`{lv}`", f"`{nv}`"])
    return md_table(["", "legacy process", "new process"], rows)


def detail_tables(legacy: dict[str, Any], new: dict[str, Any]) -> list[str]:
    L, N = legacy["results"], new["results"]
    out: list[str] = []

    def t(d: Any, *keys: str) -> str:
        v = get(d, *keys)
        return "—" if v is None else f"{v:.3f}"

    if "prep" in L and "prep" in N:
        rows = []
        for name in ("aspirin", "stachyose"):
            lp, np_ = get(L, "prep", name), get(N, "prep", name)
            if not lp or not np_:
                continue
            rows.append(
                [
                    WORKLOAD_LABEL[name],
                    t(lp, "serial", "median_s"),
                    t(np_, "serial", "median_s"),
                    t(np_, "parallel", "median_s"),
                    f"{lp['n_max']} / {np_['n_max']}",
                    fmt_bytes(lp["bytes_per_molecule"]["disk"])
                    + " / "
                    + fmt_bytes(np_["bytes_per_molecule"]["disk"]),
                    fmt_bytes(lp["bytes_per_molecule"]["ram"])
                    + " / "
                    + fmt_bytes(np_["bytes_per_molecule"]["ram"]),
                ]
            )
        out.append("### Preprocessing\n")
        out.append(
            md_table(
                [
                    "workload",
                    "legacy s (1 proc.)",
                    "new s (1 proc.)",
                    "new s (N proc.)",
                    "N_max (loc, lr) legacy / new",
                    "disk legacy / new",
                    "RAM legacy / new",
                ],
                rows,
            )
        )
        out.append(
            "\nTimes are medians of the repetitions and include reading the `.npz`, the "
            "descriptors, the split and writing the dataset. Legacy disk = the three "
            "`.pkl.gzip` files; legacy RAM = NumPy bytes of all DataFrame cells after "
            "`pd.read_pickle` (lists via `np.asarray`). New disk = the dataset directory; "
            "new RAM = bytes of all `.npy` arrays (memory-mapped).\n"
        )
    lt, nt = get(L, "train", "aspirin"), get(N, "train", "aspirin")
    if lt and nt:
        rows = []
        lv = lt["variants"]["legacy"]
        rows.append(
            [
                "legacy",
                "1",
                "false",
                ", ".join(f"{x:.2f}" for x in lv["epoch_times_s"]),
                fmt_rate(lv["window"]["steps_per_s"]),
            ]
        )
        for v in nt["variants"].values():
            rows.append(
                [
                    "new",
                    str(v["steps_per_execution"]),
                    str(v["jit_compile"]).lower(),
                    ", ".join(f"{x:.2f}" for x in v["epoch_times_s"]),
                    fmt_rate(v["window"]["steps_per_s"]),
                ]
            )
        out.append("### Training\n")
        out.append(
            md_table(
                [
                    "impl",
                    "steps_per_execution",
                    "jit_compile",
                    "epoch times s (first = warm-up)",
                    "steps/s (median of 2-epoch windows)",
                ],
                rows,
            )
        )
        out.append(
            "\nEpoch times cover the training loop only (legacy: the iteration over the "
            "training dataset inside the unchanged `start_training`, measured by a wrapper "
            "around the dataset; new: `Trainer.train_epoch`). Validation, checkpoints and "
            "logging are excluded and validation is reported separately.\n"
        )
    lf, nf = get(L, "infer", "aspirin"), get(N, "infer", "aspirin")
    if lf and nf:
        workers = get(nf, "with_descriptors_parallel", "n_workers")
        spec = [
            ("legacy: `evaluate.py` loop (1 molecule per call)", lf, "without_descriptors"),
            ("legacy: `evaluate_model` incl. `load_model` and dataset", lf, "evaluate_model_total"),
            ("legacy: `create_dataset` (descriptors, pickles)", lf, "create_dataset"),
            ("new: `predict_descriptors` (batch 256)", nf, "without_descriptors"),
            ("new: `Potential.descriptors` only", nf, "descriptors_only"),
            ("new: `predict` from positions, 1 process", nf, "with_descriptors"),
            (
                f"new: `predict` from positions, `n_workers={workers}`",
                nf,
                "with_descriptors_parallel",
            ),
        ]
        rows = [
            [label, t(d, key, "median_s"), fmt_rate(get(d, key, "mol_per_s"))]
            for label, d, key in spec
        ]
        out.append("### Inference\n")
        out.append(md_table(["measurement", "median s", "mol/s"], rows))
        out.append(
            "\nThe legacy loop time excludes the first molecule (tracing of the "
            "`tf.function`); the new measurements run after one warm-up call.\n"
        )
    return out


def equivalence_section(eq: dict[str, Any] | None) -> list[str]:
    if eq is None:
        return [
            "Not run in this invocation (`compare.py --check-equivalence` runs it). "
            "The golden tests (`pytest -m golden`) cover the same comparisons on the "
            "fixtures."
        ]
    tol = eq["tolerances"]
    out = [
        f"Overall: **{'PASSED' if eq['passed'] else 'FAILED'}** "
        f"({'quick' if eq.get('quick') else 'full'} workload).",
        "",
    ]
    rows = []
    for name, w in eq["workloads"].items():
        d, e = w["descriptors"], w["energies"]
        per = d["per_key"]
        rows.append(
            [
                f"{w['n_frames']}× {WORKLOAD_LABEL[name]}",
                f"{d['n_max_legacy']} / {d['n_max_new']}",
                ", ".join(
                    f"{k} {per[k]['max_abs_diff']:.1e} ({per[k]['violations']})" for k in per
                ),
                f"{e['max_abs_diff']:.2e} / {e['max_rel_diff']:.1e} "
                f"(|E| ≤ {e['max_abs_energy']:.3g}, {e['violations']} violations)",
                "yes" if d["passed"] and e["passed"] else "**no**",
            ]
        )
    out.append(
        md_table(
            [
                "workload",
                "N_max legacy / new",
                "descriptors: max abs diff (violations)",
                "energies, same weights: max abs / rel diff",
                "passed",
            ],
            rows,
        )
    )
    x, s = eq["xla"], eq["steps_per_execution"]
    out += [
        "",
        f"* Descriptors: atol = rtol = {tol['descriptors']['atol']:g}, integer columns "
        "exact, after reordering the legacy rows (cell traversal order) into input order. "
        'Model: `legacy-head` architecture, `assignment = "legacy"`.',
        f"* Energies: legacy model built with seed, weights exported in construction order "
        f"and loaded with `lrmlp.model.assign_legacy_weights`; rtol "
        f"{tol['energies']['rtol']:g}, atol {tol['energies']['atol_per_atom']:g} per atom. "
        "New energies come from `Potential.predict` (positions → descriptors → batched "
        "model).",
        f"* XLA: {x['steps'][0]} training steps with `jit_compile=false` and `true` from the "
        f"same weights and data order: max rel. weight diff "
        f"{x['weights']['max_rel_diff_where_abs_gt_1e-3']:.1e} (|w| > 1e-3), max abs "
        f"{x['weights']['max_abs_diff']:.1e}, rel. diff of the mean loss "
        f"{x['mean_loss_rel_diff']:.1e} (largest per-step rel. loss diff "
        f"{x['step_loss_max_rel_diff']:.1e}, reported only)"
        f" → {'passed' if x['passed'] else '**failed**'} (rtol {tol['xla']['rtol']:g}).",
        f"* `steps_per_execution` 8 vs 1, one epoch ({s['steps'][0]} steps): max abs weight "
        f"diff {s['weights']['max_abs_diff']:.1e}, rel. loss diff "
        f"{s['train_loss_rel_diff']:.1e} → {'passed' if s['passed'] else '**failed**'}.",
    ]
    return out


def recommendation_section(rec: dict[str, Any], new: dict[str, Any]) -> list[str]:
    g = rec["gains"]

    def pct(v: float | None) -> str:
        return "—" if v is None else f"{v * 100:+.0f} %"

    variants = get(new, "results", "train", "aspirin", "variants") or {}
    rows = []
    for k in (1, 8):
        rows.append(
            [
                str(k),
                fmt_rate(get(variants, f"spe{k}_xla0", "window", "steps_per_s")),
                fmt_rate(get(variants, f"spe{k}_xla1", "window", "steps_per_s")),
                pct(g[f"xla_at_spe{k}"]),
            ]
        )
    eq_text = {None: "not checked", True: "passed", False: "failed"}
    out = [
        md_table(
            ["steps_per_execution", "jit_compile=false", "jit_compile=true", "XLA gain"], rows
        ),
        "",
        f"Gain of `steps_per_execution = 8` over 1: {pct(g['spe8_at_xla0'])} without XLA, "
        f"{pct(g['spe8_at_xla1'])} with XLA.",
        "",
        f"Rule: a default changes only with a gain of at least {MIN_GAIN * 100:.0f} % and a "
        "passed equivalence check (plan section 4.4). The benchmark itself never changes "
        "defaults. Settings of the measured code: "
        + "; ".join(
            f"{tag} `steps_per_execution = {k}`, `jit_compile = {str(x).lower()}`"
            for tag, (k, x) in train_defaults(new).items()
        )
        + ".",
        "",
        f"* `train.jit_compile`: gain {pct(g['xla_at_spe1'])} / {pct(g['xla_at_spe8'])} "
        f"(K = 1 / 8), equivalence {eq_text[rec['xla_equivalent']]} → "
        + (
            "**recommend `jit_compile = true`** (where it is set: see the settings above)."
            if rec["recommend_xla"]
            else "keep `jit_compile = false`."
            if rec["xla_equivalent"] is not None or not rec["xla_fast"]
            else "fast enough, but run the equivalence check before recommending it."
        ),
        f"* `train.steps_per_execution`: gain {pct(g['spe8_at_xla0'])}, equivalence "
        f"{eq_text[rec['spe_equivalent']]} → "
        + (
            "**recommend `steps_per_execution = 8`**."
            if rec["recommend_spe8"]
            else "keep `steps_per_execution = 1`."
            if rec["spe_equivalent"] is not None or not rec["spe_fast"]
            else "fast enough, but run the equivalence check before recommending it."
        ),
        "",
        "XLA compiles the step once per input shape. The atom axis is `None` in the "
        "signature, so a dataset with molecules of different sizes would compile once per "
        "size; every dataset of the thesis holds a single molecule (one shape), the "
        "benchmark as well.",
    ]
    return out


STATIC_METHOD = """\
* **Same runtime.** Both implementations run in the same virtual environment (`.venv`,
  same TensorFlow build and thread settings). The legacy code (`Neural-Network/` at
  commit `9413bfd`, unchanged) is exported with `git archive` into a temporary
  directory (`tests/golden/legacy.py`) and runs in its own process with
  `TF_USE_LEGACY_KERAS=1` (tf_keras); `lrmlp` runs in a separate process with Keras 3.
* **Workloads.** Deterministic synthetic molecules from `tests/golden/synthetic.py`
  (Lennard-Jones energies): 2000 aspirin-like frames (21 atoms, grid G-A: Δg = 1 Å,
  l_max = 6) and 200 stachyose-like frames (87 atoms, G-C: Δg = 2 Å, l_max = 4);
  `--quick` uses 200 / 20. Both processes generate the same arrays (checked by a
  SHA-256 over E, F, R, z).
* **Profile.** Grid assignment `legacy` and the network of `configs/legacy-head.toml`
  (embedding 64-128-128-128, K1 = 30 / 6, K2 = 4 / 2, fitting 128 + 2 residual blocks,
  dropout 0) for both implementations; N_max is taken from the data. Split 0.8 / 0.1 /
  0.1, shuffle buffer 1000.
* **Timing.** `time.perf_counter`, median of at least 3 repetitions (`--repeats`).
  Preprocessing: whole `create_dataset` / `build_dataset` call. Input pipeline: one
  pass over the training split without a model (after one warm-up pass; dataset
  construction reported separately as `load`). Training: one warm-up epoch, then 3
  windows of 2 epochs on the same model, energy loss only, one molecule per step
  (|B| = 1). Inference: 1000 molecules; legacy as in `evaluate.py` (one
  `tf.function` call per molecule on the preprocessed test pickle), new with
  `Potential.predict_descriptors` / `Potential.predict` and `batch_size = 256`.
* **Observation hooks.** The legacy code is not modified. For the training and
  inference loops the dataset returned by `create_tf_dataset` is wrapped by an
  iterator that records time stamps, and `Training.get_model` is replaced by a call of
  the unchanged `Model.get_model` after `clear_session()` and a fixed seed.
"""

STATIC_MIXED_PRECISION = """\
Mixed precision is not used and not benchmarked (plan section 4.4). The molecule energy
is the sum of the atomic energies and has a magnitude of |E| ≈ 4·10⁵ kcal/mol for
aspirin. float16 overflows at 65 504; bfloat16 has 8 significand bits, i.e. a
resolution of about 2·10³ kcal/mol at that magnitude, three orders of magnitude above
the target MAE of ≈ 1.7 kcal/mol. A loss-scaling optimizer does not help, because the
problem is the representable precision of the summed energy, not gradient underflow.
"""

STATIC_SCOPE = """\
Training with more than one molecule per optimizer step (|B| > 1) is not part of this
benchmark and not part of the code: the thesis trains with |B| = 1 (p. 51) and names
larger batches as future work (Chapter 7.4). The batched paths measured here
(validation, inference) do not change the result, only the number of model calls.
"""


def reproduce_section(legacy: dict[str, Any], new: dict[str, Any]) -> str:
    quick = " --quick" if legacy["params"]["quick"] else ""
    return f"""\
```sh
source .venv/bin/activate
python benchmarks/run_benchmarks.py --impl new --suite all{quick} --out NEW.json
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 \\
    python benchmarks/run_benchmarks.py --impl legacy --suite all{quick} --out LEGACY.json
python benchmarks/compare.py LEGACY.json NEW.json --check-equivalence --markdown docs/benchmarks.md
```

Runtime of the runs on this page: legacy {legacy["metadata"]["runtime_s"]:.0f} s, new
{new["metadata"]["runtime_s"]:.0f} s, plus the equivalence check. See
`benchmarks/README.md`.
"""


def manual_block(path: Path) -> str:
    if path.is_file():
        text = path.read_text()
        if BEGIN_MANUAL in text and END_MANUAL in text:
            return text.split(BEGIN_MANUAL, 1)[1].split(END_MANUAL, 1)[0].strip("\n")
    return DEFAULT_MANUAL


def markdown(
    legacy: dict[str, Any],
    new: dict[str, Any],
    rows: list[dict[str, Any]],
    eq: dict[str, Any] | None,
    rec: dict[str, Any],
    manual: str,
) -> str:
    table = md_table(
        ["metric", "workload", "legacy", "new", "factor new vs. legacy", "target", "met"],
        [
            [
                r["category"],
                r["workload"],
                r["legacy"],
                r["new"],
                fmt_factor(r["factor"]),
                "—" if r["target"] is None else f"≥ {r['target']}×",
                {None: "—", True: "yes", False: "**no**"}[r["met"]],
            ]
            for r in rows
        ],
    )
    missed = [r for r in rows if r["met"] is False]
    missed_lines = [
        f"* {r['category']} ({r['workload']}): {fmt_factor(r['factor'])} instead of "
        f"≥ {r['target']}×."
        for r in missed
    ] or ["All targets met."]
    parts = [
        "# Benchmarks: legacy code vs. lrmlp",
        "",
        "<!-- Generated by benchmarks/compare.py; edit only the manual analysis block. -->",
        "",
        *status_lines(legacy, new),
        "",
        "Measured with `benchmarks/run_benchmarks.py` on the same TensorFlow runtime for "
        "the unchanged legacy code and for `lrmlp`. Factors > 1 mean the new code is "
        "faster (rates) or smaller (bytes).",
        "",
        "## Results",
        "",
        table,
        "",
        *detail_tables(legacy, new),
        "## Equivalence",
        "",
        *equivalence_section(eq),
        "",
        "## `steps_per_execution` and `jit_compile`",
        "",
        *recommendation_section(rec, new),
        "",
        "## Missed targets",
        "",
        *missed_lines,
        "",
        BEGIN_MANUAL,
        manual,
        END_MANUAL,
        "",
        "## Method",
        "",
        STATIC_METHOD,
        "## Environment",
        "",
        env_table(legacy, new),
        "",
        "## Mixed precision",
        "",
        STATIC_MIXED_PRECISION,
        "## Scope",
        "",
        STATIC_SCOPE,
        "## Reproducing",
        "",
        reproduce_section(legacy, new),
    ]
    return "\n".join(parts)


# =========================================================================== main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", type=Path, help="legacy and new result JSON files")
    parser.add_argument("--check-equivalence", action="store_true")
    parser.add_argument("--equivalence-json", type=Path, help="use this equivalence report")
    parser.add_argument("--save-equivalence", type=Path, help="also write the report here")
    parser.add_argument("--markdown", type=Path, help="write the Markdown report here")
    args = parser.parse_args(argv)

    legacy, new = load_results(args.results)
    rows = summary_rows(legacy, new)
    print_table(rows)

    eq = None
    if args.equivalence_json:
        eq = json.loads(args.equivalence_json.read_text())
    elif args.check_equivalence:
        eq = run_equivalence(bool(legacy["params"]["quick"]), new["params"].get("workers"))
    if eq is not None:
        print(f"\nequivalence: {'PASSED' if eq['passed'] else 'FAILED'}")
        if args.save_equivalence:
            args.save_equivalence.write_text(json.dumps(eq, indent=2) + "\n")
    rec = recommendation(new, eq)
    print(
        f"recommendation: jit_compile={'true' if rec['recommend_xla'] else 'false'}, "
        f"steps_per_execution={8 if rec['recommend_spe8'] else 1} "
        f"(gains {', '.join(f'{k} {v:+.0%}' for k, v in rec['gains'].items() if v is not None)})"
    )
    if args.markdown:
        text = markdown(legacy, new, rows, eq, rec, manual_block(args.markdown))
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text)
        print(f"wrote {args.markdown}")
    if eq is not None and not eq["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
