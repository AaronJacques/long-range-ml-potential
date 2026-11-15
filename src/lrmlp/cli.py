"""Command line interface ``lrmlp`` (also ``python -m lrmlp``).

The subcommands replace the ``__main__`` blocks of the legacy scripts
``Neural-Network/Dataset.py``, ``Training.py`` and ``evaluate.py`` and expose the
rest of the library:

========================  =====================================================
``lrmlp prepare``         preprocess raw trajectories into a descriptor dataset
``lrmlp train``           train a model on a prepared dataset
``lrmlp evaluate``        energy MAE of a checkpoint on a split (Tab. 5.3)
``lrmlp predict``         energies (and forces) of new frames
``lrmlp tune``            hyperparameter grid search (``lrmlp.tuning``)
``lrmlp plot-log``        learning curves (``lrmlp.analysis.plots``)
``lrmlp plot-weights``    weight distributions, Fig. 5.3 (``lrmlp.analysis.weights``)
``lrmlp info``            show dataset or run metadata
========================  =====================================================

Paths
-----
Path arguments of the CLI itself (``--config``, ``--dataset``, ``--run``,
``--run-dir``, ``--output``, ``--input``, ``--json``, ...) are relative to the
current working directory. Paths *inside* a configuration file and paths given
with ``--set`` (e.g. ``--set train.run_root=runs``) are relative to the directory
of the configuration file (see :func:`lrmlp.config.load_config`). No command
depends on the working directory otherwise (legacy bug B10).

Exit codes
----------
``0`` success, ``1`` runtime error (message without traceback unless
``--log-level DEBUG``), ``2`` usage error (invalid arguments, or the module of a
lazily loaded command is not available).

TensorFlow is imported only inside the subcommands that need it, so
``lrmlp --help`` and ``lrmlp --version`` start quickly.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import sys
import textwrap
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lrmlp import __version__

__all__ = ["build_parser", "main"]

logger = logging.getLogger("lrmlp.cli")

#: Log levels accepted by ``--log-level``.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
#: Format of the log records written to stderr.
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
#: Splits of a prepared dataset.
SPLITS = ("test", "val", "train")
#: Required arrays of a ``predict`` input file (``docs/data-format.md``).
PREDICT_INPUT_KEYS = ("R", "z")

_PATH_NOTE = (
    "Path arguments are relative to the current working directory; relative paths "
    "inside the config file and in --set values are relative to the config file."
)


class UsageError(Exception):
    """An error in the use of the CLI (exit code 2)."""


# --------------------------------------------------------------------------- argument types


def _path(text: str) -> Path:
    """Turn a CLI path argument into an absolute path relative to the CWD."""
    if not text:
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(text).expanduser().resolve()


def _json_target(text: str) -> str:
    """``-`` (standard output) or a non-empty path, kept as given."""
    if not text:
        raise argparse.ArgumentTypeError("path must not be empty")
    return text


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return value


def _epoch(text: str) -> str | int:
    """``best``, ``last`` or a non-negative epoch number."""
    if text in ("best", "last"):
        return text
    if text.isdigit():
        return int(text)
    raise argparse.ArgumentTypeError(f"epoch must be 'best', 'last' or an integer, got {text!r}")


def _override(text: str) -> str:
    key, sep, _ = text.partition("=")
    if sep != "=" or not key.strip():
        raise argparse.ArgumentTypeError(f"expected key=value (e.g. train.epochs=2), got {text!r}")
    return text


# --------------------------------------------------------------------------- parser


class _Formatter(argparse.ArgumentDefaultsHelpFormatter):
    """Wrap each line of a description separately and show only meaningful defaults."""

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return "\n".join(
            textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent)
            if line.strip()
            else ""
            for line in text.splitlines()
        )

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default in (None, False, [], argparse.SUPPRESS):
            return action.help
        return super()._get_help_string(action)


def _add_config(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config", "-c", type=_path, required=True, metavar="C", help="experiment TOML file"
    )
    p.add_argument(
        "--set",
        dest="overrides",
        type=_override,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=2 (repeatable; the value "
        "is a TOML literal; relative paths are relative to the config file)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``lrmlp`` with all subcommands.

    Returns
    -------
    argparse.ArgumentParser
        The parser; each subcommand stores its handler in the ``handler`` attribute
        of the parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="lrmlp",
        description=(
            "Machine learning potentials with long-range interaction (code of the "
            "master's thesis).\n\n" + _PATH_NOTE
        ),
        epilog="Run 'lrmlp <command> --help' for the options of a command.",
        formatter_class=_Formatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        type=str.upper,
        help="verbosity of the log on stderr; DEBUG also shows tracebacks",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    def command(name: str, help_text: str, description: str) -> argparse.ArgumentParser:
        return sub.add_parser(
            name,
            help=help_text,
            description=f"{description}\n\n{_PATH_NOTE}",
            formatter_class=_Formatter,
        )

    # prepare
    p = command(
        "prepare",
        "preprocess raw .npz trajectories into a descriptor dataset",
        "Compute the local and long-range descriptors of all frames, split them "
        "80/10/10 and store the dataset (preprocessing is run once, cf. thesis "
        "Sec. 4.3.2 and p. 40). Replaces 'python Dataset.py'.",
    )
    _add_config(p)
    p.add_argument(
        "--output",
        "-o",
        type=_path,
        metavar="DIR",
        help="dataset directory (default: data.output_dir of the config)",
    )
    p.add_argument(
        "--workers",
        type=_non_negative_int,
        metavar="N",
        help="worker processes, 0 = all CPUs, 1 = none (default: data.n_workers)",
    )
    p.add_argument("--overwrite", action="store_true", help="replace an existing dataset")
    p.set_defaults(handler=_cmd_prepare)

    # train
    p = command(
        "train",
        "train a model on a prepared dataset",
        "Train the atomic energy network (loss Eq. 4.21, learning-rate schedule "
        "Eq. 4.22; one molecule per step). The epoch with the lowest validation "
        "energy MSE is recorded as the best epoch (p. 52). Replaces 'python Training.py'.",
    )
    _add_config(p)
    p.add_argument(
        "--dataset",
        "-d",
        type=_path,
        metavar="DIR",
        help="prepared dataset (default: data.output_dir of the config)",
    )
    p.add_argument(
        "--run-dir",
        type=_path,
        metavar="DIR",
        help="empty or new run directory (default: a new <name>-<timestamp> "
        "directory below train.run_root)",
    )
    p.set_defaults(handler=_cmd_train)

    # evaluate
    p = command(
        "evaluate",
        "energy MAE of a checkpoint on a dataset split",
        "Evaluate a checkpoint on a split of the run's dataset. With the defaults "
        "this is the test MAE of the thesis (best validation epoch, Tab. 5.3). "
        "Replaces 'python evaluate.py'. The result is also stored as "
        "evaluation_<split>_<epoch>.json in the run directory.",
    )
    p.add_argument("--run", "-r", type=_path, required=True, metavar="R", help="run directory")
    p.add_argument("--split", choices=SPLITS, default="test", help="dataset split")
    p.add_argument(
        "--epoch",
        type=_epoch,
        default="best",
        metavar="{best,last,N}",
        help="checkpoint: best, last or an epoch number",
    )
    p.add_argument(
        "--dataset",
        "-d",
        type=_path,
        metavar="DIR",
        help="dataset directory (default: the path stored in the run's dataset.json)",
    )
    p.add_argument("--batch-size", type=_positive_int, default=256, help="molecules per model call")
    p.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="evaluate even if the dataset is not the one the run was trained on "
        "(differs from the run's dataset.json); logs a warning and marks the result",
    )
    p.add_argument(
        "--json",
        dest="json_out",
        type=_json_target,
        metavar="OUT",
        help="also write the metrics as JSON to OUT; '-' prints JSON to stdout instead "
        "of the table",
    )
    p.set_defaults(handler=_cmd_evaluate)

    # predict
    p = command(
        "predict",
        "predict energies (and forces) of new frames",
        "Predict molecular energies E = sum_i E_i (Eq. 2.7, p. 21) with a trained run.\n"
        "Input: .npz with R (S, N, 3) in Angstrom and z (N,).\n"
        "Output: .npz with E (S,) in kcal/mol and, for force models (experimental), "
        "F (S, N, 3).",
    )
    p.add_argument("--run", "-r", type=_path, required=True, metavar="R", help="run directory")
    p.add_argument(
        "--input", "-i", type=_path, required=True, metavar="IN.npz", help="input positions"
    )
    p.add_argument(
        "--output", "-o", type=_path, required=True, metavar="OUT.npz", help="output file"
    )
    p.add_argument(
        "--epoch",
        type=_epoch,
        default="best",
        metavar="{best,last,N}",
        help="checkpoint: best, last or an epoch number",
    )
    p.add_argument("--batch-size", type=_positive_int, default=256, help="molecules per model call")
    p.add_argument(
        "--workers",
        type=_non_negative_int,
        default=1,
        metavar="N",
        help="worker processes for the descriptors, 0 = all CPUs, 1 = none",
    )
    p.set_defaults(handler=_cmd_predict)

    # tune
    p = command(
        "tune",
        "hyperparameter grid search (needs lrmlp.tuning)",
        "Train one model per point of a hyperparameter grid (cf. the parameter "
        "study behind Tab. 5.2). Replaces 'HyperparameterTester.py'. Needs "
        "TensorBoard (extra 'tune'); --dry-run works without it.",
    )
    p.add_argument(
        "--config", "-c", type=_path, required=True, metavar="C", help="base experiment TOML"
    )
    p.add_argument(
        "--space", "-s", type=_path, required=True, metavar="S", help="search space TOML"
    )
    p.add_argument(
        "--max-trials", type=_positive_int, metavar="N", help="upper bound on the trials"
    )
    p.add_argument("--dry-run", action="store_true", help="only list the trials, do not train")
    p.set_defaults(handler=_cmd_tune)

    # plot-log
    p = command(
        "plot-log",
        "plot learning curves (needs lrmlp.analysis)",
        "Plot the learning curves of a training log (cf. thesis Fig. 5.1 and 5.2).",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", "-r", type=_path, metavar="R", help="run directory")
    source.add_argument("--log", type=_path, metavar="CSV", help="training_log.csv")
    p.add_argument(
        "--output", "-o", type=_path, metavar="DIR", help="output directory of the figures"
    )
    p.add_argument(
        "--skip", type=_non_negative_int, default=0, metavar="N", help="skip the first N epochs"
    )
    p.set_defaults(handler=_cmd_plot_log)

    # plot-weights
    p = command(
        "plot-weights",
        "compare weight distributions of two runs (needs lrmlp.analysis)",
        "Plot the weight distributions of two runs, e.g. with and without long range "
        "(thesis Fig. 5.3). Replaces 'weigtht_distribution.py'.",
    )
    p.add_argument("--run-a", type=_path, required=True, metavar="R1", help="first run")
    p.add_argument("--run-b", type=_path, required=True, metavar="R2", help="second run")
    p.add_argument("--output", "-o", type=_path, required=True, metavar="PDF", help="output figure")
    p.add_argument("--epoch-a", type=_epoch, default="best", help="checkpoint of the first run")
    p.add_argument("--epoch-b", type=_epoch, default="best", help="checkpoint of the second run")
    p.set_defaults(handler=_cmd_plot_weights)

    # info
    p = command(
        "info",
        "show the metadata of a dataset or a run",
        "Show a prepared dataset (metadata, padding N_max, split sizes) or a run "
        "(core configuration, best epoch, checkpoints, evaluations).",
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", "-d", type=_path, metavar="D", help="dataset directory")
    target.add_argument("--run", "-r", type=_path, metavar="R", help="run directory")
    p.add_argument(
        "--json",
        dest="json_out",
        type=_json_target,
        nargs="?",
        const="-",
        metavar="OUT",
        help="write JSON instead of the table, to OUT or (without OUT) to stdout",
    )
    p.set_defaults(handler=_cmd_info)
    return parser


# --------------------------------------------------------------------------- output helpers


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list | tuple):
        return ", ".join(_format_value(v) for v in value) if value else "-"
    if value is None:
        return "-"
    return str(value)


def _table(rows: Sequence[tuple[str, Any]], title: str | None = None) -> str:
    """Render ``(key, value)`` rows as an aligned two-column table."""
    width = max((len(k) for k, _ in rows), default=0)
    lines = [title] if title else []
    lines += [f"  {k.ljust(width)}  {_format_value(v)}" for k, v in rows]
    return "\n".join(lines) + "\n"


def _emit_json(data: Any, target: str) -> None:
    """Write ``data`` as JSON to stdout (``target == "-"``) or to the file ``target``."""
    text = json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
    if target == "-":
        sys.stdout.write(text)
        return
    out = _path(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info("wrote %s", out)


def _lazy(module: str, function: str) -> Callable[..., Any]:
    """Import ``module.function`` of a command implemented in another module.

    Raises
    ------
    UsageError
        If the module (or the function) is not available.
    """
    message = f"this command requires {module} (not installed / not available)"
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        parts = module.split(".")
        prefixes = {".".join(parts[: i + 1]) for i in range(len(parts))}
        if exc.name in prefixes:
            raise UsageError(message) from exc
        raise
    func = getattr(mod, function, None)
    if not callable(func):
        raise UsageError(f"{message}: {module}.{function} is missing")
    return func


# --------------------------------------------------------------------------- commands


def _cmd_prepare(args: argparse.Namespace) -> int:
    from lrmlp.config import load_config
    from lrmlp.dataset import build_dataset, load_dataset

    config = load_config(args.config, args.overrides)
    output = args.output if args.output is not None else config.data.output_dir
    logger.info("preparing dataset %r from %d source(s)", config.name, len(config.data.sources))
    path = build_dataset(config, output, args.workers, overwrite=args.overwrite)
    dataset = load_dataset(path)
    sizes = ", ".join(f"{name} {len(split)}" for name, split in dataset.splits.items())
    logger.info("dataset written to %s (%s molecules)", path, sizes)
    sys.stdout.write(f"{path}\n")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from lrmlp.config import load_config
    from lrmlp.training import train

    config = load_config(args.config, args.overrides)
    result = train(config, args.dataset, args.run_dir)
    logger.info(
        "training finished: best epoch %d, val energy MSE %.6g (kcal/mol)^2, %d steps",
        result.best_epoch,
        result.best_val_energy_mse,
        result.steps,
    )
    logger.info("run directory: %s", result.run_dir)
    sys.stdout.write(f"{result.run_dir}\n")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from lrmlp.evaluation import evaluate

    result = evaluate(
        args.run,
        args.split,
        args.epoch,
        args.dataset,
        batch_size=args.batch_size,
        allow_dataset_mismatch=args.allow_dataset_mismatch,
    )
    if args.json_out is not None:
        _emit_json(result.to_dict(), args.json_out)
        if args.json_out == "-":
            return 0
    rows = [
        ("split", result.split),
        ("epoch", result.epoch),
        ("molecules", result.n_molecules),
        ("energy MAE [kcal/mol]", result.energy_mae),
        ("energy RMSE [kcal/mol]", result.energy_rmse),
        ("energy MSE [(kcal/mol)^2]", result.energy_mse),
    ]
    if result.force_mae is not None:
        rows += [
            ("force MAE [kcal/(mol A)]", result.force_mae),
            ("force RMSE [kcal/(mol A)]", result.force_rmse),
        ]
    if result.dataset_mismatch:
        rows.append(("dataset matches run", "NO (see warning)"))
    sys.stdout.write(_table(rows, f"evaluation of {args.run}"))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import numpy as np

    from lrmlp.potential import Potential

    if not args.input.is_file():
        raise FileNotFoundError(f"input file {args.input} not found")
    with np.load(args.input, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [k for k in PREDICT_INPUT_KEYS if k not in keys]
        if missing:
            raise ValueError(
                f"{args.input}: missing array(s) {missing}; expected R (S, N, 3) and z (N,), "
                f"found {sorted(keys)}"
            )
        extra = sorted(keys - set(PREDICT_INPUT_KEYS))
        if extra:
            logger.warning("%s: ignoring array(s) %s", args.input, extra)
        positions = np.asarray(data["R"])
        numbers = np.asarray(data["z"])
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"{args.input}: R must have shape (S, N, 3), got {positions.shape}")
    if numbers.ndim != 1 or numbers.shape[0] != positions.shape[1]:
        raise ValueError(
            f"{args.input}: z must have shape (N,) = ({positions.shape[1]},), got {numbers.shape}"
        )

    potential = Potential.from_run(args.run, args.epoch)
    logger.info(
        "predicting %d frame(s) of %d atoms with epoch %s of %s",
        positions.shape[0],
        positions.shape[1],
        potential.epoch,
        args.run,
    )
    prediction = potential.predict(
        positions, numbers, batch_size=args.batch_size, n_workers=args.workers
    )
    arrays: dict[str, Any] = {"E": np.asarray(prediction.energies, dtype=np.float64)}
    if prediction.forces is not None:
        arrays["F"] = np.asarray(prediction.forces, dtype=np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        np.savez(fh, **arrays)
    logger.info("wrote %s (%s)", args.output, ", ".join(sorted(arrays)))
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    run_search = _lazy("lrmlp.tuning", "run_search")
    from lrmlp.run import tensorboard_available
    from lrmlp.tuning import TENSORBOARD_REQUIRED

    if not args.dry_run and not tensorboard_available():
        raise UsageError(TENSORBOARD_REQUIRED)
    result = run_search(args.config, args.space, args.max_trials, args.dry_run)
    if result is not None:
        logger.info("search finished: %s", result)
    return 0


def _cmd_plot_log(args: argparse.Namespace) -> int:
    plot_training_log = _lazy("lrmlp.analysis.plots", "plot_training_log")
    source = args.run if args.run is not None else args.log
    written = plot_training_log(source, args.output, args.skip)
    logger.info("wrote %s", ", ".join(str(p) for p in written))
    return 0


def _cmd_plot_weights(args: argparse.Namespace) -> int:
    plot_weight_distribution = _lazy("lrmlp.analysis.weights", "plot_weight_distribution")
    result = plot_weight_distribution(
        args.run_a, args.run_b, args.output, args.epoch_a, args.epoch_b
    )
    logger.info("wrote %s", result if result is not None else args.output)
    return 0


def _dataset_info(path: Path) -> tuple[dict[str, Any], str]:
    from lrmlp.dataset import load_dataset

    dataset = load_dataset(path)
    meta = dataset.metadata
    grid = meta.get("grid", {})
    splits = {
        name: {"n_molecules": len(split), "n_atoms": split.n_atoms}
        for name, split in dataset.splits.items()
    }
    info = {
        "path": str(dataset.path),
        "name": meta.get("name"),
        "format_version": meta.get("format_version"),
        "lrmlp_version": meta.get("lrmlp_version"),
        "created": meta.get("created"),
        "sources": meta.get("sources", []),
        "grid": grid,
        "species": meta.get("species"),
        "n_max_local": dataset.n_max_local,
        "n_max_long_range": dataset.n_max_long_range,
        "n_samples": meta.get("n_samples"),
        "splits": splits,
        "split_seed": meta.get("split_seed"),
        "units": meta.get("units"),
    }
    rows: list[tuple[str, Any]] = [
        ("path", info["path"]),
        ("name", info["name"]),
        ("created", info["created"]),
        ("format / lrmlp version", f"{info['format_version']} / {info['lrmlp_version']}"),
    ]
    for i, src in enumerate(info["sources"]):
        rows.append(
            (
                f"source {i}",
                f"{src.get('path')} ({src.get('n_frames_used')} of "
                f"{src.get('n_frames_available')} frames, {src.get('n_atoms')} atoms)",
            )
        )
    rows += [
        ("species", info["species"]),
        (
            "grid",
            f"size {grid.get('grid_size')}, max_local_level {grid.get('max_local_level')}, "
            f"assignment {grid.get('assignment')}",
        ),
        (
            "cutoff / inner cutoff",
            f"{_format_value(grid.get('cutoff'))} / {_format_value(grid.get('inner_cutoff'))}",
        ),
        ("n_max local / long range", f"{info['n_max_local']} / {info['n_max_long_range']}"),
        ("split seed", info["split_seed"]),
    ]
    for name, size in splits.items():
        rows.append((f"split {name}", f"{size['n_molecules']} molecules, {size['n_atoms']} atoms"))
    return info, _table(rows, "dataset")


def _run_info(path: Path) -> tuple[dict[str, Any], str]:
    from lrmlp.run import RunDirectory, read_json

    run = RunDirectory(path)
    if not run.path.is_dir():
        raise FileNotFoundError(f"run directory {run.path} not found")
    config = run.read_config()
    dataset_meta = run.read_dataset_metadata()
    best = run.read_best() if run.best_path.is_file() else None
    checkpoints = run.list_checkpoints()
    evaluations = {}
    for p in sorted(run.path.glob("evaluation_*.json")):
        data = read_json(p)
        if isinstance(data, Mapping):
            evaluations[p.name] = {
                k: data.get(k) for k in ("split", "epoch", "energy_mae", "n_molecules")
            }
    m, t, g = config.model, config.train, config.grid
    model = {
        "use_long_range": m.use_long_range,
        "predict_forces": m.predict_forces,
        "embedding_dims": list(m.embedding_dims),
        "fitting_width": m.fitting_width,
        "fitting_res_blocks": list(m.fitting_res_blocks),
        "k1_local": m.k1_local,
        "k1_long": m.k1_long,
        "k2_local": m.k2_local,
        "k2_long": m.k2_long,
        "dropout_rate": m.dropout_rate,
    }
    train = {
        "epochs": t.epochs,
        "learning_rate": t.learning_rate,
        "lr_decay_steps": t.lr_decay_steps,
        "seed": t.seed,
        "checkpoints": t.checkpoints,
    }
    grid = {
        "grid_size": g.grid_size,
        "max_local_level": g.max_local_level,
        "assignment": g.assignment,
        "cutoff": g.cutoff,
        "inner_cutoff": g.inner_cutoff,
    }
    info = {
        "path": str(run.path),
        "name": config.name,
        "grid": grid,
        "model": model,
        "train": train,
        "dataset": dataset_meta.get("path"),
        "n_max_local": dataset_meta.get("n_max_local"),
        "n_max_long_range": dataset_meta.get("n_max_long_range"),
        "best": best,
        "checkpoints": checkpoints,
        "evaluations": evaluations,
    }
    rows: list[tuple[str, Any]] = [
        ("path", info["path"]),
        ("name", info["name"]),
        ("dataset", info["dataset"]),
        ("n_max local / long range", f"{info['n_max_local']} / {info['n_max_long_range']}"),
        ("grid", ", ".join(f"{k} {_format_value(v)}" for k, v in grid.items())),
        ("model", ", ".join(f"{k} {_format_value(v)}" for k, v in model.items())),
        ("train", ", ".join(f"{k} {_format_value(v)}" for k, v in train.items())),
        (
            "best epoch",
            "-"
            if best is None
            else f"{best.get('epoch')} "
            f"(val energy MSE {_format_value(best.get('val_energy_mse'))})",
        ),
        ("checkpoints", checkpoints),
    ]
    for name, ev in evaluations.items():
        rows.append(
            (
                name,
                f"{ev['split']} epoch {ev['epoch']}: energy MAE "
                f"{_format_value(ev['energy_mae'])} kcal/mol ({ev['n_molecules']} molecules)",
            )
        )
    return info, _table(rows, "run")


def _cmd_info(args: argparse.Namespace) -> int:
    if args.dataset is not None:
        info, table = _dataset_info(args.dataset)
    else:
        info, table = _run_info(args.run)
    if args.json_out is None:
        sys.stdout.write(table)
    else:
        _emit_json(info, args.json_out)
    return 0


# --------------------------------------------------------------------------- main


@contextlib.contextmanager
def _logging_to_stderr(level: str) -> Iterator[None]:
    """Send the records of the ``lrmlp`` loggers to stderr with time and level.

    The handler and the previous level are removed again on exit, so calling
    :func:`main` in-process (tests, notebooks) leaves no handler behind.
    """
    pkg = logging.getLogger("lrmlp")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    previous = pkg.level
    pkg.addHandler(handler)
    pkg.setLevel(level)
    try:
        yield
    finally:
        pkg.removeHandler(handler)
        pkg.setLevel(previous)


def _exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    sys.stderr.write(f"{code}\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the ``lrmlp`` command line interface.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: ``0`` on success, ``1`` on a runtime error, ``2`` on a usage
        error (including a missing module of a lazily loaded command).
    """
    parser = build_parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return _exit_code(exc.code)
    with _logging_to_stderr(args.log_level):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    """Run the handler of the parsed command and map exceptions to exit codes."""
    debug = args.log_level == "DEBUG"
    try:
        code: int = args.handler(args)
    except UsageError as exc:
        sys.stderr.write(f"lrmlp {args.command}: error: {exc}\n")
        return 2
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130
    except Exception as exc:
        if debug:
            logger.exception("lrmlp %s failed", args.command)
        else:
            logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    return code
