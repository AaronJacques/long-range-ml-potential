# Benchmarks: legacy code vs. lrmlp

`run_benchmarks.py` measures preprocessing, the input pipeline, training and
inference of the unchanged legacy code (`Neural-Network/` at commit `9413bfd`) and of
`lrmlp`. Both run on the same TensorFlow runtime of `.venv`. `compare.py` combines the
two result files, checks the numerical equivalence and writes
[`docs/benchmarks.md`](../docs/benchmarks.md).

The legacy code is never imported from the working tree. `tests/golden/legacy.py`
exports it with `git archive` into a temporary directory. It needs Keras 2, so it runs
in its own process with `TF_USE_LEGACY_KERAS=1` (extra `legacy`: pandas, tf-keras).
`lrmlp` runs in a separate process with Keras 3.

## Running

```sh
source .venv/bin/activate

# quick smoke run (reduced workloads, ~1 min per implementation)
python benchmarks/run_benchmarks.py --impl new --suite all --quick --out /tmp/bench-new.json
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 \
    python benchmarks/run_benchmarks.py --impl legacy --suite all --quick --out /tmp/bench-legacy.json
python benchmarks/compare.py /tmp/bench-legacy.json /tmp/bench-new.json --check-equivalence

# reference run (commit the JSON files and the generated page)
python benchmarks/run_benchmarks.py --impl new --suite all \
    --out benchmarks/results/reference-$(date +%Y%m%d)-macos-arm64-new.json
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 python benchmarks/run_benchmarks.py \
    --impl legacy --suite all \
    --out benchmarks/results/reference-$(date +%Y%m%d)-macos-arm64-legacy.json
python benchmarks/compare.py benchmarks/results/reference-*-legacy.json \
    benchmarks/results/reference-*-new.json --check-equivalence --markdown docs/benchmarks.md
```

Run the reference measurement on an otherwise idle machine. Each result file stores
the load average at its start and end. `--note TEXT` adds a remark to the metadata
(for example `--note "preliminary (machine under load)"`), and `compare.py` shows it at
the top of the page. If the glob matches results of several days, `compare.py` uses
the most recent file of each implementation. Delete superseded reference files.

### Options of `run_benchmarks.py`

| option | meaning |
|---|---|
| `--impl {legacy,new}` | implementation; `legacy` requires `TF_USE_LEGACY_KERAS=1`, `new` requires it unset |
| `--suite {prep,input,train,infer,all}` | what to measure |
| `--quick` | 200 aspirin-like / 20 stachyose-like frames, 200 inference molecules |
| `--workers N` | processes for the parallel preprocessing and inference (default: all CPUs) |
| `--repeats N` | repetitions per measurement, median reported (default and minimum 3) |
| `--out PATH` | result JSON |
| `--note TEXT` | free text in the metadata |
| `--check --check-dir DIR` | equivalence mode (started by `compare.py --check-equivalence`) |
| `--keep` | keep the temporary work directory |

### Options of `compare.py`

`compare.py RESULT.json ... [--check-equivalence] [--markdown PATH]
[--equivalence-json PATH] [--save-equivalence PATH]`. It finds the legacy and the new
file by their `impl` field. `--equivalence-json` reuses a saved report instead of
running the check again.

Exit codes: 0 = done (missed performance targets are reported, but they do not change
the exit code), 1 = an equivalence tolerance was violated, 2 = invalid input.

## Duration (Apple M4, 10 CPUs)

| run | duration |
|---|---|
| `--quick`, new / legacy | ~50 s / ~20 s |
| `--quick` equivalence check | ~25 s |
| full, new / legacy | ~2 min / ~2.5 min (quiet machine, C4 reference run; `metadata.runtime_s`) |
| full equivalence check | ~1 min |

## What is measured

| suite | workload | legacy | new |
|---|---|---|---|
| `prep` | 2000× aspirin-like (21 atoms, Δg 1, l_max 6), 200× stachyose-like (87 atoms, Δg 2, l_max 4) | `create_dataset` (pandas, `.pkl.gzip`) | `build_dataset` with 1 and `--workers` processes (`.npy`) |
| `input` | training split of the aspirin workload (1600 molecules) | `create_tf_dataset` (`from_generator` over `iterrows`) | `make_tf_dataset` |
| `train` | 1600 train / 200 val molecules, energy only, \|B\| = 1 | unchanged `start_training`, one warm-up epoch + 3 × 2 timed epochs | `Trainer.train_epoch`, same schedule, for `steps_per_execution` ∈ {1, 8} × `jit_compile` ∈ {false, true} |
| `infer` | first 1000 aspirin-like frames | `evaluate.py` loop, one `tf.function` call per molecule | `Potential.predict_descriptors` / `Potential.predict`, `batch_size = 256`, with and without descriptor computation |

Both implementations use the grid assignment `legacy` and the network of
`configs/legacy-head.toml` (the state of `Constants.py@9413bfd`). The workloads come
from `tests/golden/synthetic.py` and are deterministic. A SHA-256 over the arrays is
stored in every result, and `compare.py` refuses result files whose workloads
differ.

Timing uses `time.perf_counter`. Every value is the median of at least 3 repetitions;
all repetitions are stored in `times_s`. For training and inference the first call
(tracing) is excluded: there is a warm-up epoch or a warm-up call. The legacy loops
are timed through a wrapper around the dataset that `create_tf_dataset` returns, so
the legacy code itself stays unchanged.

## Equivalence check

`compare.py --check-equivalence` runs `run_benchmarks.py --check` twice. The legacy
process builds the dataset of each workload with `create_dataset`, records the legacy
atom order (`create_grid`), builds the legacy model with a fixed seed and exports its
weights in construction order together with its energies (`reduce_sum` over the atoms,
one molecule per call). The `lrmlp` process then compares:

| check | tolerance |
|---|---|
| descriptors (`compute_descriptors` + `pad_descriptors`, assignment `legacy`) vs. the legacy pickles, after reordering the legacy rows into input order | atol = rtol = 1e-6, integer columns exact, equal N_max |
| energies with the same weights (`assign_legacy_weights`, `Potential.predict`) | rtol 1e-5, atol 1e-5 per atom |
| `jit_compile = true` vs. `false`, 100 training steps from the same weights on the same molecules | weights rtol 1e-4 / atol 1e-6, mean loss rtol 1e-4 |
| `steps_per_execution = 8` vs. 1, one epoch | weights rtol 1e-4 / atol 1e-6, epoch loss rtol 1e-4 |

## Interpretation

* Factors are new ÷ legacy for rates and legacy ÷ new for bytes, so a factor > 1
  always means the new code is better. The targets come from plan section 4.6.
  Missed targets stay in the table and are explained on the generated page, in the
  manual analysis block. `compare.py` keeps that block when it regenerates the page.
* Measurements on a loaded machine are noisy, especially the parallel ones. Compare
  the load averages in the metadata before drawing conclusions.
* `jit_compile` and `steps_per_execution` are recommended only with a gain of at
  least 15 % **and** a passed equivalence check. The benchmark never changes the
  defaults in `src/`.
* Training with |B| > 1 and mixed precision are out of scope (see the page).
