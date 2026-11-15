# Documentation

Documentation of `lrmlp`, the code of the master's thesis *Machine Learning
Potentials with Long Range Interaction* (Aaron Schulz, TUM, 2024). Start with the
[README](../README.md) for the idea, the installation and a quickstart.

## Using the package

| Page | Content |
|---|---|
| [README](../README.md) | Idea of the thesis, installation, quickstart on synthetic data, results reported in the thesis, project layout, citation, license |
| [reproducing.md](reproducing.md) | Step by step through the thesis experiments: sGDML data, `prepare` / `train` / `evaluate` per molecule, expected runtime, presets versus `legacy-head`, learning curves and weight plots |
| [configuration.md](configuration.md) | Reference of every configuration field (type, default, meaning, thesis symbol, legacy name), `--set` overrides, path resolution, presets compared |
| [data-format.md](data-format.md) | Raw trajectories, preprocessed dataset (`metadata.json`, arrays), run directory, training log, evaluation files, prediction input and output |

## Method and code

| Page | Content |
|---|---|
| [methodology.md](methodology.md) | The method with chapter, equation and page references; neighbour assignment modes; deviations of the original code (A01-A17) and decisions; open questions (F1-F7); fixed defects of the legacy code (B01-B19); invariances; forces; known limitations; symbols and configuration fields |
| [architecture.md](architecture.md) | Modules and their dependencies, data flow, public API, design decisions, extension rules |
| [benchmarks.md](benchmarks.md) | Speed of `lrmlp` against the original code (preprocessing, input pipeline, training, inference), equivalence checks, execution defaults, missed targets |

## Further material

| Item | Content |
|---|---|
| [The thesis (PDF)](thesis/Masterarbeit__Machine_Learning_Potentials_with_Long_Range_Interaction.pdf) | Master's thesis as submitted on 05.10.2024. Page references in the documentation are printed page numbers (PDF page = printed page + 8) |
| [assets/legacy-runs/README.md](assets/legacy-runs/README.md) | Origin of the plots of early runs of the original code (not the thesis figures) |
| [benchmarks/README.md](../benchmarks/README.md) | Running the benchmarks and the equivalence check |
| [tests/fixtures/README.md](../tests/fixtures/README.md) | Synthetic test data and golden fixtures from the original code |
| [notebooks/training_curves.ipynb](../notebooks/training_curves.ipynb) | Figures of Ch. 5 (learning curves, weight distributions) from trained runs |
| [CITATION.cff](../CITATION.cff) | Citation metadata |
| [configs/](../configs/) | Presets per molecule, `legacy-head`, hyperparameter search spaces |
