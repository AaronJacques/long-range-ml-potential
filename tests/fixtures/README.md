# Test fixtures

Deterministic synthetic inputs and golden fixtures frozen from the **unchanged
legacy code** (`Neural-Network/` at commit `9413bfd`). Later refactoring tasks
prove their equivalence against these files. Do not edit them by hand.

## Inputs (`tests/golden/synthetic.py`, NumPy only)

| File | Content |
|---|---|
| `synthetic_small.npz` | 60 frames, aspirin-like: z = C9H8O4 in fixed order `[6]*9 + [1]*8 + [8]*4` (21 atoms), box ~ 9 Å, `aspirin_like(60, seed=0)` |
| `stachyose_like.npz` | 5 frames, 87 atoms (C24H42O21), box ~ 16 Å, `stachyose_like(5, seed=1)` |
| `edge_cases.npz` | hand-built single molecules, keys `<case>__R` (N, 3) and `<case>__z` (N,) |

MD17 layout: `E` (S,) kcal/mol, `F` (S, N, 3) kcal/(mol·Å), `R` (S, N, 3) Å,
`z` (N,). Frames are a random reference geometry plus Gaussian displacements
(σ = 0.1 Å), positions rounded to 2⁻¹⁰ Å, all pair distances ≥ 1.0 Å, so all
positions are unique. Energy and forces are Lennard-Jones (ε = 0.1, σ = 1.2)
with analytic forces `F = -∇E`.

Edge cases (descriptions in `synthetic.edge_cases()`): `level_lmax2_within_rc`
(A01), `dropped_pairs` (A02), `partial_cell` (cell on level l_max+1 with atoms on
both sides of r_c), `single_cell`, `no_long_range`, `cell_boundary` (atoms exactly
on cell boundaries), `gb_assignment` (A01/A02 for G-B), `multi_element`.

## Golden configurations

Binding for all tasks. Always `MAX_ATOM_ELEMENTS = 100`, `use_charge_number = True`,
`CUT_OFF = (l_max + 1 + 0.5**0.5) * Δg`, `INNER_CUT_OFF = CUT_OFF - 0.2`.

| Name | Δg [Å] | l_max | r_c [Å] |
|---|---|---|---|
| G-A | 1.0 | 6 | 7.7071 |
| G-B | 1.0 | 2 | 3.7071 |
| G-C | 2.0 | 4 | 11.4142 |

## Golden files (`golden/`)

| File | Source |
|---|---|
| `data_descriptors_GA_aspirin.npz` | G-A, first 10 frames of `synthetic_small` |
| `data_descriptors_GB_aspirin.npz` | G-B, same 10 frames |
| `data_descriptors_GB_aspirin_sorted.npz` | G-B with `sort_matrix=True` |
| `data_descriptors_GA_edge.npz`, `data_descriptors_GB_edge.npz` | all edge cases, one frame each (`frame_names`) |
| `data_descriptors_GC_stachyose.npz` | G-C, first 3 frames of `stachyose_like` |
| `data_dataset_GA.npz`, `data_dataset_GB.npz` | `create_dataset([synthetic_small.npz], …)` → the three pickles (split 80/10/10, `random_state=42`) |
| `manifest_data.json` | legacy commit, versions, constants per config, per-case stats, sha256 of every file |

Descriptor goldens hold the raw output of `create_grid` + `create_matrices` per
frame; dataset goldens hold the padded float32 arrays, labels and split indices.
The key layout is documented in `tests/golden/fixtures.py`, which also provides
the loaders (`load_descriptor_golden`, `load_dataset_golden`, `split_ragged`,
`to_input_order`).

Legacy behaviour these files freeze (plan.md Appendix A):

- Rows are in **cell traversal order** (cells lexicographic by grid index, atoms
  in input order within a cell), not input order. `order[k]` is the input index
  of legacy row `k`. Forces and positions stay in input order (bug B01).
- A02: atoms beyond r_c in cells on level ≤ l_max are dropped.
- A01: atoms inside r_c in cells on level ≥ l_max+2 enter only via the cell
  centroid.

`.npz` files are written by `fixtures.save_npz` instead of `np.savez_compressed`.
The container is the same (zip of `.npy`, deflate, readable with `np.load`), but
`np.savez_compressed` stamps every zip entry with the current time, so identical
arrays would get new bytes and a new sha256 on every regeneration. `save_npz` uses
a fixed timestamp (1980-01-01), which keeps the bytes and the manifest sha256
reproducible.

## Regeneration

Needs the golden env (legacy code, TF 2.21 + tf_keras). The legacy code is taken
from `git archive 9413bfd` into a temp dir and never imported from the working
tree.

```sh
G=.overhaul/env/venv-py312-tf221/bin/python
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 $G tests/golden/generate_data_golden.py          # write
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 $G tests/golden/generate_data_golden.py --check  # OK: identical
```

`--check` regenerates into a temp dir and compares all arrays bit for bit, plus
every manifest field except the environment `versions` and the file sha256
(differences in those are only reported as notes). The manifest has no
timestamp. `tests/test_golden_fixtures.py` checks the committed files without
the legacy env.
