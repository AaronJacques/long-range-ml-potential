# Methodology

This page describes the method of the master's thesis *Machine Learning
Potentials with Long Range Interaction* (TUM, 2024), how `lrmlp` implements it,
where the legacy code (commit `9413bfd`) deviated from it, and what was decided.
The refactoring changes the code that implements the method, not the method.

The thesis is in the repository:
[`docs/thesis/Masterarbeit__Machine_Learning_Potentials_with_Long_Range_Interaction.pdf`](thesis/Masterarbeit__Machine_Learning_Potentials_with_Long_Range_Interaction.pdf).
References like "p. 43" are **printed** page numbers. The PDF page number is the
printed page number + 8 (p. 43 is PDF page 51). Equation, figure and table
numbers are those of the thesis. Reference numbers in brackets ([14], [16]) are
those of the thesis bibliography (p. 64-65). File formats are described in
[data-format.md](data-format.md).

Where the thesis does not say something, this page says "not specified in the
thesis". Legacy code locations (`Dataset.py:310-356` etc.) refer to
`Neural-Network/` at commit `9413bfd`. The old code is available with
`git archive 9413bfd`.

## Overview

The thesis extends the Deep Potential approach ([14], DeepPot-SE) with a
long-range part inspired by the Fast Multipole Method (Ch. 3, Ch. 4 intro,
p. 36):

1. **Energy decomposition.** The energy of a molecule is a sum of atomic
   energies, $E = \sum_{i=1}^{N} E_i$ (Eq. 2.7, p. 21). One network predicts every
   $E_i$ from a description of the surroundings of atom i (p. 45).
2. **Local part (Sec. 4.1, p. 37-38).** All atoms j within the cutoff radius
   $r_c$ of atom i form the local environment. They enter through the generalised
   coordinates $\tilde R_i$ with the smooth weight $s(r)$ (Eq. 4.4-4.6).
3. **Long-range part (Sec. 4.2, p. 39, and Sec. 4.3.2, p. 40-44).** A fixed
   axis-aligned grid with cell size $\Delta g$ is laid over the molecule. All
   atoms outside $r_c$ are represented by the Z-weighted centres of mass of
   their grid cells (Eq. 4.12) and by per-cell element counts. No recursive
   refinement is used (p. 39).
4. **Descriptor.** For both parts an embedding network produces a matrix
   $G_i$, and the feature matrix is $D_i = G_{i1}^\top \tilde R_i \tilde R_i^\top G_{i2}$
   (Eq. 4.8, p. 38 and p. 45).
5. **Energy network.** The flattened local and long-range $D_i$ are
   concatenated and mapped to $E_i$ (Fig. 4.4, p. 48).
6. **Training.** Only energies are used (p. 40). The loss is the squared error
   of the total energy, one molecule per step, $\lvert B\rvert = 1$ (Eq. 4.21,
   p. 51); Adam with an exponentially decaying learning rate (Eq. 4.22, p. 51).
7. **Model selection and evaluation.** The epoch with the lowest validation MSE
   is evaluated on the test split; the metric is the MAE of the molecular energy
   (p. 52, Tab. 5.3, p. 53).

Data flow in `lrmlp`: raw `.npz` (`lrmlp.io`) → grid and partition
(`lrmlp.grid`) → descriptor matrices (`lrmlp.descriptors`) → stored dataset
(`lrmlp.dataset`, `lrmlp prepare`) → network (`lrmlp.model`) → training
(`lrmlp.training`, `lrmlp.schedules`, `lrmlp train`) → evaluation and inference
(`lrmlp.evaluation`, `lrmlp.potential`, `lrmlp evaluate`, `lrmlp predict`).

## Thesis map

| Thesis | Pages (printed / PDF) | Content | Implemented in |
|---|---|---|---|
| Sec. 2.1.1 | 20-21 / 28-29 | Deep Potentials: $F_i$ as negative energy gradient (Eq. 2.1), loss with $p_e, p_f, p_\nu$ (Eq. 2.3-2.6), $E = \sum_i E_i$ (Eq. 2.7), local environment $N_{r_c}(i)$ | [`training.py`](../src/lrmlp/training.py), [`potential.py`](../src/lrmlp/potential.py) |
| Sec. 2.2 | 26 / 34 | Symmetries: translation, rotation, permutation invariance; energy conservation $F = -\nabla U$ | [Invariances](#invariances), [Forces](#forces) |
| Sec. 4.1 | 37-38 / 45-46 | Baseline model: $\Omega_i = R_i R_i^\top$ (Eq. 4.1), $\tilde R_i$ (Eq. 4.3-4.6), $s(r)$ (Eq. 4.5), $G_i$ (Eq. 4.7, 4.9), $D_i$ (Eq. 4.8), baseline loss (Eq. 4.10) | [`descriptors.py`](../src/lrmlp/descriptors.py), [`model.py`](../src/lrmlp/model.py) |
| Sec. 4.2 | 39 / 47 | Long range via cell centres of mass, fixed grid size | [`grid.py`](../src/lrmlp/grid.py) |
| Sec. 4.3 | 40 / 48 | Implementation, link to the code repository | – |
| Sec. 4.3.1 | 40 / 48 | Datasets [16], first 65 000 frames, 80/10/10 split, Tab. 4.1 | [`io.py`](../src/lrmlp/io.py), [`dataset.py`](../src/lrmlp/dataset.py) |
| Sec. 4.3.2 | 40-44 / 48-52 | Preprocessing: grid (Eq. 4.11), counts and centre of mass (Eq. 4.12), levels (Eq. 4.13), $r_c$ (Eq. 4.14), local matrices (Eq. 4.15), reduced grid, long-range matrices (Eq. 4.16-4.20), padding; Fig. 4.1 | [`grid.py`](../src/lrmlp/grid.py), [`descriptors.py`](../src/lrmlp/descriptors.py), [`dataset.py`](../src/lrmlp/dataset.py) |
| Sec. 4.3.3 | 45-50 / 53-58 | Architecture: Dense-Res-Block (Fig. 4.2), embedding (Fig. 4.3), energy network (Fig. 4.4), full and local-only network (Fig. 4.5, 4.6) | [`model.py`](../src/lrmlp/model.py) |
| Sec. 4.3.4 | 51 / 59 | Training: loss (Eq. 4.21), learning rate (Eq. 4.22), $\lvert B\rvert = 1$ | [`training.py`](../src/lrmlp/training.py), [`schedules.py`](../src/lrmlp/schedules.py) |
| Ch. 5 | 52-55 / 60-63 | Parameters (Tab. 5.1, 5.2), model selection, test MAE (Tab. 5.3), learning curves (Fig. 5.1, 5.2), weight distributions (Fig. 5.3) | [`configs/`](../configs/), [`evaluation.py`](../src/lrmlp/evaluation.py), `lrmlp.analysis` |
| Ch. 6 | 56 / 64 | Comparison with DeepPot-SE [14] and sGDML [17] (Tab. 6.1) | – (reported values only) |
| Ch. 7 | 57-58 / 65-66 | Outlook: non-smooth long-range descriptor (7.1), hierarchical grid (7.2), larger datasets (7.3), larger batch size (7.4), systematic tuning (7.5) | [Known limitations](#known-limitations) |

## Grid and neighbour levels

All quantities are computed per frame, for every atom i of the molecule
(`lrmlp.grid`, `lrmlp.descriptors`; Sec. 4.3.2, p. 40-44).

**Grid (Eq. 4.11, p. 42).** The origin is the component-wise minimum of the
positions, $g_{0,0,0} = (\min_i x_i, \min_i y_i, \min_i z_i)$. The axes are
parallel to the coordinate axes and the cell edge is $\Delta g$
(`grid.grid_size`). Atom i lies in the cell with index
$\lfloor (r_i - g_{0,0,0}) / \Delta g \rfloor$ (component-wise).

**Levels (Eq. 4.13, p. 42).** Seen from atom i in cell $(j, k, l)$, a cell
$(\tilde j, \tilde k, \tilde l)$ has level
$\max(\lvert j-\tilde j\rvert, \lvert k-\tilde k\rvert, \lvert l-\tilde l\rvert)$
(Chebyshev distance). Level 0 is the own cell, level 1 the direct neighbours.
$G_i$ is the set of cells with level $\le l_{max}$ (`grid.max_local_level`).

**Cutoff (Eq. 4.14, p. 42).** $r_c$ is not an input; it follows from $l_{max}$:

$$r_c = (l_{max} + 1)\,\Delta g + \sqrt{2 (\Delta g / 2)^2} = \left(l_{max} + 1 + \tfrac{1}{\sqrt 2}\right) \Delta g .$$

The inner cutoff of the switching function is $r_{cs} = r_c - 0.2$ Å
(Tab. 5.1, p. 52; `grid.inner_cutoff_offset`). Both are derived properties of
`GridConfig` (`cutoff`, `inner_cutoff`) and are recorded in the dataset
metadata and in `config.json` (`derived`).

| Molecule (Tab. 5.1, p. 52) | $\Delta g$ (Å) | $l_{max}$ | $r_c$ (Å) | $r_{cs}$ (Å) |
|---|---|---|---|---|
| Paracetamol | 1 | 6 | 7.71 | 7.51 |
| Aspirin | 1 | 6 | 7.71 | 7.51 |
| Ac-Ala3-NHMe | 2 | 2 | 7.41 | 7.21 |
| Stachyose | 2 | 4 | 11.41 | 11.21 |

**Cell aggregates (Eq. 4.12, p. 42).** For a cell k with atoms of atomic number
$z_i^k$ at $r_i^k$, the centre of mass is
$r_m^k = \sum_i z_i^k r_i^k / \sum_i z_i^k$ (weighted by atomic number, not by
mass). The feature vector $f^k \in \mathbb{N}^{m_{max}}$ counts the atoms of
every element in the cell (p. 42). `lrmlp` stores count index Z − 1 for
element Z and does not count elements with $Z > m_{max}$
(`grid.max_atomic_number`, default 100; the value of $m_{max}$ is not specified
in the thesis). The thesis text is not consistent here: it says that the i-th
entry counts atomic number i + 1 and that atoms "higher than $m_{max}+1$" are not
considered (p. 42), while $f^k$ has $m_{max}$ entries and Eq. 4.19 indexes them
$1 \dots m_{max}$. The implementation follows the dimension $m_{max}$ (as the
legacy code `Dataset.py:34-49`). In practice it makes no difference: the
datasets contain only H, C, N, O, and `lrmlp prepare` rejects atomic numbers
above `grid.max_atomic_number`.

**Reduced grid (p. 43, Fig. 4.1, p. 41).** For every atom i, the atoms of its
local environment are removed from a copy of the grid, then all empty cells are
removed. The remaining cells form the reduced grid of atom i. Its centres of mass
and counts are computed from the remaining atoms only (Fig. 4.1: "the feature
vector $f^k$ and the center of mass $r_m^k$ are formed using only the gray
atoms").

## Neighbour assignment modes

The thesis defines the local environment by distance alone:
$N_{r_c}(i) = \{ j : r_{ij} \le r_c \}$, "The grid is not relevant for this
step" (p. 43). All other atoms stay in the reduced grid and enter the long-range
part (p. 43, Fig. 4.1). The legacy code assigned atoms cell by cell instead
(`Dataset.py:310-356`). `grid.assignment` selects the rule for every ordered pair
(i, j), i ≠ j:

| | `"thesis"` (default) | `"legacy"` |
|---|---|---|
| $r_{ij} \le r_c$ | local | local if the cell of j has level $\le l_{max}+1$; **long range** (whole cell, via its centre of mass) if level $\ge l_{max}+2$ (A01) |
| $r_{ij} > r_c$ | long range (reduced grid) | long range if level $\ge l_{max}+1$; **dropped** (neither local nor long range) if level $\le l_{max}$ (A02) |
| Thesis reference | p. 43, Fig. 4.1 | – (`Dataset.py:310-356`) |
| Used by | all molecule presets in `configs/` | `configs/legacy-head.toml` |

**Geometry note.** The thesis states that $r_c$ "is chosen such that all cells
included in the local interaction $G_i$ lie within the cutoff radius" (p. 42),
and the legacy code assumes the same (`Constants.py:33`: "all atoms in grid
cells with level <= MAX_LOCAL_LEVEL are included"). In three dimensions this
does not hold. Atom i and an atom in a cell of level $l_{max}$ can be up to
$l_{max}+1$ cell edges apart along each axis, so their distance can approach

$$\sqrt 3\,(l_{max} + 1)\,\Delta g \approx 12.1 \text{ Å} \quad (\Delta g = 1 \text{ Å}, l_{max} = 6), \qquad r_c = 7.71 \text{ Å}.$$

The legacy code silently drops such atoms (A02). The thesis itself does
not depend on the assumption: on p. 42-43 it adds that atoms within $r_c$ in
cells outside $G_i$ "are also included in the local interaction", and on p. 43
it defines the long-range cells as "precisely the cells/atoms that remain after
the calculation of the local matrix". The `"thesis"` mode implements exactly that
and loses no atom. The `"legacy"` mode is kept so that results of the old code
can be reproduced and is covered by the golden tests.

In the `"thesis"` mode, $l_{max}$ and $\Delta g$ therefore act only through
$r_c$ (Eq. 4.14) and the cell size of the long-range part.

## Descriptors

`lrmlp.descriptors.compute_descriptors` builds four matrices per atom i
(Sec. 4.3.2, "Local Matrices", "Long Range Matrices" and "Final Model Input",
p. 43-44).

**Local generalised coordinates (Eq. 4.3-4.6, p. 37-38).** One row per local
neighbour j:

$$(\tilde R_i^{local})_j = \left( s(r_{ji}),\ \frac{s(r_{ji})\,x_{ji}}{r_{ji}},\ \frac{s(r_{ji})\,y_{ji}}{r_{ji}},\ \frac{s(r_{ji})\,z_{ji}}{r_{ji}} \right)$$

with the switching function (Eq. 4.5, p. 38)

$$s(r) = \begin{cases} \dfrac{1}{r} & r < r_{cs} \\[4pt] \dfrac{1}{r}\left( \dfrac12 \cos\!\left[\pi \dfrac{r - r_{cs}}{r_c - r_{cs}}\right] + \dfrac12 \right) & r_{cs} \le r < r_c \\[4pt] 0 & r \ge r_c \end{cases}$$

The thesis writes strict inequalities on both sides and leaves $r = r_{cs}$ and
$r = r_c$ open; the implementation (as the legacy `get_s`, `Dataset.py:243-255`)
uses the cases above, which make $s$ continuous at both radii. A neighbour at
exactly $r = r_c$ is local with $s = 0$.

**Local atomic numbers (Eq. 4.15, p. 43).** $Z_i$ has one row $(z_i, z_j)$ per
row of $\tilde R_i^{local}$, "sorted such that $(Z_i)_l$ are the atomic numbers
corresponding to the respective distance $(\tilde R_i)_l$": row l of $Z_i$
belongs to the same neighbour as row l of $\tilde R_i^{local}$. No sorting by
distance is required (A17).

**Long-range coordinates (Eq. 4.16-4.18, p. 43-44).** One row per cell k of
the reduced grid of atom i, with the distance $r_{ki}$ to the centre of mass
$r_m^k$:

$$(\tilde R_i^{long})_k = \left( \frac{1}{r_{ki}},\ \frac{x_{ki}}{r_{ki}},\ \frac{y_{ki}}{r_{ki}},\ \frac{z_{ki}}{r_{ki}} \right)$$

The direction is a unit vector **without** the factor $s$ (p. 44 defines
$s(r) = 1/r$ and $\hat x = x/r$ for the long range). If a centre of mass
coincides with the atom ($r_{ki} = 0$), the row is $(0, 0, 0, 0)$, as in the
legacy code; this case is not specified in the thesis.

**Long-range features (Eq. 4.19-4.20, p. 44).** $F_i$ has one row
$(z_i, (f^k)_1, \dots, (f^k)_{m_{max}})$ per row of $\tilde R_i^{long}$. The dataset
stores only the counts of the elements that occur; the full
$(m_{max}+1)$-column matrix is rebuilt for the model (see
[data-format.md](data-format.md#rebuilding-the-101-columns-of-f_i)).

**Sign convention (A08).** The thesis uses $r_{ji} = r_j - r_i$ (Eq. 4.3) and
$r_{ki} = r_m^k - r_i$ (p. 43). The code uses $r_i - r_j$ and $r_i - r_m^k$ (as
the legacy code, `Dataset.py:268, 332`). This flips the sign of the three
direction columns of every row, $\tilde R' = \tilde R\,\mathrm{diag}(1, -1, -1, -1)$,
so $\tilde R' \tilde R'^\top = \tilde R \tilde R^\top$; the first column, and
therefore $G$, is unchanged. $D$ (Eq. 4.8) is identical.

**Row order per atom.** The thesis does not specify the order of the rows. The
implementation uses the traversal order of the legacy code
(`Dataset.py:299-356`): local rows by (level of the cell of j, lexicographic
cell rank, input index j), long-range rows by (level, cell rank). With
`grid.sort_neighbours = true` the legacy option `sort_matrix`
(`Dataset.py:362-376`, descending by the first column) is applied on top. The
order matters for the network, see [Invariances](#invariances).

**Atom order.** The descriptor rows of the atoms are in input order, aligned
with the force labels (fixes bug B01 of the legacy code, which emitted atoms in
grid order).

**Padding (p. 44).** For the network, $\tilde R_i^{local}$ and $Z_i$ are padded
with zero rows to $N_{max}^{local}$, $\tilde R_i^{long}$ and $F_i$ to
$N_{max}^{long}$. Both are the maxima over all frames of the dataset, determined
during preprocessing (`n_max_local`, `n_max_long_range` in `metadata.json`).
Zero rows of $\tilde R$ contribute nothing to $D$ (Eq. 4.8), whatever $G$ holds
in those rows, so no explicit mask is needed.

## Network architecture

`lrmlp.model.build_model` (Sec. 4.3.3, p. 45-50). The thesis uses $M_1, M_2$ in
Sec. 4.1 (p. 38) and $K_1, K_2$ in Sec. 4.3.3 (p. 45) for the same quantities;
this page and the code use $K_1, K_2$.

**Dense-Res-Block (Fig. 4.2, p. 46).** Input of dimension D →
LayerNormalization → Dense(N) → Dense(N) → Dense(D) → Dropout(0.2) → Add (with
the block input) → LayerNormalization. An ELU follows every Dense layer
(`model.activation`, default `"elu"`). Class `DenseResBlock`.

**Embedding network (Fig. 4.3, p. 47; p. 45).** Input: the first column
$s_i = (s(r_{1i}), \dots, s(r_{N_{max}i}))$ of $\tilde R_i$, concatenated with the
row-wise flattened $Z_i$ (local) or $F_i$ (long range). Unlike [14], **one**
embedding network is used for all species combinations; species information
enters only through $Z_i$/$F_i$ (p. 45). The figure shows five "Dense-Res-Blocks"
64, 128, 128, 256, $N_{max} \cdot K_1$, followed by a reshape to
$G_i \in \mathbb{R}^{N_{max} \times K_1}$. Since a Dense-Res-Block returns its input
dimension D (Fig. 4.2), the first and the last block can only be read as
projections. The code builds Dense(64) → Res-Blocks 128, 128, 256 →
Dense($N_{max} K_1$) → Reshape, configured by
`model.embedding_dims = [64, 128, 128, 256]` (first entry = projection, the rest
= Res-Block widths). The legacy code at `9413bfd` used `[64, 128, 128, 128]`
(A04).

**Descriptor (Eq. 4.8, p. 38; p. 45).** $G_{i1} = G_i$, $G_{i2}$ = the first
$K_2$ columns of $G_i$ (Eq. 4.9), and

$$D_i = G_{i1}^\top\, \tilde R_i\, \tilde R_i^\top\, G_{i2} \in \mathbb{R}^{K_1 \times K_2}$$

for both branches (layer `FeatureMatrix`). The thesis requires $K_2 < K_1$
(p. 38, 45); the code accepts $K_2 \le K_1$.

**Energy network (Fig. 4.4, p. 48).** Flatten($D_i^{local}$) ⊕
Flatten($D_i^{long}$) → Dense(128, ELU) → Dropout(0.2) → 3 Dense-Res-Blocks
(128) → Dense(1, linear) → $E_i$. Configured by `model.fitting_width = 128`,
`model.fitting_res_blocks = [128, 128, 128]`, `model.dropout_rate = 0.2`. The
legacy code had two Res-Blocks (A05) and an additional Dense layer in one
configuration (A06).

**Full and local-only network (Fig. 4.5, p. 49; Fig. 4.6, p. 50).**
`model.use_long_range = false` removes the long-range branch (Fig. 4.6), the
comparison model of Ch. 5.

**Molecular energy.** The network is applied to every atom and the outputs are
summed, $E = \sum_i E_i$ (Eq. 2.7, p. 21; p. 45 and p. 51). $E_i$ "in itself has
no physical meaning" (p. 45). For several molecules per call the atoms are
concatenated and summed per molecule (`segment_sum`); the result equals one
molecule per call.

**K values (Tab. 5.2, p. 52).** "$N_{max}^{local} = K_1^{local}$ and
$N_{max}^{long} = K_1^{long}$ were always chosen" (p. 52). `model.k1_local` /
`model.k1_long = "auto"` sets $K_1 = N_{max}$ of the dataset. Tab. 5.2 itself
gives $K_1 = 30/6$ for Stachyose with $N_{max} = 86/28$, so the Stachyose preset
sets $K_1$ explicitly; the table wins over the sentence.

| Molecule (Tab. 5.2, p. 52) | $N_{max}^{local}$ | $N_{max}^{long}$ | $K_1^{local}$ | $K_1^{long}$ | $K_2^{local}$ | $K_2^{long}$ | $\eta_{init}$ | s | epochs |
|---|---|---|---|---|---|---|---|---|---|
| Paracetamol | 19 | 6 | 19 | 6 | 4 | 1 | 5e-5 | 2.08e5 | 200 |
| Aspirin | 20 | 4 | 20 | 4 | 4 | 1 | 1e-4 | 2.08e5 | 200 |
| Ac-Ala3-NHMe | 41 | 20 | 41 | 20 | 4 | 2 | 5e-5 | 2.08e5 | 150 |
| Stachyose | 86 | 28 | 30 | 6 | 4 | 2 | 5e-5 | 8.727e4 | 130 |

The presets in `configs/` (`paracetamol.toml`, `aspirin.toml`,
`ac-ala3-nhme.toml`, `stachyose.toml`) take these values. $N_{max}$ is not
configured; it is read from the dataset (see
[Known limitations](#known-limitations) for why it can differ from the table).

## Training

`lrmlp.training` (Sec. 4.3.4, p. 51).

**Data (Sec. 4.3.1, p. 40, Tab. 4.1).** Datasets of [16] (DFT, FHI-aims),
positions in Å, energies in kcal/mol. At most the first 65 000 frames of a
dataset are used (`data.n_samples_per_source = 65000`; Stachyose: all 27 272,
`"all"`). Random split 80/10/10 (`data.val_fraction = data.test_fraction = 0.1`).
The seed is not specified in the thesis; `data.split_seed = 42` is the value of
the legacy code (`Dataset.py:475-477`), whose pandas split is reproduced exactly
(see [data-format.md](data-format.md#split-semantics)). "The force labels were
not used further, as only the energy was used for training" (p. 40).

**Loss (Eq. 4.21, p. 51).**

$$L = \frac{1}{\lvert B\rvert} \sum_{k \in B} \left( E_k - E_k^w \right)^2, \qquad E_k^w = \sum_i E_i, \qquad \lvert B\rvert = 1 .$$

One molecule per optimizer step (A15). Unlike Eq. 2.4 of DeepPot (p. 20) there is
no $1/N$ factor; the code follows Eq. 4.21.

**Optimizer and learning rate (Eq. 4.22, p. 51).** Adam (Keras defaults; the
Adam parameters are not specified in the thesis) with

$$\eta(t) = \eta_{init} \cdot 0.97^{\,t/s}$$

where t is the optimizer step and s the decay steps (`train.learning_rate`,
`train.lr_decay_steps`, `train.lr_decay_rate`). "s was chosen to be four times
the size of the training dataset" (p. 52); with $\lvert B\rvert = 1$ that is
every four epochs (2.08e5 = 4 × 52 000, 8.727e4 ≈ 4 × 21 818). The code decays
in steps, $0.97^{\lfloor t/s\rfloor}$ (`train.lr_staircase = true`, A07);
`lr_staircase = false` gives the continuous Eq. 4.22.

**Dropout.** Active during training with rate 0.2 (Fig. 4.2, 4.4; A03).

**Not specified in the thesis.** Shuffling (`train.shuffle_buffer = 1000`,
legacy value), weight initialisation (Keras defaults), random seeds
(`train.seed = 42`, `train.deterministic`), checkpointing. Choices of the code
are listed in the configuration reference (`docs/configuration.md`).

**Hyperparameters.** The values of Tab. 5.2 "were determined through
hyperparameter tuning" (p. 52); systematic tuning was not possible
(Ch. 5, p. 53; Sec. 7.5, p. 58). `lrmlp tune` runs a grid search over
configuration values; it is a tool, not part of the method.

## Evaluation and model selection

**Model selection (p. 52).** "All models were trained for the specified epochs,
and the epoch with the best Mean Square Error (MSE) on the validation dataset was
selected for benchmarking on the test dataset." After every epoch the trainer
computes `val_energy_mse` and records the best epoch in `best.json` (A12).
`lrmlp evaluate --run R` uses `--split test --epoch best` by default.

**Metric (Tab. 5.3, p. 53).** Mean absolute error of the molecular energy on the
test split, in kcal/mol:

$$\mathrm{MAE} = \frac{1}{S} \sum_{k=1}^{S} \Big\lvert \sum_i E_i^{(k)} - E^{(k)} \Big\rvert .$$

RMSE and MSE are reported in addition. Before evaluating, the dataset is checked
against the run's `dataset.json`, so that a test split built with another grid or
seed cannot be evaluated by accident (see
[data-format.md](data-format.md#evaluation-results)).

Results reported in the thesis (MAE in kcal/mol on the test split; not
reproduced in this repository, which contains no real data):

| Molecule | With long range | Only local | Rel. improvement (%) |
|---|---|---|---|
| Paracetamol | 2.465 | 3.024 | 22.68 |
| Aspirin | 1.670 | 1.657 | −0.78 |
| Ac-Ala3-NHMe | 4.039 | 4.087 | 1.17 |
| Stachyose | 5.841 | 5.861 | 0.34 |

(Tab. 5.3, p. 53. Tab. 6.1, p. 56, compares the long-range values with
DeepPot-SE [14] (Aspirin 0.155) and sGDML [17] (0.113, 0.047, 0.391, 4.002).)
The thesis notes that the result is "particularly sensitive to the choice of
parameters $K_1$ and $K_2$" (p. 53).

These numbers were obtained with the legacy code, i.e. with the `"legacy"`
assignment (A01, A02), inactive dropout (A03) and the code state at submission
(see F3).

## Deviations and decisions

Every difference found between the legacy code at `9413bfd` and the thesis, with
the decision taken in `lrmlp`. "Aligned" means that the new default follows the
thesis; where a switch exists, the profile `configs/legacy-head.toml` sets it to
the legacy behaviour, and the golden tests check that path against the legacy
code.

| ID | Deviation | Legacy code (`9413bfd`) | Thesis | Decision | Switch |
|---|---|---|---|---|---|
| A01 | Atoms **within** $r_c$ in cells of level $\ge l_{max}+2$ enter only through the cell's centre of mass, not locally | `Dataset.py:313-321` | p. 43: local = all j with $r_{ij} \le r_c$, "the grid is not relevant" | Bug, aligned to the thesis | `grid.assignment = "thesis"` (default) / `"legacy"` |
| A02 | Atoms **beyond** $r_c$ in cells of level $\le l_{max}$ are dropped (neither local nor long range) | `Dataset.py:336-340, 349` | p. 43, Fig. 4.1: every non-local atom stays in the reduced grid | Bug, aligned. Cause: the geometric assumption of p. 42 / `Constants.py:33` (see [geometry note](#neighbour-assignment-modes)) | `grid.assignment` |
| A03 | Dropout never active during training (model called without `training=True`) | `Training.py:69-71, 103-105` | Fig. 4.2, 4.4: dropout 0.2 | Bug, aligned: called with `training=True`. `dropout_rate = 0.0` is exactly the legacy behaviour | `model.dropout_rate` (presets 0.2, `legacy-head` 0.0) |
| A04 | Embedding widths `[64, 128, 128, 128]` | `Constants.py:61` | Fig. 4.3: 64, 128, 128, 256 | Open question F3; presets follow the thesis, `legacy-head` the legacy code | `model.embedding_dims` |
| A05 | Energy network with 2 instead of 3 Res-Blocks | `Model.py:54-59` | Fig. 4.4: 3 × Res-Block 128 | F3, as A04 | `model.fitting_res_blocks` |
| A06 | Extra `Dense(M2_local)` on the concatenated descriptors if local and long-range K1/K2 are equal | `Model.py:151-156` | Fig. 4.4: flatten and concatenate only | Aligned: branch removed. Unreachable with every parameter set of Tab. 5.2 ($K_2^{local} \ne K_2^{long}$), so no change of results | – |
| A07 | Learning rate decays in steps (`staircase=True`) | `Training.py:178` | Eq. 4.22: continuous $0.97^{t/s}$ | Deliberate deviation, kept and documented | `train.lr_staircase = true` (`false` = Eq. 4.22) |
| A08 | Relative vectors $r_i - r_j$, $r_i - r_m^k$ | `Dataset.py:268, 332` | Eq. 4.3: $r_{ji} = r_j - r_i$; p. 43: $r_{ki} = r_m^k - r_i$ | Kept: the sign flip of the direction columns leaves $\tilde R \tilde R^\top$ and $D$ unchanged (see [Descriptors](#descriptors)) | – |
| A09 | Force mode: forces as a direct 3-component network output, not $-\nabla E$ | `Model.py:103, 164`, `Training.py:59-96` | Forces not used (p. 40, 51); Eq. 4.10 and 2.1 only for the baseline | Kept as **experimental**, default off (F4); force/atom alignment bug B01 fixed; no $-\nabla E$ (see [Forces](#forces)) | `model.predict_forces = false` |
| A10 | Invariance: $D$ is invariant only for a fixed neighbour order; the embedding sees the flattened, ordered list, and order and cells depend on the axis-aligned grid | `Model.py:31-51`, `Dataset.py:299-311` | p. 38 claims invariance (for DeepPot-SE); p. 45 / Fig. 4.3 describe exactly this single embedding over the flattened list | Property of the method: not changed, documented; tests cover only the guaranteed invariances (see [Invariances](#invariances)) | – |
| A11 | Alternative centre of mass weighted by main group number | `Dataset.py:71-91, 104-112` (unreachable) | Only Z weighting (Eq. 4.12) | Aligned: removed | – |
| A12 | Model selection manual only (final evaluation of the last epoch on the validation split) | `Training.py:310-348` | p. 52: epoch with the best validation MSE → test MAE | Aligned: `best.json`, `evaluate(epoch="best")` | `train.checkpoints`, `--epoch` |
| A13 | Mixed constants: grid 1/6 (Aspirin), model 86/28/30/6 (Stachyose), dataset folder Stachyose 2/4 | `Constants.py:25-32, 57-80` | Tab. 5.1/5.2 per molecule | Aligned: one preset per molecule; $N_{max}$ from the data | `configs/*.toml` |
| A14 | Self-exclusion by position equality: a second atom at the same position is silently skipped as if it were the atom itself (its row is missing, no error) | `Dataset.py:328, 246, 259` | not specified in the thesis | Self pair excluded by index; two atoms at distance 0 raise `ValueError` (B12) | – |
| A15 | $\lvert B\rvert = 1$ (one molecule per training step) | `Dataset.py:589-661` (no `.batch`) | p. 51; Sec. 7.4 ("technical limitations") | Conform, kept. Training with $\lvert B\rvert > 1$ would be the outlook of Sec. 7.4, i.e. new method, and is not implemented. Evaluation and inference process several molecules per call with the same result | – |
| A16 | First 65 000 frames, random frame split (frames are time-correlated) | `Dataset.py:415-421, 475-477` | p. 40: identical | Conform, kept; correlation risk documented ([Known limitations](#known-limitations)) | `data.n_samples_per_source`, `data.val_fraction`, `data.test_fraction`, `data.split_seed` |
| A17 | $Z_i$ not sorted by distance (`sort_matrix=False`) | `Constants.py:40` | p. 43: rows of $Z_i$ correspond to the rows of $\tilde R_i$; no distance sorting required | Conform, kept | `grid.sort_neighbours = false` |

### Open questions

Questions to the author. Each has a default that is implemented as long as there
is no other answer.

| ID | Question | Default |
|---|---|---|
| F1 | The legacy code assigns atoms differently from p. 43 / Fig. 4.1 (A01, A02). Should the default follow the thesis? | **Yes:** `grid.assignment = "thesis"` in all presets; `"legacy"` stays selectable and is set in `legacy-head` |
| F2 | Dropout was never active in training (A03), although the thesis describes 0.2. Activate it? | **Yes:** presets use `dropout_rate = 0.2`, active; `legacy-head` sets `0.0`, which is exactly the legacy behaviour |
| F3 | Which widths apply: Fig. 4.3/4.4 (embedding `[64, 128, 128, 256]`, 3 Res-Blocks in the energy network) or the legacy HEAD `9413bfd` of 10.10.2024 (`[64, 128, 128, 128]`, 2 Res-Blocks)? HEAD changed these values after the submission (05.10.2024, title page) | **Fig. 4.3/4.4** in the molecule presets, HEAD values in `legacy-head`; both configurable (A04, A05) |
| F4 | Force mode (direct force prediction, not part of the thesis, with the alignment bug B01, not equivariant): keep or remove? | **Keep as experimental**, default off, B01 fixed, limits documented in [Forces](#forces); no $-\nabla E$ |
| F5 | Under which licence are code and thesis PDF published? | **No licence file.** The README states that the licence is not yet specified (all rights reserved by the author). The PDF is included as requested |

<a id="invariances"></a>

## Invariances

The thesis requires invariance of the energy under translation, rotation and
permutation of identical atoms (Sec. 2.2, p. 26; Sec. 4.1, p. 37) and states it
for $D_i$ (p. 38) and for the input of the energy network (Fig. 4.4 caption,
p. 48). For DeepPot-SE [14] this holds because every row of $G_i$ depends on one
neighbour only (Eq. 4.7, p. 38): then $D_i$ is a sum over neighbours and does not
depend on their order. The thesis replaces this by **one** embedding network
over the flattened, ordered list $s_i \oplus Z_i$ (resp. $s_i \oplus F_i$) (p. 45,
Fig. 4.3). Every row of $G_i$ then depends on all neighbours and on their order,
and the order is defined by the axis-aligned grid. What follows (A10; tests in
`tests/test_invariance.py`, `tests/test_grid.py`, `tests/test_descriptors.py`,
`tests/test_model.py`):

| Property | Status | Reason |
|---|---|---|
| Translation | **Guaranteed.** Descriptors are bit-identical for shifts on a dyadic lattice, energies equal | The grid origin is the minimum of the positions (Eq. 4.11), so cells, levels and row order move with the molecule; long-range vectors are computed relative to the origin |
| Rotation, local neighbourhood (`"thesis"` mode) | **Guaranteed:** set of local neighbours, their $s$ values and the Gram matrix $\tilde R^{local} \tilde R^{local\top}$ (up to row order) | Membership depends only on distances (p. 43); $(\tilde R \tilde R^\top)_{jk} = s_j s_k (1 + \cos\theta_{jk})$ |
| Rotation, energy | **Not guaranteed** (test marked `xfail(strict=False)`) | The grid is axis-aligned (p. 42): a rotation changes the cells, so the long-range rows (centres of mass, counts) and the row order of both branches change. The embedding sees the ordered list, so $G_i$ and $D_i$ change |
| Permutation of identical atoms, descriptors | **Guaranteed:** the outputs are permuted with the atoms, the multiset of rows of every atom is unchanged | Rows depend on positions and atomic numbers only |
| Permutation, energy: order within every cell kept | **Guaranteed** | The row order is (level, cell rank, input index); such a permutation keeps the row order of every atom, and $\sum_i E_i$ does not depend on the order of the atoms |
| Permutation, energy: arbitrary | **Not guaranteed** (test marked `xfail(strict=False)`) | Swapping two identical atoms in the same cell swaps their rows in the ordered list seen by the embedding |
| Smoothness, local rows | **Guaranteed:** $s(r)$ is continuous at $r_{cs}$ and $r_c$, and in the `"thesis"` mode a neighbour's row goes to 0 continuously when it crosses $r_c$ | Eq. 4.5 with the case boundaries of [Descriptors](#descriptors) |
| Smoothness, long range | **Not guaranteed** | Sec. 7.1, p. 57: centres of mass jump when an atom changes cell or crosses $r_c$ (Fig. 7.1) |

Two consequences of the flattened embedding that the thesis does not discuss:

- When a neighbour crosses $r_c$, its row leaves $\tilde R_i^{local}$ and $Z_i$, and
  the following rows move up by one. The row itself goes to zero continuously,
  but its entry $(z_i, z_j)$ in $Z_i$ disappears and the flattened embedding
  input shifts, so $G_i$ and the energy can jump. The local descriptor is
  smooth in the sense of Eq. 4.5 (p. 57), the network input as a whole is not.
- Zero padding is harmless for $D_i$ (zero rows of $\tilde R$ contribute nothing),
  but the padded positions of the flattened input still feed the embedding
  network.

These are properties of the method as described on p. 45 and in Fig. 4.3 and
were deliberately not changed.

<a id="forces"></a>

## Forces (experimental)

The thesis trains on energies only: "The force labels were not used further"
(p. 40), loss Eq. 4.21 (p. 51). Forces appear only in the background: the force
is the negative gradient of the energy (Eq. 2.1, p. 20; $F = -\nabla U$, Sec. 2.2,
p. 26), and the DeepPot losses Eq. 2.3-2.6 (p. 20-21) and Eq. 4.10 (p. 38) weight
energy and force errors with $p_e$, $p_f$.

The legacy code contains an additional mode in which the network predicts the
forces **directly** (A09). `lrmlp` keeps it as experimental (F4):

- `model.predict_forces = true` (default `false`) gives the energy network four
  outputs per atom: $E_i$ and three force components.
- Loss: $p_e \cdot (\sum_i E_i - E)^2 + p_f \cdot \mathrm{mean}\big((F^w - F)^2\big)$,
  the mean over all atoms and components (as the $1/3N$ of Eq. 2.5). The
  prefactors move from a start value to a limit,
  $p(t) = p_{limit}(1 - f) + p_{start} f$, $f = r^{\,t/\tau}$
  (`train.p_energy_start/limit`, `train.p_force_start/limit`,
  `train.loss_weight_decay_steps` τ, `train.loss_weight_decay_rate` r; legacy
  `Training.py:14-16`). This schedule and its values are not specified in the
  thesis.
- The force labels are aligned with the descriptor rows (atom order = input
  order). In the legacy code they were not (B01), so legacy force models were
  trained on permuted labels.
- `lrmlp evaluate` reports `force_mae` and `force_rmse`; `lrmlp predict` writes
  `F`.

Limits:

- The forces are **not** $-\nabla E$ and are not conservative: they are a second
  network output, unrelated to the energy surface by construction.
- They are **not rotation-equivariant**: the network sees descriptors that are
  (at best) rotation-invariant, so it cannot produce a vector that rotates with
  the molecule.
- A $-\nabla E$ force (and a finite-difference test of it) is not possible with
  this design: the descriptors are computed once in NumPy during preprocessing
  (p. 40), so there is no differentiable path from positions to energy. Adding
  one would be a new method and is not implemented.

## Known limitations

- **Rotation and permutation invariance** hold only in the restricted sense of
  [Invariances](#invariances) (A10).
- **Long-range descriptor is not smooth** (Sec. 7.1, p. 57). The fixed grid
  (Sec. 4.2, p. 39) is not hierarchical (Sec. 7.2, p. 58).
- **$\lvert B\rvert = 1$** (p. 51). The validation loss fluctuates strongly (p. 53,
  Sec. 7.4, p. 58); larger batches are not implemented (A15).
- **Correlated split.** Frames of an MD trajectory are split at random (p. 40), so
  test frames have close neighbours in the training split (A16). The test MAE
  measures interpolation within the trajectory.
- **$N_{max}$ in the `"thesis"` mode.** $N_{max}$ of Tab. 5.2 (p. 52) was
  obtained with the legacy assignment. In the `"thesis"` mode no atom is dropped
  (A02) and every atom within $r_c$ is local (A01), so $N_{max}$, and with
  `k1 = "auto"` also $K_1$, can differ from Tab. 5.2. This could not be checked
  here because the repository contains no real data.
- **Comparability with Tab. 5.3.** The molecule presets differ from the code that
  produced the thesis results in assignment (A01, A02), dropout (A03) and
  possibly widths (A04, A05, F3). Use `configs/legacy-head.toml` to reproduce the
  legacy behaviour.
- **Fixed input size.** The network is built for the $N_{max}$ of its training
  dataset (p. 44). Inference on a frame in which an atom has more local or
  long-range rows raises an error instead of truncating.
- **Element range.** Atomic numbers above `grid.max_atomic_number` (default 100)
  or 255 are rejected by `lrmlp prepare`; a long-range cell can hold at most 255
  atoms of one element in the stored dataset.
- **Forces** are experimental and not physical in the sense of
  [Forces](#forces) (A09).
- **No hyperparameter study.** The thesis could not tune systematically
  (Sec. 7.5, p. 58); the values of Tab. 5.2 are not claimed to be optimal.

## Symbols and configuration

Thesis symbols, the configuration fields of `lrmlp` (TOML, see the presets in
[`configs/`](../configs/)) and the legacy constants of
`Neural-Network/Constants.py` at `9413bfd`. The complete field reference is
`docs/configuration.md`.

| Symbol | Thesis | Config field | Default | Legacy constant |
|---|---|---|---|---|
| $\Delta g$ | grid length, Eq. 4.11, Tab. 5.1 | `grid.grid_size` | 1.0 Å | `Dataset.GRID_SIZE` |
| $l_{max}$ | maximum local level, Eq. 4.13, Tab. 5.1 | `grid.max_local_level` | 6 | `Dataset.MAX_LOCAL_LEVEL` |
| $r_c$ | cutoff, Eq. 4.14 | derived `grid.cutoff` | 7.71 Å | `Dataset.CUT_OFF` |
| $r_{cs}$ | inner cutoff, Eq. 4.5, Tab. 5.1 ($r_c - 0.2$ Å) | `grid.inner_cutoff_offset` (derived `grid.inner_cutoff`) | 0.2 Å | `Dataset.INNER_CUT_OFF = CUT_OFF − 0.2` |
| $m_{max}$ | length of $f^k$, p. 42 (value not specified) | `grid.max_atomic_number` | 100 | `Dataset.MAX_ATOM_ELEMENTS` |
| – | local environment rule, p. 43 | `grid.assignment` | `"thesis"` | `"legacy"` = `Dataset.py:310-356` |
| – | row order of $Z_i$, p. 43 | `grid.sort_neighbours` | `false` | `Dataset.sort_matrix` |
| – | datasets [16], Sec. 4.3.1 | `data.sources` | – | `create_dataset(paths)` |
| – | first 65 000 frames, p. 40 | `data.n_samples_per_source` | 65000 (`"all"` possible) | `n_samples_per` |
| – | 80/10/10 split, p. 40 | `data.val_fraction`, `data.test_fraction`, `data.split_seed` | 0.1, 0.1, 42 | `Dataset.py:475-477` |
| $N_{max}^{local}$, $N_{max}^{long}$ | padding, p. 44, Tab. 5.2 | – (from the dataset, `metadata.json`) | – | `Model.n_max_local`, `Model.n_max_long_range` (hard-coded 86/28) |
| – | ELU, Fig. 4.2, 4.4 | `model.activation` | `"elu"` | `Model.activation` |
| – | embedding widths, Fig. 4.3 | `model.embedding_dims` | [64, 128, 128, 256] | `Hyperparameters.embedding_dims` ([64, 128, 128, 128]) |
| – | energy network, Fig. 4.4 | `model.fitting_width`, `model.fitting_res_blocks` | 128, [128, 128, 128] | `Model.py:54-59` (128, [128, 128]) |
| – | dropout 0.2, Fig. 4.2, 4.4 | `model.dropout_rate` | 0.2 | 0.2, never active (A03) |
| $K_1$ ($M_1$) | columns of $G_i$, p. 45, Tab. 5.2 | `model.k1_local`, `model.k1_long` | `"auto"` (= $N_{max}$) | `Hyperparameters.M1_local`, `M1_long` |
| $K_2$ ($M_2$) | columns of $G_{i2}$, p. 45, Tab. 5.2 | `model.k2_local`, `model.k2_long` | 4, 1 | `Hyperparameters.M2_local`, `M2_long` |
| – | Fig. 4.5 vs. Fig. 4.6 | `model.use_long_range` | `true` | `Model.use_long_range` |
| – | forces not trained, p. 40 | `model.predict_forces` | `false` | `not Model.predict_only_energy` |
| – | epochs, Tab. 5.2 | `train.epochs` | 200 | `Hyperparameters.epochs` |
| $\eta_{init}$ | Eq. 4.22, Tab. 5.2 | `train.learning_rate` | 1e-4 | `Hyperparameters.initial_learning_rate` |
| s | decay steps, Eq. 4.22, Tab. 5.2 | `train.lr_decay_steps` | 2.08e5 | `Hyperparameters.lr_decay_steps` |
| 0.97 | decay base, Eq. 4.22 | `train.lr_decay_rate` | 0.97 | `Hyperparameters.decay_rate` |
| – | continuous decay, Eq. 4.22 | `train.lr_staircase` | `true` (A07) | `staircase=True` |
| $p_e$, $p_f$ | prefactors, Eq. 4.10 (force mode only) | `train.p_energy_start/limit`, `train.p_force_start/limit` | 1, 1, 1, 0 | `Hyperparameters.p_*_start/limit` |
| – | not specified in the thesis | `train.loss_weight_decay_steps`, `train.loss_weight_decay_rate` | 7.5e3, 0.97 | `Hyperparameters.decay_steps`, `decay_rate` (shared with the learning rate) |
| – | not specified in the thesis | `train.shuffle_buffer` | 1000 | `Dataset.SHUFFLE_BUFFER_SIZE` |
| – | not specified in the thesis | `train.seed`, `train.deterministic` | 42, `false` | no seeds |
| – | model selection, p. 52 | `train.checkpoints` | `"best_and_last"` | every epoch saved |
| $\lvert B\rvert$ | batch size, p. 51 | – (fixed to 1) | 1 | no `.batch` (`BATCH_SIZE = 64` unused) |
