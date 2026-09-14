# Gate 5B Spectral Audit, Gate 6 Many Worlds, and Visual Explainer Design

**Date:** 14 September 2026

## Purpose

AnttisNeuron Gate 5 established one controlled synthetic case in which an alignment-blind local conductance rule on an 11-node branched graph cable improved held-out Oja-to-physical-mode alignment. The next work must distinguish *how* that improvement happened before testing whether it survives many different worlds.

The order is fixed:

1. **Gate 5B — spectral autopsy** of the already-frozen Gate 5 mechanism.
2. **Gate 6 — many-world robustness** with the Gate 5 local rule and hyperparameters frozen before seeing results.
3. **`index.html` — visual explainer** that separates the playful hand-waving entry point from the executable scientific claims.

Gate 5 remains unchanged. Gate 5B and Gate 6 are additive experiments with their own receipts and tests.

## Scientific discipline

- A negative or mixed Gate 5B/Gate 6 result is scientifically valid.
- CI must never assert that scientific deltas are positive.
- The Gate 5 adaptation rule is frozen for Gate 6. Do not tune `eta`, `q_target` construction, conductance bounds, Oja hyperparameters, or scoring after seeing multi-world failures.
- Common-random-number controls must receive identical external tapes and identical Oja initialization per matched evaluation.
- Shuffled-credit controls must receive the exact local update-value multiset for each phase, merely permuted onto the wrong edges.
- All receipts must be deterministic under an explicit seed.
- Interpretations must distinguish physical eigenvalue changes, eigenvector/subspace changes, observation geometry, and Oja covariance tracking.

---

## Gate 5B — spectral autopsy

### Question

Gate 5 improved held-out alignment. Did that happen mainly because adaptation changed:

1. **timescales/eigenvalues** while leaving the physical basis mostly fixed,
2. **the physical eigenvectors/subspace** while preserving similar timescales,
3. **both**, or
4. mostly the **sensor-visible covariance** without meaningful state-space basis reorientation?

Gate 5B must answer this without changing Gate 5 training.

### Inputs

Gate 5B imports and reuses the deterministic Gate 5 construction:

- same 11-node topology,
- same ports and sensors,
- same adaptation and held-out tapes,
- same seed `17` for the frozen receipt,
- same final frozen and adaptive conductances,
- same Oja settings,
- same three slowest visible non-uniform physical modes used by Gate 5.

It may call Gate 5 private helpers because both experiments live in the same repository and Gate 5B is explicitly a diagnostic audit of Gate 5. It must not modify Gate 5 output.

### Spectral decomposition

For frozen and adapted symmetric cable operators,

```math
A_f = Q_f \Lambda_f Q_f^T,
\qquad
A_a = Q_a \Lambda_a Q_a^T.
```

The audit records:

- all sorted eigenvalues,
- the three slowest non-uniform eigenvalues,
- absolute eigenvector overlap matrix `|Q_f^T Q_a|`, excluding the uniform mode for the main summary,
- best-match overlap for each of the three slowest non-uniform frozen modes,
- principal angles between the frozen and adapted three-dimensional slow non-uniform subspaces,
- corresponding projected/sensor-visible mode overlaps.

Because eigenvectors are sign-ambiguous, all single-vector comparisons use absolute cosine.

### Diagnostic counterfactual operators

Construct two symmetric diagnostic operators:

```math
A_lambda = Q_f Lambda_a Q_f^T
```

(new spectrum, frozen basis) and

```math
A_Q = Q_a Lambda_f Q_a^T
```

(adapted basis, frozen spectrum).

These are **diagnostic linear operators only**. They are not claimed to correspond to a realizable positive-conductance tree.

Run the same five held-out tapes through:

- frozen `A_f`,
- eigenvalue-only `A_lambda`,
- eigenvector-only `A_Q`,
- fully adapted `A_a`.

Use the same sensor matrix, input matrix, burn-in, RMS normalization, and per-tape Oja initialization. Score Oja against the physical modes belonging to the operator under evaluation.

Report mean, median, minimum, and per-tape alignment for all four conditions.

### Cross-basis learner audit

For each held-out tape, retain the learned Oja vector for the frozen and adapted systems. Measure four pairings:

- frozen Oja vs frozen visible modes,
- frozen Oja vs adapted visible modes,
- adapted Oja vs frozen visible modes,
- adapted Oja vs adapted visible modes.

Aggregate each pairing across held-out tapes. This distinguishes whether the learned statistical direction moved toward a changed physical basis, whether the basis itself moved toward the learned direction, or both.

### Gate 5B interpretation

Gate 5B must not force a categorical label. It reports decomposition evidence. The README/paper may say “mostly eigenvalue”, “mostly basis”, or “mixed” only if the numerical controls support that description.

---

## Gate 6 — many-world robustness

### Question

Does the Gate 5 local adaptation advantage survive when topology, port placement, sensor placement, covariance geometry, and random seed vary under one frozen rule?

### Frozen rule

Gate 6 inherits these constants from Gate 5 unchanged:

- `dt = 0.08`
- `leak = 0.08`
- `coupling = 0.65`
- `eta = 0.10`
- `g_min = 0.45`
- `g_max = 1.8`
- four adaptation phases
- five held-out covariance environments
- Oja `lr = 0.001`, `epochs = 3`
- 250-sample evaluation burn-in

The homeostatic target `q_target` remains measured once from the initial frozen structure on the first adaptation phase of each world, exactly as in Gate 5.

### World generation

Gate 6 uses a deterministic suite of **24 worlds** by default. A world is generated from a master seed and contains:

- an 11-node tree,
- four distinct input ports selected from leaves when at least four leaves exist, otherwise from distinct non-root nodes,
- four distinct sensor nodes selected independently from non-root nodes,
- adaptation covariance angles and spectra derived from the world RNG around the Gate 5 scale,
- five held-out covariance angles not identical to the adaptation angles,
- independent external Gaussian tapes shared by every control in that world.

Tree generation must always produce a connected acyclic graph with 11 nodes. Use a deterministic random-parent construction: for each node `i=1..10`, connect it to a uniformly sampled parent in `[0, i-1]`. Reject/resample worlds whose default physical operator is unstable or whose sensor projection exposes fewer than three non-uniform physical modes.

World generation has a bounded deterministic retry count and raises a clear error if a valid world cannot be formed.

### Conditions

Every world evaluates:

1. frozen structure,
2. correct local adaptation,
3. exact-multiset shuffled credit,
4. uniform credit.

The shuffled control consumes the local condition's exact per-phase update-signal multiset, permuted with an explicit world-specific RNG.

### Per-world outputs

Each world reports:

- frozen/local/shuffled/uniform held-out mean alignment,
- adaptive-minus-frozen delta,
- adaptive-minus-shuffled delta,
- held-out minimum alignment for frozen/local/shuffled,
- final conductance coefficient of variation,
- spectral radius for every condition,
- exact shuffled update-multiset error,
- common-tape digest,
- topology edge list,
- port nodes,
- sensor nodes,
- world seed.

### Aggregate outputs

Gate 6 reports across all 24 worlds:

- mean and median `adaptive_minus_frozen`,
- mean and median `adaptive_minus_shuffled`,
- lower quartile and minimum of both deltas,
- fraction of worlds with `adaptive_minus_frozen > 0`,
- fraction of worlds with `adaptive_minus_shuffled > 0`,
- mean frozen/local/shuffled/uniform alignment,
- count of worlds where local loses to frozen,
- count of worlds where local loses to shuffled,
- per-world records sorted by world index.

No sign-based threshold is an engineering pass/fail criterion.

### Gate 6 runtime

The default 24-world experiment should remain practical in GitHub Actions. Tests may use a small smoke configuration such as 2–3 worlds, but the committed scientific receipt uses all 24 worlds. Reuse NumPy only; add no heavy dependency.

---

## Visual explainer — `index.html`

### Purpose

The page is the human entry point for the accompanying informal video and the repository. It may be playful at the top but must make a hard transition into falsifiable claims.

### Constraints

- one self-contained `index.html` at repository root,
- no external JavaScript or CSS dependencies,
- works as a static GitHub Pages file,
- no invented or placeholder video URL,
- responsive on desktop/mobile,
- no claim beyond the committed receipts.

### Visual structure

The hero uses the informal framing “Wild hand waving” and presents an animated schematic:

```text
statistical input / synapses
        -> branched dendritic operator
        -> soma mixture
        -> AIS-like output boundary
        -> axon
```

The animation should be explanatory rather than biologically literal: pulses enter several branches, branch paths accumulate different intensity, and the downstream boundary fires when pooled activity crosses a visual threshold.

A prominent divider states:

> **Hand waving ends here. Executable claims begin here.**

Below it, show compact cards for Gates 0–6. Each card has:

- the question,
- the result or current status,
- one sentence on what it does **not** establish,
- links to the relevant source/receipt where available.

The Gate 5 card visualizes frozen `0.782185`, adaptive `0.900191`, and shuffled `0.759219`. Gate 5B and Gate 6 values are populated only after their receipts are frozen.

Footer links to `README.md`, `PAPER.md`, `results/`, and the repository source.

### Testing

Add a lightweight pytest that reads `index.html` as text and verifies:

- it exists,
- it contains the hard-divider sentence,
- it references Gates 0 through 6,
- it contains no `<script src=` or `<link rel="stylesheet" href=` external dependency,
- it contains no fake video URL marker.

The page itself does not become evidence; the receipts remain authoritative.

---

## Documentation updates

After Gate 5B and Gate 6 are frozen:

- add both rows to the README gate table,
- add experiment/method/result/limitations text to `PAPER.md`,
- update “next experiments” so AIS/load adaptation comes after the robustness result,
- explicitly retain the caveat that the graph cable is a synthetic leaky-diffusion model, not a conductance-based biological dendrite.

## Success criteria

Engineering success means:

- deterministic Gate 5B and Gate 6 receipts,
- complete Python 3.11/3.12 CI,
- Gate 5 unchanged,
- exact shuffled-control invariants preserved,
- `index.html` static and dependency-free,
- documentation values copied from frozen receipts.

Scientific success is **not predeclared**. Gate 5B may show mostly timescale change, mostly basis rotation, or a mixed mechanism. Gate 6 may confirm, weaken, or reject generality. All are acceptable outcomes if measured honestly.