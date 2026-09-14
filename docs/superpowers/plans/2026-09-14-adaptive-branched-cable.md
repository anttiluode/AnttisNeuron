# Adaptive Branched Cable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic Gate 5 that tests whether an alignment-blind local conductance rule on a real branched graph can improve bounded Oja-to-physical-mode alignment on held-out input statistics.

**Architecture:** `anttis_neuron/cable.py` owns the fixed tree, graph Laplacian/cable operator, bounded sensing, simulation, and local conductance adaptation. `experiments/gate5_adaptive_cable.py` constructs common-tape frozen/adaptive/shuffled/uniform conditions and reports scientific metrics without asserting a positive result. Existing experiment infrastructure, Oja learning, receipts, paper, README, and CI are extended rather than replaced.

**Tech Stack:** Python 3.11+, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-adaptive-branched-cable-design.md`

## Global Constraints

- Use NumPy + pytest only; no PyTorch/JAX/NEURON dependency.
- All random experiments use explicit NumPy generators and fixed seeds.
- Gate 5 scientific null/negative results are valid; CI fails only on engineering invariants.
- The local adaptor must not accept physical eigenvectors, Oja weights, alignment scores, labels, or held-out metrics.
- Common-tape controls must receive identical external input arrays.
- Conductances remain finite, positive, bounded, and mean-normalized.
- The near-uniform global mode is excluded from the main physical-mode alignment score.
- README/PAPER numbers must come from the committed `results/gate5.json` receipt.

---

### Task 1: Branched graph cable operator

**Files:**
- Create: `anttis_neuron/cable.py`
- Create: `tests/test_cable.py`

**Interfaces:**
- Produces `default_tree() -> tuple[int, np.ndarray]`, where edges have shape `(n_edges, 2)`.
- Produces `weighted_laplacian(n_nodes: int, edges: np.ndarray, conductances: np.ndarray) -> np.ndarray`.
- Produces `cable_operator(n_nodes, edges, conductances, *, dt, leak, coupling) -> np.ndarray`.
- Produces `default_ports_and_sensors(n_nodes: int) -> tuple[np.ndarray, np.ndarray]` returning `B` shape `(n_nodes, 4)` and `S` shape `(4, n_nodes)`.
- Produces `simulate_cable(inputs, A, B, S, *, burn=0) -> np.ndarray` returning bounded observations.
- Produces `visible_physical_modes(A, S, *, exclude_uniform=True) -> np.ndarray` with normalized sensor-space modes as rows.

- [ ] **Step 1: Write failing operator tests**

```python
import numpy as np

from anttis_neuron.cable import (
    cable_operator,
    default_ports_and_sensors,
    default_tree,
    simulate_cable,
    visible_physical_modes,
    weighted_laplacian,
)
from anttis_neuron.branch import spectral_radius


def test_default_tree_is_connected_tree():
    n, edges = default_tree()
    assert n == 11
    assert edges.shape == (10, 2)
    seen = {0}
    for _ in range(n):
        for i, j in edges:
            if int(i) in seen or int(j) in seen:
                seen.add(int(i)); seen.add(int(j))
    assert seen == set(range(n))


def test_weighted_laplacian_is_symmetric_zero_row_sum():
    n, edges = default_tree()
    g = np.linspace(0.7, 1.3, len(edges))
    L = weighted_laplacian(n, edges, g)
    assert np.allclose(L, L.T, atol=1e-12)
    assert np.allclose(L.sum(axis=1), 0.0, atol=1e-12)


def test_default_operator_is_stable_and_observation_shape_is_bounded():
    n, edges = default_tree()
    A = cable_operator(n, edges, np.ones(len(edges)), dt=0.08, leak=0.08, coupling=0.65)
    assert spectral_radius(A) < 1.0
    B, S = default_ports_and_sensors(n)
    x = np.zeros((20, 4)); x[0, 0] = 1.0
    z = simulate_cable(x, A, B, S)
    assert z.shape == (20, 4)
    assert np.all(np.isfinite(z))


def test_visible_modes_are_normalized_and_nontrivial():
    n, edges = default_tree()
    A = cable_operator(n, edges, np.ones(len(edges)), dt=0.08, leak=0.08, coupling=0.65)
    _, S = default_ports_and_sensors(n)
    modes = visible_physical_modes(A, S)
    assert modes.ndim == 2 and modes.shape[1] == 4
    assert modes.shape[0] >= 2
    assert np.allclose(np.linalg.norm(modes, axis=1), 1.0, atol=1e-12)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_cable.py -v`

Expected: import failure because `anttis_neuron.cable` does not exist.

- [ ] **Step 3: Implement the operator**

Use this fixed topology:

```python
edges = np.array([
    [0, 1], [1, 2], [2, 3],
    [2, 4], [4, 5], [5, 6],
    [2, 7], [7, 8], [8, 9], [9, 10],
], dtype=int)
```

Use distal input ports `[3, 6, 9, 10]` and sensors `[1, 5, 8, 10]`, each as one-hot columns/rows. Implement the weighted Laplacian by accumulating `+g` on both incident diagonals and `-g` symmetrically off diagonal. Construct `A = I - dt * (leak * I + coupling * L)` and reject nonfinite parameters or `spectral_radius(A) >= 1`.

`simulate_cable` starts from zero state and applies `d = A @ d + B @ x_t`, returning `S @ d` after burn-in.

`visible_physical_modes` uses `np.linalg.eigh(A)` because the graph operator is symmetric, orders modes by descending absolute eigenvalue, projects each eigenvector with `S`, excludes the most uniform state-space eigenvector by maximum cosine with `ones/sqrt(n)`, discards near-zero sensor projections, normalizes the remainder, and removes sensor-space duplicates with absolute cosine above `1 - 1e-10`.

- [ ] **Step 4: Run operator tests and full suite**

Run: `pytest tests/test_cable.py -v && pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add anttis_neuron/cable.py tests/test_cable.py
git commit -m "feat: add branched graph cable operator"
```

---

### Task 2: Alignment-blind local conductance adaptation

**Files:**
- Modify: `anttis_neuron/cable.py`
- Modify: `tests/test_cable.py`

**Interfaces:**
- Produces `edge_difference_energy(states: np.ndarray, edges: np.ndarray) -> np.ndarray`.
- Produces `normalize_mean_conductance(g: np.ndarray, target_mean: float, g_min: float, g_max: float) -> np.ndarray`.
- Produces `adapt_conductances(states, edges, conductances, *, eta, q_target, g_min, g_max, mode="local", rng=None) -> np.ndarray`.
- Extend simulation with `return_states: bool = False`; when true return `(observations, states)`.

- [ ] **Step 1: Add failing adaptation tests**

```python
import inspect


def test_local_adaptor_signature_has_no_alignment_or_mode_argument():
    names = set(inspect.signature(adapt_conductances).parameters)
    forbidden = {"alignment", "score", "eigenvectors", "physical_modes", "oja", "labels"}
    assert names.isdisjoint(forbidden)


def test_mean_normalization_and_bounds_are_preserved():
    n, edges = default_tree()
    rng = np.random.default_rng(4)
    states = rng.normal(size=(200, n))
    g0 = np.ones(len(edges))
    g1 = adapt_conductances(states, edges, g0, eta=0.08, q_target=1.0, g_min=0.45, g_max=1.8)
    assert np.all(np.isfinite(g1))
    assert np.all((g1 >= 0.45) & (g1 <= 1.8))
    assert np.isclose(g1.mean(), g0.mean(), atol=1e-10)
    assert np.std(g1) > 0.0


def test_uniform_control_preserves_relative_conductances():
    n, edges = default_tree()
    rng = np.random.default_rng(5)
    states = rng.normal(size=(100, n))
    g0 = np.linspace(0.8, 1.2, len(edges))
    g1 = adapt_conductances(states, edges, g0, eta=0.1, q_target=1.0, g_min=0.4, g_max=2.0, mode="uniform")
    assert np.allclose(g1 / g1.mean(), g0 / g0.mean(), atol=1e-12)


def test_shuffled_control_is_deterministic_for_fixed_rng():
    n, edges = default_tree()
    rng_states = np.random.default_rng(6)
    states = rng_states.normal(size=(150, n))
    g0 = np.ones(len(edges))
    a = adapt_conductances(states, edges, g0, eta=0.05, q_target=1.0, g_min=0.4, g_max=2.0, mode="shuffled", rng=np.random.default_rng(9))
    b = adapt_conductances(states, edges, g0, eta=0.05, q_target=1.0, g_min=0.4, g_max=2.0, mode="shuffled", rng=np.random.default_rng(9))
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run adaptation tests and verify RED**

Run: `pytest tests/test_cable.py -v`

Expected: failures because adaptation functions/signatures are missing.

- [ ] **Step 3: Implement local adaptation**

For states shape `(T, n_nodes)`, compute per-edge energy `mean((states[:, i] - states[:, j]) ** 2)`.

Define raw log-update

```python
signal = energy / q_target - 1.0
```

For `mode="local"`, use signal as-is. For `mode="shuffled"`, permute signal with the required RNG. For `mode="uniform"`, replace signal by its scalar mean at every edge. Update with

```python
proposal = g * np.exp(eta * signal)
```

then clip and mean-normalize with a bounded iterative rescale: repeatedly multiply by `target_mean/current_mean`, clip, and stop when mean error is `<1e-12` or after 20 iterations; reject impossible target means outside `[g_min, g_max]`.

- [ ] **Step 4: Run adaptation tests and full suite**

Run: `pytest tests/test_cable.py -v && pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add anttis_neuron/cable.py tests/test_cable.py
git commit -m "feat: add local cable conductance adaptation"
```

---

### Task 3: Gate 5 common-tape adaptation and held-out evaluation

**Files:**
- Create: `experiments/gate5_adaptive_cable.py`
- Modify: `tests/test_experiments.py`

**Interfaces:**
- Produces `run(seed: int = 17) -> dict` matching existing gate convention.
- Private helper `_make_input_tapes(seed) -> tuple[np.ndarray, list[np.ndarray]]` returns one adaptation tape and at least five held-out tapes.
- Private helper `_train_structure(mode, adaptation_tape, ...)` returns final conductances plus diagnostics.
- Private helper `_evaluate(g, tapes, ...)` returns per-tape and aggregate alignment metrics.

- [ ] **Step 1: Write failing experiment invariants**

Add Gate 5 import and include it in determinism/finiteness iteration.

```python
from experiments.gate5_adaptive_cable import run as run5


def test_gate5_reports_controls_without_requiring_positive_result():
    result = run5(seed=17)
    assert result["gate"] == 5
    assert result["seed"] == 17
    for key in (
        "frozen_heldout_mean_alignment",
        "adaptive_heldout_mean_alignment",
        "shuffled_heldout_mean_alignment",
        "uniform_heldout_mean_alignment",
    ):
        assert 0.0 <= result[key] <= 1.0
    assert -1.0 <= result["adaptive_minus_frozen"] <= 1.0
    assert -1.0 <= result["adaptive_minus_shuffled"] <= 1.0
    assert len(result["adaptive_conductances"]) == 10
    assert result["adaptive_spectral_radius"] < 1.0
    assert result["common_tape_digest"] == result["control_tape_digest"]
```

Do **not** assert that either delta is positive.

- [ ] **Step 2: Run focused test and verify RED**

Run: `pytest tests/test_experiments.py -v`

Expected: import failure because Gate 5 does not exist.

- [ ] **Step 3: Implement deterministic Gate 5**

Use physical defaults from Tasks 1-2. Generate external four-dimensional Gaussian inputs with covariance

```python
C(theta, spectrum) = R01(theta) @ diag(spectrum) @ R01(theta).T
```

where `R01` rotates dimensions 0 and 1 and leaves dimensions 2-3 unchanged.

Adaptation phases use angles `[12, 31, 53, 74]` degrees, 2500 samples each, spectra alternating `[2.6, 0.8, 1.2, 0.6]` and `[1.8, 0.7, 2.1, 0.5]`. Held-out angles are `[22, 42, 63, 83, 103]` degrees with 5000 samples each and spectra `[2.2, 0.9, 1.6, 0.55]`.

Start all controls from `g=np.ones(10)`. For each adaptation phase and each condition, build `A`, simulate the **same** phase input tape with `return_states=True`, then apply one structural update for local/shuffled/uniform. Frozen never changes.

Set `eta=0.10`, `g_min=0.45`, `g_max=1.8`. Set `q_target` once from the median edge-difference energy of the initial frozen structure on the first adaptation phase; use that same scalar in every condition.

For evaluation, build each condition's final `A`; generate bounded observations on each held-out tape; discard 250 burn-in samples; RMS-normalize observations; fit Oja with `lr=0.001`, `epochs=3`, and a deterministic seed derived from condition+tape index. Evaluate maximum cosine against that condition's current visible nontrivial physical modes.

Compute a SHA-256 digest over the raw adaptation and held-out tape bytes once and place the same value in `common_tape_digest` and `control_tape_digest` to make common-random-number use inspectable.

Report conductances, spectral radii, held-out per-tape alignments and means, adaptive-minus-frozen, adaptive-minus-shuffled, coefficient of variation, `q_target`, and an edge-energy/conductance-change Pearson correlation (0 if either vector has zero variance).

- [ ] **Step 4: Run Gate 5 and full suite**

Run:

```bash
pytest tests/test_experiments.py -v
python experiments/gate5_adaptive_cable.py --out /tmp/gate5.json
pytest -q
```

Expected: deterministic finite receipt and all engineering tests green. Scientific deltas may have either sign.

- [ ] **Step 5: Commit**

```bash
git add experiments/gate5_adaptive_cable.py tests/test_experiments.py
git commit -m "exp: add adaptive branched cable gate"
```

---

### Task 4: Freeze receipt, document result, and extend CI

**Files:**
- Create: `results/gate5.json`
- Modify: `README.md`
- Modify: `PAPER.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- CI invokes `python experiments/gate5_adaptive_cable.py --out /tmp/gate5.json` after Gates 0-4.
- README/PAPER quote exact values from the committed receipt.

- [ ] **Step 1: Generate the canonical receipt**

Run: `python experiments/gate5_adaptive_cable.py --seed 17 --out results/gate5.json`

Inspect the JSON and classify the result as positive, mixed, null, or negative using the signs/patterns of `adaptive_minus_frozen`, `adaptive_minus_shuffled`, and per-held-out alignments. Do not change model parameters after seeing the receipt unless an engineering defect is found; scientific disappointment is not a defect.

- [ ] **Step 2: Update README and PAPER from the receipt**

README adds Gate 5 to the gate table and a short interpretation paragraph. PAPER adds a Gate 5 methods subsection, results row(s), interpretation, and limitations. Both must explicitly say the local rule was alignment-blind and that the graph is synthetic.

- [ ] **Step 3: Extend CI smoke run**

Append:

```yaml
python experiments/gate5_adaptive_cable.py --out /tmp/gate5.json
```

to the existing scientific-gate smoke step.

- [ ] **Step 4: Fresh clean verification**

Run:

```bash
python -m pip install -e ".[test]"
pytest -q
for g in experiments/gate{0,1,2,3,4,5}_*.py; do python "$g" --out "/tmp/$(basename "$g" .py).json"; done
```

Expected: install succeeds, all tests pass, all six gate scripts exit zero.

- [ ] **Step 5: Commit**

```bash
git add results/gate5.json README.md PAPER.md .github/workflows/ci.yml
git commit -m "docs: report adaptive cable gate"
```

## Plan self-review

- Spec coverage: operator, bounded observation, local rule, controls, common tape, held-out evaluation, engineering invariants, receipt, documentation, and CI are each assigned to a task.
- No scientific-positive assertion appears in CI/tests.
- The adaptor API has no spectral/alignment argument.
- Function names used by Tasks 2-4 match Task 1/2 interfaces.
- No placeholder/TODO steps remain.