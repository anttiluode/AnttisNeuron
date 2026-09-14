# Gate 7 Load Compensation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a falsifiable Gate 7 experiment testing whether a purely local AIS-like firing-rate homeostat can preserve a nontrivial input-output transfer function across the 24 existing dendritic worlds better than a fixed boundary, with a closed-form load-ratio oracle as a diagnostic reference.

**Architecture:** Keep Gate 6 numerically unchanged and reuse its deterministic world generation plus local dendritic adaptation. Add a focused `output_boundary.py` module for driving-point load, logistic rate-boundary operations, threshold calibration, homeostatic gain updates, and curve metrics; add a Gate-7 experiment layer that produces fresh calibration/evaluation tapes, freezes the three output-boundary conditions, aggregates the 24-world result, and writes a frozen receipt. CI tests engineering invariants only and never requires a positive scientific result.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-gate7-load-compensation-design.md`

## Global Constraints

- Reuse the 24 deterministic Gate-6 `CableWorld` instances; do not change `results/gate6.json` or Gate-6 numerical behavior.
- Reproduce the Gate-6 local dendritic adaptation, then freeze dendritic conductances before output-boundary adaptation.
- Treat node `0` as the soma/output-boundary interface.
- Gate-7 calibration seed: `SeedSequence([7017, world.seed, world.index])`; evaluation seed: `SeedSequence([8017, world.seed, world.index])`.
- Calibration tape length: `3000`; evaluation base tape length: `5000`; cable burn-in: `250`.
- Reference world: world `0`; reference target rate: `0.25`; `beta = 2 / soma_voltage_std`; calibrate `theta` by deterministic bisection at unit input scale.
- Fixed boundary gain: `a = 1`.
- Homeostat: 12 phases, `log(a) += 0.5 * (0.25 - mean_rate)`, clipped to `a in [0.2, 5.0]`; input interface contains only mean rate, target rate, and current gain.
- Oracle: `a_oracle = g_load(world) / g_load(reference)`, unclipped.
- Evaluation scales: exactly `(0.50, 0.75, 1.00, 1.25, 1.50)`.
- Rheobase analogue threshold: `0.10`; F-I gain uses least-squares slope at `(0.75, 1.00, 1.25)`.
- Reference transfer-curve dynamic range must be at least `0.10`.
- Unit-scale rates `<0.02` or `>0.98` are flagged collapsed but remain in aggregates.
- No scientific-positive result is encoded as a correctness test.

---

### Task 1: Output-boundary primitives

**Files:**
- Create: `anttis_neuron/output_boundary.py`
- Create: `tests/test_output_boundary.py`

**Interfaces:**
- Consumes: `anttis_neuron.cable.weighted_laplacian` and NumPy arrays.
- Produces:
  - `driving_point_conductance(n_nodes: int, edges: np.ndarray, conductances: np.ndarray, *, leak: float, coupling: float, soma_node: int = 0) -> float`
  - `rate_boundary(voltage: np.ndarray, *, gain: float, theta: float, beta: float) -> np.ndarray`
  - `calibrate_threshold(voltage: np.ndarray, *, gain: float, beta: float, target_rate: float, iterations: int = 80) -> float`
  - `homeostatic_gain(gain: float, mean_rate: float, *, target_rate: float = 0.25, eta: float = 0.5, gain_min: float = 0.2, gain_max: float = 5.0) -> float`
  - `transfer_metrics(scales: np.ndarray, rates: np.ndarray, reference_rates: np.ndarray, *, rheobase_rate: float = 0.10) -> dict`

- [ ] **Step 1: Write failing primitive tests**

Create tests that verify: a two-node leaky graph has positive finite Schur-complement load; the load matches direct scalar Schur-complement arithmetic; `rate_boundary` is finite and bounded in `[0,1]`; threshold calibration reaches the requested mean rate to `1e-10`; `homeostatic_gain` increases when rate is below target, decreases when above target, and obeys `[0.2,5.0]`; `transfer_metrics` returns zero RMSE against itself, interpolates the `0.10` crossing, and computes the three-central-point least-squares slope.

Representative test fixture:

```python
import math
import numpy as np

from anttis_neuron.output_boundary import (
    calibrate_threshold,
    driving_point_conductance,
    homeostatic_gain,
    rate_boundary,
    transfer_metrics,
)


def test_two_node_driving_point_conductance_matches_schur_complement():
    edges = np.array([[0, 1]], dtype=int)
    g = np.array([2.0])
    got = driving_point_conductance(2, edges, g, leak=0.08, coupling=0.65)
    G00 = 0.08 + 0.65 * 2.0
    G11 = G00
    G01 = -0.65 * 2.0
    expected = G00 - (G01 * G01) / G11
    assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
    assert got > 0.0


def test_threshold_calibration_hits_target_rate():
    voltage = np.linspace(-1.0, 1.0, 2001)
    theta = calibrate_threshold(voltage, gain=1.0, beta=2.0, target_rate=0.25)
    rate = rate_boundary(voltage, gain=1.0, theta=theta, beta=2.0)
    assert abs(float(np.mean(rate)) - 0.25) < 1e-10
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q tests/test_output_boundary.py`
Expected: collection fails because `anttis_neuron.output_boundary` does not yet exist.

- [ ] **Step 3: Implement minimal output-boundary module**

Use the weighted graph Laplacian to construct `G = leak*I + coupling*L`. Compute the Schur complement with `np.linalg.solve` rather than an explicit inverse. Validate finite positive parameters and arrays. Implement a numerically stable logistic with clipped logits, deterministic 80-iteration bisection for `theta`, log-space homeostatic gain clipping, and transfer metrics with linear crossing interpolation; use `None` when the sweep never crosses the rheobase threshold.

Core formulas:

```python
G = leak * np.eye(n_nodes) + coupling * weighted_laplacian(n_nodes, edges, conductances)
rest = np.array([i for i in range(n_nodes) if i != soma_node], dtype=int)
g00 = float(G[soma_node, soma_node])
g0r = G[soma_node, rest]
grr = G[np.ix_(rest, rest)]
load = g00 - float(g0r @ np.linalg.solve(grr, G[rest, soma_node]))

z = np.clip(beta * (gain * voltage - theta), -60.0, 60.0)
rates = 1.0 / (1.0 + np.exp(-z))

log_gain = np.log(gain) + eta * (target_rate - mean_rate)
new_gain = float(np.exp(np.clip(log_gain, np.log(gain_min), np.log(gain_max))))
```

- [ ] **Step 4: Run primitive tests GREEN**

Run: `pytest -q tests/test_output_boundary.py`
Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add AIS-like output boundary primitives`.

---

### Task 2: Reuse Gate-6 dendritic training without changing Gate 6

**Files:**
- Modify: `experiments/gate6_many_worlds.py`
- Modify: `tests/test_gate6.py`

**Interfaces:**
- Consumes: existing `_make_world_tapes`, `_initial_q_target`, `_train_conditions`.
- Produces: `trained_world_conductances(world: CableWorld, *, tape_seed: int) -> dict[str, np.ndarray]`, a public helper that reproduces Gate-6 condition conductances without evaluation.

- [ ] **Step 1: Pin the existing Gate-6 receipt and helper result before refactor**

Add a test that loads `results/gate6.json` and verifies its gate, world count, alignment metric string, and aggregate numbers exactly as currently committed. Add a two-world helper test requiring the returned keys to be `frozen`, `local`, `shuffled`, `uniform`, with local conductance count matching edge count and uniform byte-identical to frozen.

- [ ] **Step 2: Run tests RED because helper is absent**

Run: `pytest -q tests/test_gate6.py`
Expected: fail importing or calling `trained_world_conductances`.

- [ ] **Step 3: Factor the helper minimally**

Implement:

```python
def trained_world_conductances(world: CableWorld, *, tape_seed: int) -> dict[str, np.ndarray]:
    adaptation_tapes, _ = _make_world_tapes(world, tape_seed)
    q_target = _initial_q_target(world, adaptation_tapes[0])
    trained, _ = _train_conditions(world, adaptation_tapes, q_target=q_target, seed=tape_seed)
    return {name: values.copy() for name, values in trained.items()}
```

Refactor `run_world` to call the helper only if doing so preserves all existing Gate-6 receipt values. If the helper would require recomputing multiset-error metadata separately, keep `run_world` unchanged and let Gate 7 call the helper independently; preserving Gate 6 is higher priority than DRY.

- [ ] **Step 4: Verify Gate-6 regression GREEN**

Run: `pytest -q tests/test_gate6.py`
Then run: `python experiments/gate6_many_worlds.py --worlds 24 --out /tmp/gate6.json && cmp /tmp/gate6.json results/gate6.json`
Expected: tests pass and `cmp` exits 0.

- [ ] **Step 5: Commit**

Commit message: `refactor: expose frozen Gate 6 conductance training`.

---

### Task 3: Gate-7 single-world protocol and invariants

**Files:**
- Create: `experiments/gate7_load_compensation.py`
- Create: `tests/test_gate7.py`

**Interfaces:**
- Consumes: `generate_world`, `port_sensor_matrices`, `trained_world_conductances`, cable constants `_DT/_LEAK/_COUPLING`, `simulate_cable`, and output-boundary primitives.
- Produces:
  - `_make_gate7_tapes(world: CableWorld) -> tuple[np.ndarray, np.ndarray]`
  - `_soma_voltage(world: CableWorld, conductances: np.ndarray, tape: np.ndarray, scale: float = 1.0) -> np.ndarray`
  - `_reference_parameters(reference_world: CableWorld, reference_g: np.ndarray) -> dict`
  - `_adapt_homeostatic_gain(calibration_voltage: np.ndarray, *, theta: float, beta: float, target_rate: float) -> float`
  - `run_world(world: CableWorld, *, reference: dict) -> dict`

- [ ] **Step 1: Write two-world failing tests**

Tests must verify deterministic calibration/evaluation tape digests; all three conditions use the same held-out base tape per world; fixed and homeostatic gains both start from `1.0`; only the homeostat's rate value flows into `homeostatic_gain`; oracle gain equals load/reference-load exactly; condition rates are finite in `[0,1]`; five scales are exactly `[0.5,0.75,1.0,1.25,1.5]`; and no assertion requires homeostasis to improve the scientific metric.

- [ ] **Step 2: Run Gate-7 tests RED**

Run: `pytest -q tests/test_gate7.py`
Expected: collection fails because `experiments.gate7_load_compensation` does not yet exist.

- [ ] **Step 3: Implement deterministic tapes and soma-voltage extraction**

Generate calibration/evaluation inputs with `np.random.default_rng(np.random.SeedSequence([...]))`. Use each world's held-out covariance family at its first held-out angle for Gate-7 tapes so all four input channels are driven by the same established world statistics. Build the cable operator with the frozen local-adapted conductances and return node-0 state after a 250-sample burn. Do not use the sensor matrix for the AIS assay; request full states from `simulate_cable` and select column `0`.

- [ ] **Step 4: Implement reference calibration and three conditions**

World `0` defines `beta`, `theta`, target rate, reference load, and the fixed-boundary reference transfer curve. For each world compute fixed `a=1`, 12-phase local homeostatic gain, and unclipped oracle load-ratio gain. Evaluate all three on the same held-out base tape multiplied by the five frozen scales.

- [ ] **Step 5: Record per-world outputs**

Each world result must contain: topology/world IDs; load; tape digests; reference load; fixed/homeostatic/oracle gains; gain-start values; five mean-rate arrays; curve RMSE; rheobase analogue; F-I gain; unit-scale rate; collapse flag; and per-world deltas `fixed_rmse - homeostatic_rmse` and `fixed_rmse - oracle_rmse`.

- [ ] **Step 6: Run focused tests GREEN**

Run: `pytest -q tests/test_output_boundary.py tests/test_gate6.py tests/test_gate7.py`
Expected: all pass.

- [ ] **Step 7: Commit**

Commit message: `feat: add Gate 7 load compensation protocol`.

---

### Task 4: Gate-7 24-world aggregate and frozen receipt

**Files:**
- Modify: `experiments/gate7_load_compensation.py`
- Modify: `tests/test_gate7.py`
- Create: `results/gate7.json`

**Interfaces:**
- Consumes: Task-3 `run_world` and reference calibration.
- Produces: `run(seed: int = 17, n_worlds: int = 24) -> dict` plus CLI `--seed`, `--worlds`, `--out`.

- [ ] **Step 1: Write aggregate tests**

Require `1 <= n_worlds <= 24`; deterministic two-world equality; reference world excluded from delta summaries but retained in `worlds`; aggregate keys for mean/median/q25/min/win-fraction of both compensation deltas; condition mean/median/q75 RMSE; unit-rate/F-I/rheobase dispersion; collapse counts; descriptive Pearson correlations for load-vs-fixed-error and learned-gain-vs-oracle-gain; and no positive-delta assertion.

- [ ] **Step 2: Run aggregate tests RED**

Run: `pytest -q tests/test_gate7.py`
Expected: fail because `run()`/aggregate keys are missing.

- [ ] **Step 3: Implement aggregation and CLI**

Use only finite rheobases for rheobase standard deviation and separately report missing count. Pearson correlation returns `None` if either vector has zero variance. Include an interpretation string that explicitly calls the experiment synthetic and notes that local output homeostasis cannot read dendritic load.

- [ ] **Step 4: Run two-world Gate-7 GREEN**

Run: `pytest -q tests/test_gate7.py`
Expected: all tests pass.

- [ ] **Step 5: Run the predeclared 24-world receipt once**

Run: `python experiments/gate7_load_compensation.py --seed 17 --worlds 24 --out results/gate7.json`
Do not change the frozen constants after inspecting this result. Record the result exactly, including failures/collapsed worlds.

- [ ] **Step 6: Add receipt-identity regression**

Add a test loading `results/gate7.json` and requiring `run(seed=17, n_worlds=24) == json.load(...)` using deterministic JSON-compatible values. If bit-level floating serialization differs across supported NumPy/Python versions, run the identity check only on Python 3.11 and keep cross-version tests to finite/invariant tolerances; do not alter scientific values to force cross-platform bit equality.

- [ ] **Step 7: Commit**

Commit message: `science: freeze Gate 7 load compensation receipt`.

---

### Task 5: CI, paper, README, and visual companion update

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `PAPER.md`
- Modify: `index.html`
- Modify: `tests/test_index_html.py`

**Interfaces:**
- Consumes: frozen `results/gate7.json`.
- Produces: final public interpretation and verification workflow.

- [ ] **Step 1: Add failing page/docs expectations**

Extend the static-page test to require `Gate 7`, `load compensation`, and explicit wording that the AIS surrogate is synthetic; retain the existing requirement that biological AIS claims are not presented as validation.

- [ ] **Step 2: Update CI**

Run full engineering tests on Python 3.11 and 3.12. Add Gate-7 two-world scientific smoke to both versions. Add full 24-world Gate-7 verification only on Python 3.11, alongside the existing full Gate-6 verification. The full run compares generated Gate-7 JSON to the committed receipt or uses the Python-3.11 receipt-identity test.

- [ ] **Step 3: Update public interpretation from the actual receipt**

README/PAPER/index must report the measured fixed/homeostatic/oracle outcomes exactly, including negative or null results. State the distinction between biological anchors and synthetic mechanism. Add the Aizenbud/Leterrier/Hay motivation only to the extent already justified by the design; do not claim biological validation.

- [ ] **Step 4: Preserve the two follow-up hypotheses without implementing them**

Document succinctly that the next independent experiments are: (a) the 24x24 shape-as-signal fingerprint matrix with spectral-gap analysis, and (b) a temporal/path-length matching gate inspired by recursive filtering, gamma-phase arrival efficacy, and predominantly time-yoked auditory integration windows.

- [ ] **Step 5: Run fresh full verification**

Run: `pytest -q`
Run: Gates 0-5B smoke commands already present in CI.
Run: `python experiments/gate6_many_worlds.py --worlds 24 --out /tmp/gate6.json && cmp /tmp/gate6.json results/gate6.json`
Run: `python experiments/gate7_load_compensation.py --worlds 24 --out /tmp/gate7.json && cmp /tmp/gate7.json results/gate7.json`
Expected: all engineering tests pass and both frozen receipts reproduce exactly on the reference Python runner.

- [ ] **Step 6: Open PR and verify exact head**

Create a PR from `study/gate7-load-compensation` to `main`, mark ready, and wait for the exact-head Python 3.11/3.12 checks. Do not merge on an older green parent.

- [ ] **Step 7: Commit**

Commit message: `docs: report Gate 7 load compensation result`.
