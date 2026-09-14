# Gate 7 Load Compensation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a falsifiable Gate 7 experiment testing whether a purely local AIS-like firing-rate homeostat can preserve a nontrivial input-output transfer function across the 24 existing dendritic worlds better than a fixed boundary, with a closed-form load-ratio oracle as a diagnostic reference.

**Architecture:** Keep Gate 6 numerically unchanged and reuse its deterministic world generation plus local dendritic adaptation. Add `anttis_neuron/output_boundary.py` for electrical-load and output-boundary primitives, then `experiments/gate7_load_compensation.py` for the fixed/homeostatic/oracle assay. Gate 7 probes every frozen dendritic world with the same input distribution so differences are not confounded by different covariance families.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-gate7-load-compensation-design.md`

## Global Constraints

- Reuse the 24 deterministic Gate-6 `CableWorld` instances; do not change `results/gate6.json` or Gate-6 numerical behavior.
- Reproduce the Gate-6 local dendritic adaptation, then freeze dendritic conductances before output-boundary adaptation.
- Treat node `0` as the soma/output-boundary interface.
- Gate-7 calibration seed: `SeedSequence([7017, world.seed, world.index])`; evaluation seed: `SeedSequence([8017, world.seed, world.index])`.
- Calibration tape length: `3000`; evaluation base tape length: `5000`; cable burn-in: `250`.
- Every Gate-7 tape is sampled from the same `N(0, I_4)` distribution; only the deterministic sample realization changes by world seed. Gate 7 therefore tests the frozen physical substrates under matched probe statistics rather than re-testing the Gate-6 covariance families.
- Reference world: world `0`; reference target rate: `0.25`; `beta = 2 / soma_voltage_std`; calibrate `theta` by deterministic bisection at unit input scale.
- Fixed boundary gain: `a = 1`.
- Homeostat: 12 phases, `log(a) += 0.5 * (0.25 - mean_rate)`, clipped to `a in [0.2, 5.0]`; its function interface contains only mean rate, target rate, and current gain.
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
  - `driving_point_conductance(n_nodes, edges, conductances, *, leak, coupling, soma_node=0) -> float`
  - `rate_boundary(voltage, *, gain, theta, beta) -> np.ndarray`
  - `calibrate_threshold(voltage, *, gain, beta, target_rate, iterations=80) -> float`
  - `homeostatic_gain(gain, mean_rate, *, target_rate=0.25, eta=0.5, gain_min=0.2, gain_max=5.0) -> float`
  - `transfer_metrics(scales, rates, reference_rates, *, rheobase_rate=0.10) -> dict`

- [ ] **Step 1: Write failing primitive tests**

Create `tests/test_output_boundary.py` with a two-node graph Schur-complement test, bounded finite logistic-rate test, deterministic threshold-calibration test, homeostatic direction/bounds tests, and transfer-metric tests.

```python
import math
import numpy as np
from anttis_neuron.output_boundary import (
    calibrate_threshold, driving_point_conductance,
    homeostatic_gain, rate_boundary, transfer_metrics,
)


def test_two_node_load_matches_schur_complement():
    edges = np.array([[0, 1]], dtype=int)
    g = np.array([2.0])
    got = driving_point_conductance(2, edges, g, leak=0.08, coupling=0.65)
    diag = 0.08 + 0.65 * 2.0
    off = -0.65 * 2.0
    expected = diag - off * off / diag
    assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
    assert got > 0.0


def test_threshold_calibration_hits_target_rate():
    voltage = np.linspace(-1.0, 1.0, 2001)
    theta = calibrate_threshold(voltage, gain=1.0, beta=2.0, target_rate=0.25)
    rate = rate_boundary(voltage, gain=1.0, theta=theta, beta=2.0)
    assert abs(float(np.mean(rate)) - 0.25) < 1e-10
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_output_boundary.py`
Expected: collection fails because `anttis_neuron.output_boundary` does not exist.

- [ ] **Step 3: Implement primitives minimally**

Construct `G = leak*I + coupling*weighted_laplacian(...)`. Compute the Schur complement with `np.linalg.solve`, never an explicit inverse. Use a clipped logistic exponent, 80-step deterministic bisection, and log-space gain clipping.

```python
rest = np.array([i for i in range(n_nodes) if i != soma_node], dtype=int)
load = float(G[soma_node, soma_node]) - float(
    G[soma_node, rest] @ np.linalg.solve(G[np.ix_(rest, rest)], G[rest, soma_node])
)
z = np.clip(beta * (gain * voltage - theta), -60.0, 60.0)
rates = 1.0 / (1.0 + np.exp(-z))
log_gain = np.log(gain) + eta * (target_rate - mean_rate)
new_gain = float(np.exp(np.clip(log_gain, np.log(gain_min), np.log(gain_max))))
```

`transfer_metrics` returns curve RMSE, rheobase (linear interpolation, otherwise `None`), central three-point least-squares F-I slope, and unit-scale rate.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_output_boundary.py`
Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit: `feat: add AIS-like output boundary primitives`.

---

### Task 2: Expose Gate-6 trained conductances without changing Gate 6

**Files:**
- Modify: `experiments/gate6_many_worlds.py`
- Modify: `tests/test_gate6.py`

**Interfaces:**
- Produces: `trained_world_conductances(world: CableWorld, *, tape_seed: int) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Add failing regression/helper tests**

Pin the committed Gate-6 receipt metadata/aggregate and import the not-yet-existing helper. Require keys `frozen/local/shuffled/uniform`, one conductance per edge, and uniform byte-identical to frozen.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_gate6.py`
Expected: helper import/call fails.

- [ ] **Step 3: Implement helper only**

```python
def trained_world_conductances(world: CableWorld, *, tape_seed: int) -> dict[str, np.ndarray]:
    adaptation_tapes, _ = _make_world_tapes(world, tape_seed)
    q_target = _initial_q_target(world, adaptation_tapes[0])
    trained, _ = _train_conditions(world, adaptation_tapes, q_target=q_target, seed=tape_seed)
    return {name: values.copy() for name, values in trained.items()}
```

Do not refactor `run_world` unless the full Gate-6 receipt remains byte-identical.

- [ ] **Step 4: Verify Gate 6 is unchanged**

Run: `pytest -q tests/test_gate6.py`
Run: `python experiments/gate6_many_worlds.py --worlds 24 --out /tmp/gate6.json && cmp /tmp/gate6.json results/gate6.json`
Expected: tests pass and `cmp` exits 0.

- [ ] **Step 5: Commit**

Commit: `refactor: expose frozen Gate 6 conductance training`.

---

### Task 3: Gate-7 single-world protocol

**Files:**
- Create: `experiments/gate7_load_compensation.py`
- Create: `tests/test_gate7.py`

**Interfaces:**
- Produces:
  - `_make_gate7_tapes(world) -> tuple[np.ndarray, np.ndarray]`
  - `_soma_voltage(world, conductances, tape, scale=1.0) -> np.ndarray`
  - `_reference_parameters(reference_world, reference_g) -> dict`
  - `_adapt_homeostatic_gain(calibration_voltage, *, theta, beta, target_rate) -> float`
  - `run_world(world, *, reference) -> dict`

- [ ] **Step 1: Write failing two-world tests**

Require deterministic calibration/evaluation tape digests, `N(0,I_4)` sample shape/finite values, identical held-out base tape across fixed/homeostatic/oracle conditions within a world, fixed and local gains starting at `1.0`, exact oracle load ratio, five frozen scales, bounded finite rates, and no positive-outcome assertion. Use `inspect.signature(homeostatic_gain)` to ensure there is no load/topology/conductance/spectral argument.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_gate7.py`
Expected: collection fails because `experiments.gate7_load_compensation` does not exist.

- [ ] **Step 3: Implement matched-distribution tapes and soma extraction**

```python
cal_rng = np.random.default_rng(np.random.SeedSequence([7017, world.seed, world.index]))
eval_rng = np.random.default_rng(np.random.SeedSequence([8017, world.seed, world.index]))
calibration = cal_rng.standard_normal((3000, 4))
evaluation = eval_rng.standard_normal((5000, 4))
```

Build the cable operator from the frozen local Gate-6 conductances. Use `port_sensor_matrices(world)` for `B`; call `simulate_cable(..., return_states=True, burn=250)` and return full-state column `0`. The observation matrix exists only to satisfy the existing simulator interface and is not used by the output boundary.

- [ ] **Step 4: Implement reference and three conditions**

World 0 calibrates `beta`, `theta`, target rate, reference load, and reference fixed-boundary transfer curve. Each world evaluates fixed gain `1.0`, 12-phase local homeostasis, and unclipped oracle load ratio on the same held-out base tape at scales `(0.5,0.75,1.0,1.25,1.5)`.

- [ ] **Step 5: Record complete per-world receipt fields**

Include world/seed/topology identifiers, load/reference load, calibration/evaluation tape digests, fixed/local starting gains, final three gains, three five-rate curves, three metric dictionaries, collapse flags, and `fixed_rmse - homeostatic_rmse` / `fixed_rmse - oracle_rmse`.

- [ ] **Step 6: Verify GREEN**

Run: `pytest -q tests/test_output_boundary.py tests/test_gate6.py tests/test_gate7.py`
Expected: all pass.

- [ ] **Step 7: Commit**

Commit: `feat: add Gate 7 load compensation protocol`.

---

### Task 4: 24-world aggregate and frozen receipt

**Files:**
- Modify: `experiments/gate7_load_compensation.py`
- Modify: `tests/test_gate7.py`
- Create: `results/gate7.json`

**Interfaces:**
- Produces: `run(seed: int = 17, n_worlds: int = 24) -> dict` and CLI `--seed/--worlds/--out`.

- [ ] **Step 1: Write failing aggregate tests**

Require `1 <= n_worlds <= 24`, deterministic two-world equality, reference world retained but excluded from delta summaries, distribution summaries for both deltas, per-condition RMSE/rate/F-I/rheobase dispersion and collapse counts, and descriptive load-error / learned-oracle correlations.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_gate7.py`
Expected: aggregate interface/keys missing.

- [ ] **Step 3: Implement aggregate/CLI**

Finite rheobases contribute to rheobase SD; missing count is reported separately. Pearson correlation is `None` for zero-variance inputs. Interpretation explicitly says synthetic and states that the homeostat cannot read load.

- [ ] **Step 4: Verify two-world GREEN**

Run: `pytest -q tests/test_gate7.py`
Expected: all pass.

- [ ] **Step 5: Run the predeclared 24-world assay once**

Run: `python experiments/gate7_load_compensation.py --seed 17 --worlds 24 --out results/gate7.json`
After this command, do not alter frozen scientific constants in response to the result.

- [ ] **Step 6: Add receipt identity regression**

On the reference Python 3.11 runner require regenerated 24-world JSON to equal the committed receipt. Cross-version engineering tests use invariant/tolerance checks rather than changing scientific values for bit identity.

- [ ] **Step 7: Commit**

Commit: `science: freeze Gate 7 load compensation receipt`.

---

### Task 5: CI and public interpretation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `PAPER.md`
- Modify: `index.html`
- Modify: `tests/test_index_html.py`

- [ ] **Step 1: Write failing page expectations**

Require `Gate 7`, `load compensation`, and wording that the AIS/output-boundary model is a synthetic surrogate, while retaining the previous anti-overclaiming checks.

- [ ] **Step 2: Update CI**

Run engineering tests and Gate-7 two-world smoke on Python 3.11/3.12. Run full 24-world Gate-7 receipt verification only on Python 3.11, alongside existing full Gate 6.

- [ ] **Step 3: Update README/PAPER/index from the frozen result**

Report fixed/homeostatic/oracle outcomes exactly, whether positive, null, or negative. Biological papers are motivation/constraints only. Preserve the next independent hypotheses: (a) 24x24 shape-as-signal fingerprint matrix plus spectral-gap analysis, and (b) temporal/path-length matching inspired by recursive filtering, gamma-phase arrival efficacy, and predominantly time-yoked auditory integration.

- [ ] **Step 4: Commit documentation/CI**

Commit: `docs: report Gate 7 load compensation result`.

- [ ] **Step 5: Run fresh full verification**

Run: `pytest -q`
Run the existing Gates 0-5B smoke commands.
Run: `python experiments/gate6_many_worlds.py --worlds 24 --out /tmp/gate6.json && cmp /tmp/gate6.json results/gate6.json`
Run: `python experiments/gate7_load_compensation.py --worlds 24 --out /tmp/gate7.json && cmp /tmp/gate7.json results/gate7.json`
Expected: all engineering tests pass and both reference receipts reproduce exactly on Python 3.11.

- [ ] **Step 6: Open PR and verify exact head**

Create a PR from `study/gate7-load-compensation` to `main`, mark it ready, and wait for the exact-head Python 3.11/3.12 checks. Do not merge on an older green parent.
