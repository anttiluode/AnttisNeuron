# Gate 7 Load Compensation Implementation Plan

> **For agentic workers:** use `superpowers:executing-plans` or subagent-driven development, with TDD and explicit RED/GREEN checkpoints.

**Goal:** Test whether a purely local AIS-like rate homeostat can preserve a nontrivial somatic-current→output transfer function across the 24 existing frozen dendritic worlds better than a fixed output boundary.

**Corrected architecture:** Gate 7 now probes **somatic driving-point load directly**. Reproduce each world's Gate-6 local-adapted conductances, freeze them, inject deterministic steady somatic current at node `0`, compute `v_soma = I/g_load`, and compare fixed/local/oracle output-boundary gains. The load-ratio oracle is therefore exactly matched to the measured transfer problem. The 24-world scientific receipt has not yet run.

**Tech stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-gate7-load-compensation-design.md`

## Frozen protocol

- Reuse the 24 deterministic Gate-6 `CableWorld` instances and exact Gate-6 local dendritic training.
- Do not modify `results/gate6.json` or Gate-6 numerical behavior.
- Soma/output interface: node `0`.
- Continuous conductance matrix: `G = leak*I + coupling*L(g)`.
- Driving-point load: Schur complement at node `0`.
- Somatic steady voltage: `v(I) = I/g_load`.
- Reference world: world `0`.
- Reference target rate: `r*=0.25`.
- Reference unit-current voltage: `v*=1/g_load_ref`.
- Output slope: `beta = 4/v*`.
- Threshold: `theta = v* - logit(0.25)/beta`.
- Fixed gain: `a=1`.
- Homeostat: 12 phases at `I=1`, `log(a) += 0.5*(0.25-r)`, clipped to `[0.2,5.0]`.
- Homeostat inputs: current gain, current output rate, target rate only.
- Oracle gain: `a_oracle = g_load(world)/g_load(reference)`, unclipped.
- Evaluation currents: exactly `(0.50,0.75,1.00,1.25,1.50)`.
- Rheobase analogue threshold: `0.10`.
- F-I gain: least-squares slope through `(0.75,1.00,1.25)`.
- Reference curve dynamic range must be at least `0.10`.
- Unit-current rate `<0.02` or `>0.98` is flagged collapsed and retained.
- No positive local-homeostasis result is encoded as a correctness test.

---

## Task 1 — Preserve completed primitive work and add steady-state voltage invariant

**Existing completed work:** `anttis_neuron/output_boundary.py` and `tests/test_output_boundary.py` already have RED→GREEN coverage for Schur-complement load, bounded logistic rate, threshold calibration, local gain update, and transfer metrics.

**Files:**
- Modify: `anttis_neuron/output_boundary.py`
- Modify: `tests/test_output_boundary.py`

- [ ] Add a failing test for `steady_state_soma_voltage(current, load)` (or equivalent exact helper): positive finite load required; scalar/vector current supported; result equals `current/load` exactly within floating tolerance.
- [ ] Add a two-node/direct-solve test showing the Schur-complement load gives the same soma voltage as `np.linalg.solve(G, I*e0)[0]`.
- [ ] Verify RED.
- [ ] Implement the minimal helper/validation.
- [ ] Verify focused GREEN.
- [ ] Commit: `fix: align Gate 7 primitives with somatic load assay`.

---

## Task 2 — Preserve completed Gate-6 helper work

`trained_world_conductances(world, tape_seed=...)` is already implemented by TDD. Do not route existing Gate-6 `run_world()` through it. Keep the frozen receipt identity test and verify full Gate 6 remains unchanged before final integration.

---

## Task 3 — Replace the rejected distal-tape Gate-7 protocol

**Files:**
- Modify: `experiments/gate7_load_compensation.py`
- Replace/update: `tests/test_gate7.py`
- Delete after incorporating evidence: `tests/test_gate7_diagnostic.py`

- [ ] Rewrite Gate-7 tests first so they no longer reference random tapes, cable burn-in, distal ports, or stochastic voltage calibration.
- [ ] New tests require:
  - reference analytic calibration gives exactly rate `0.25` at unit current;
  - reference five-current curve has dynamic range `>=0.10`;
  - per-world load is positive finite;
  - direct-solve and Schur-complement voltage agree;
  - fixed and local gains both start at `1.0`;
  - oracle gain equals exact load ratio;
  - oracle five-point curve equals reference five-point curve to numerical precision;
  - homeostat function signature exposes no load/topology/conductance/spectral inputs;
  - all condition rates are finite and in `[0,1]`;
  - no assertion requires local homeostasis to improve the science metric.
- [ ] Verify RED against the rejected distal-tape implementation.
- [ ] Replace the scientific protocol with deterministic somatic-current evaluation.
- [ ] Reference calibration uses only world `0` load and the analytic formulas in the spec.
- [ ] Local homeostasis runs exactly 12 unit-current phases.
- [ ] Per-world receipt includes world IDs/topology, load/reference load/load ratio, fixed/local/oracle gains, five rate curves, transfer metrics, collapse flags, `fixed_rmse-homeostatic_rmse`, and oracle numerical error.
- [ ] Delete the temporary diagnostic test once corrected tests encode the discovered root cause.
- [ ] Verify focused GREEN.
- [ ] Commit: `fix: probe somatic load directly in Gate 7`.

---

## Task 4 — 24-world aggregate and frozen scientific receipt

**Files:**
- Modify: `experiments/gate7_load_compensation.py`
- Modify: `tests/test_gate7.py`
- Create: `results/gate7.json`

- [ ] Add aggregate tests first.
- [ ] Require `1 <= n_worlds <= 24`, deterministic two-world equality, world `0` retained but excluded from local-delta summary, local delta mean/median/q25/min/win fraction, per-condition RMSE and operating-rate summaries, collapse counts, finite-rheobase dispersion/missing count, F-I dispersion, and descriptive load-ratio↔learned-gain correlation.
- [ ] Require oracle RMSE max to stay within numerical tolerance; this is an engineering identity control, not a scientific-positive assertion.
- [ ] Verify RED.
- [ ] Implement aggregate and CLI.
- [ ] Verify two-world GREEN.
- [ ] Run the predeclared 24-world assay **once**: `python experiments/gate7_load_compensation.py --seed 17 --worlds 24 --out results/gate7.json`.
- [ ] After the 24-world result exists, do not alter frozen scientific constants to improve the outcome.
- [ ] Add Python-3.11 receipt identity regression.
- [ ] Commit: `science: freeze Gate 7 somatic load receipt`.

---

## Task 5 — CI and public interpretation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `PAPER.md`
- Modify: `index.html`
- Modify: `tests/test_index_html.py`

- [ ] Add/extend failing page expectations for `Gate 7`, `load compensation`, `somatic current`, and explicit wording that the AIS/output-boundary model is synthetic.
- [ ] Add Gate-7 two-world smoke to Python 3.11/3.12 CI.
- [ ] Add full 24-world Gate-7 receipt verification only on Python 3.11, alongside the existing Gate-6 verification.
- [ ] Update README/PAPER/index from the actual frozen result, whether positive, null, or negative.
- [ ] Explain the pre-assay correction plainly: the first draft mixed distal transfer with a somatic-load oracle; no 24-world Gate-7 result existed before correction.
- [ ] Keep biological claims as motivation/constraints only.
- [ ] Preserve the two independent follow-ups: 24×24 shape-as-signal fingerprint matrix and temporal/path-length matching.
- [ ] Commit: `docs: report Gate 7 somatic load result`.

---

## Task 6 — Fresh verification, review, PR, integration

- [ ] Run full `pytest -q` on the exact final head.
- [ ] Run existing Gates 0–5B smoke commands.
- [ ] Regenerate Gate 6 on Python 3.11 and compare byte-for-byte with `results/gate6.json`.
- [ ] Regenerate Gate 7 on Python 3.11 and compare byte-for-byte with `results/gate7.json`.
- [ ] Use `superpowers:verification-before-completion` before any completion claim.
- [ ] Use `superpowers:requesting-code-review` / inspect PR patch for scientific leakage or overclaiming.
- [ ] Open PR from `study/gate7-load-compensation` to `main`.
- [ ] Wait for exact-head Python 3.11/3.12 checks; do not merge on an older green parent.
- [ ] Use `superpowers:finishing-a-development-branch` for integration. User has already authorized continuing repo work, but exact-head green verification remains mandatory.
