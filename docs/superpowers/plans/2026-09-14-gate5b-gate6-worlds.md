# Gate 5B, Gate 6, and Visual Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the mechanism behind Gate 5, test the frozen local rule across 24 deterministic branched worlds, and publish a self-contained static explainer grounded in frozen receipts.

**Architecture:** Gate 5 remains immutable. `experiments/gate5b_spectral_audit.py` imports Gate 5 helpers and adds decomposition/counterfactual diagnostics. `anttis_neuron/worlds.py` owns reusable deterministic world generation and bounded port/sensor construction; `experiments/gate6_many_worlds.py` owns control training/evaluation and aggregates 24 worlds. `index.html` is a standalone static explanation populated only after Gate 5B/6 receipts are frozen.

**Tech Stack:** Python 3.11+, NumPy, pytest, GitHub Actions, standalone HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-14-gate5b-gate6-worlds-design.md`

## Global Constraints

- Gate 5 code and `results/gate5.json` are not modified.
- NumPy + pytest only for scientific code; no heavy dependency.
- Gate 6 uses Gate 5 physical/adaptation/Oja constants unchanged.
- Scientific sign/delta is never a CI pass criterion.
- Every matched control shares external tapes and Oja initialization.
- Shuffled credit uses the exact local per-phase update-signal multiset.
- Default Gate 6 receipt uses exactly 24 deterministic worlds.
- `index.html` has no external CSS or JavaScript dependency and no invented video URL.

---

### Task 1: Gate 5B spectral decomposition primitives and diagnostic operators

**Files:**
- Create: `experiments/gate5b_spectral_audit.py`
- Create: `tests/test_gate5b.py`

**Interfaces:**
- `sorted_nonuniform_eigendecomposition(a) -> tuple[np.ndarray, np.ndarray]`: eigenvalues/eigenvectors ordered slowest-first after excluding the most uniform state-space eigenvector.
- `principal_angles(q_left, q_right, rank=3) -> np.ndarray`: returns principal angles in radians for the leading subspaces.
- `hybrid_operators(a_frozen, a_adapted) -> tuple[np.ndarray, np.ndarray]`: returns `(A_lambda, A_Q)` where `A_lambda` uses frozen eigenvectors + adapted sorted eigenvalues and `A_Q` uses adapted eigenvectors + frozen sorted eigenvalues, with the uniform mode included consistently.
- `evaluate_operator(a, tapes, b, s, *, oja_seed_base) -> dict`: Gate-5-compatible held-out evaluation plus learned Oja vectors.

- [ ] **Step 1: Write failing spectral primitive tests**

Create tests that assert:

```python
def test_hybrid_operator_reconstruction_parts():
    result = build_gate5_operators(seed=17)
    a_f, a_a = result["frozen"], result["adapted"]
    a_lambda, a_q = hybrid_operators(a_f, a_a)
    vf = np.linalg.eigvalsh(a_f)
    va = np.linalg.eigvalsh(a_a)
    assert np.allclose(np.sort(np.linalg.eigvalsh(a_lambda)), np.sort(va), atol=1e-12)
    assert np.allclose(np.sort(np.linalg.eigvalsh(a_q)), np.sort(vf), atol=1e-12)
    assert np.allclose(a_lambda, a_lambda.T, atol=1e-12)
    assert np.allclose(a_q, a_q.T, atol=1e-12)


def test_principal_angles_identity_are_zero():
    q = np.eye(5)[:, :3]
    assert np.allclose(principal_angles(q, q, rank=3), 0.0, atol=1e-12)
```

Also assert all four diagnostic operators have spectral radius `< 1.0`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/test_gate5b.py -v`

Expected: import failure because `experiments.gate5b_spectral_audit` does not exist.

- [ ] **Step 3: Implement minimal decomposition code**

Use `np.linalg.eigh` for symmetric operators. Identify the uniform eigenvector by maximum absolute cosine with `ones/sqrt(n)`. For full-basis hybrid construction, align sorted eigenpairs by descending eigenvalue (equivalently slowest discrete mode first for the positive stable cable operator); keep the uniform eigenpair in the full decomposition. Construct hybrids by matrix multiplication and symmetrize with `(A + A.T)/2` only to remove floating-point asymmetry.

For principal angles, QR-orthonormalize the supplied subspaces, compute singular values of `Q1.T @ Q2`, clip to `[0,1]`, and return `arccos(s)`.

- [ ] **Step 4: Run focused + full suite**

Run: `pytest tests/test_gate5b.py -v && pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add Gate 5B spectral audit primitives`

---

### Task 2: Gate 5B frozen experiment and receipt

**Files:**
- Modify: `experiments/gate5b_spectral_audit.py`
- Modify: `tests/test_gate5b.py`
- Modify: `tests/test_experiments.py`
- Create after execution: `results/gate5b.json`

**Interfaces:**
- `run(seed: int = 17) -> dict`
- Output keys include:
  - `gate: "5B"`
  - `seed`
  - frozen/adapted slow non-uniform eigenvalues
  - `slow_subspace_principal_angles_degrees`
  - `slow_mode_best_match_overlaps`
  - `visible_mode_overlap_matrix`
  - held-out summaries for frozen/eigenvalue-only/eigenvector-only/adapted
  - cross-basis four-way mean alignments
  - no predeclared categorical mechanism label.

- [ ] **Step 1: Add failing Gate 5B receipt invariants**

Tests must require deterministic finite output, valid cosine/alignment bounds, exactly five per-tape alignments per diagnostic condition, three principal angles, symmetric stable hybrids, and no assertion that adapted exceeds any counterfactual.

Add a regression check that `run5(seed=17)` still exactly matches committed Gate 5 headline values:

```python
assert run5(seed=17)["adaptive_heldout_mean_alignment"] == 0.9001914311455128
assert run5(seed=17)["frozen_heldout_mean_alignment"] == 0.7821845260371927
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_gate5b.py tests/test_experiments.py -v`

Expected: failures for missing `run()`/receipt fields only; Gate 5 regression remains green.

- [ ] **Step 3: Implement Gate 5B run**

Use Gate 5 `_make_input_tapes`, `_initial_q_target`, `_train_local_with_signals`, `_operator`, constants, sensors/ports, and common Oja seeds. Do not alter Gate 5.

For cross-basis scoring, retain frozen/adapted Oja vectors for each held-out tape and score each vector against the three slowest visible modes from frozen/adapted operators.

Report both radians and degrees only if helpful; the committed receipt must at minimum include degrees.

- [ ] **Step 4: Run Gate 5B and full suite**

Run:

```bash
pytest tests/test_gate5b.py tests/test_experiments.py -v
python experiments/gate5b_spectral_audit.py --out /tmp/gate5b.json
pytest -q
```

Scientific values may have any sign/order.

- [ ] **Step 5: Freeze exact runner receipt**

After CI prints the deterministic JSON, create `results/gate5b.json` with exactly that output. Do not rerun with adjusted parameters because of an unattractive result.

- [ ] **Step 6: Commit**

Commit message: `exp: freeze Gate 5B spectral autopsy`

---

### Task 3: Deterministic many-world generator

**Files:**
- Create: `anttis_neuron/worlds.py`
- Create: `tests/test_worlds.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CableWorld:
    index: int
    seed: int
    n_nodes: int
    edges: np.ndarray
    port_nodes: tuple[int, int, int, int]
    sensor_nodes: tuple[int, int, int, int]
    adaptation_angles: tuple[float, float, float, float]
    heldout_angles: tuple[float, float, float, float, float]
    adaptation_spectra: tuple[tuple[float, float, float, float], ...]
    heldout_spectrum: tuple[float, float, float, float]


def generate_world(index: int, master_seed: int = 1701, *, max_attempts: int = 64) -> CableWorld

def port_sensor_matrices(world: CableWorld) -> tuple[np.ndarray, np.ndarray]
```

- [ ] **Step 1: Write failing world-generation tests**

Require same `(index, master_seed)` to reproduce byte-identical scalar/list fields; different indices differ; graph has 11 nodes/10 edges and is connected; ports/sensors are four distinct valid non-root nodes; default operator is stable; at least three non-uniform visible modes exist.

Test 24 generated worlds without asserting scientific behavior.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_worlds.py -v`

Expected: import failure for `anttis_neuron.worlds`.

- [ ] **Step 3: Implement generator**

Seed each attempt with a deterministic `SeedSequence([master_seed, index, attempt])`. Build the random-parent tree. Compute leaves from node degree; choose ports from leaves if at least four, otherwise from non-root nodes. Choose sensors independently without replacement from nodes `1..10`.

Generate adaptation angles by perturbing `(12,31,53,74)` with world-specific bounded jitter `[-12,+12]` degrees; generate held-out angles from `(22,42,63,83,103)` with bounded jitter `[-12,+12]`, then if any held-out angle is within `1e-6` of an adaptation angle add `0.5` degrees. Jitter Gate-5 spectral values multiplicatively by factors sampled uniformly from `[0.85,1.15]`, retaining positive entries and the same ordering scale.

Reject invalid worlds deterministically and continue to the next attempt.

- [ ] **Step 4: Run focused + full suite**

Run: `pytest tests/test_worlds.py -v && pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add deterministic branched world generator`

---

### Task 4: Gate 6 many-world controls and aggregate receipt

**Files:**
- Create: `experiments/gate6_many_worlds.py`
- Create: `tests/test_gate6.py`
- Modify: `tests/test_experiments.py`
- Create after execution: `results/gate6.json`

**Interfaces:**
- `run_world(world: CableWorld, *, tape_seed: int) -> dict`
- `run(seed: int = 17, n_worlds: int = 24) -> dict`

- [ ] **Step 1: Write failing Gate 6 engineering tests**

Use `n_worlds=2` in tests. Require deterministic finite receipts; two world records; common tape digest per world; `control_update_multiset_max_error < 1e-12`; every spectral radius `< 1`; all alignments in `[0,1]`; aggregate fractions in `[0,1]`. Do not assert deltas positive.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_gate6.py -v`

Expected: import failure because Gate 6 does not exist.

- [ ] **Step 3: Implement tapes and controls**

For each world, construct four adaptation tapes of 2500 samples and five held-out tapes of 5000 samples from its angles/spectra using an explicit RNG derived from `(seed, world.index)`. Use identical tape objects for frozen/local/shuffled/uniform.

Compute `q_target` from initial frozen structure/first adaptation tape. Train local first while saving per-phase update signals. Apply exact permuted signals to shuffled and scalar-mean signals to uniform. Maintain mean conductance normalization and bounds through existing cable utilities.

Evaluate all conditions with common Oja seed base per world/tape. Use the same three-slowest-visible-mode scoring as Gate 5.

- [ ] **Step 4: Implement aggregation**

Return exact per-world records plus aggregate mean, median, q25, minimum, win fractions, loss counts, and mean alignments. `run(seed=17, n_worlds=24)` is the scientific default. Validate `1 <= n_worlds <= 24` so the committed suite remains predeclared.

- [ ] **Step 5: Run smoke and full suite**

Run:

```bash
pytest tests/test_gate6.py tests/test_experiments.py -v
python - <<'PY'
from experiments.gate6_many_worlds import run
print(run(seed=17, n_worlds=2))
PY
pytest -q
```

- [ ] **Step 6: Run the predeclared 24-world receipt exactly once for freezing**

Run: `python experiments/gate6_many_worlds.py --out /tmp/gate6.json`

Do not alter the frozen rule after inspecting this output.

- [ ] **Step 7: Freeze `results/gate6.json`**

Copy the exact deterministic runner output into the repository and rerun the script in CI to confirm reproduction.

- [ ] **Step 8: Commit**

Commit message: `exp: add Gate 6 many-world robustness test`

---

### Task 5: Static `index.html`, documentation, and CI integration

**Files:**
- Create: `index.html`
- Create: `tests/test_index_html.py`
- Modify: `README.md`
- Modify: `PAPER.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Static HTML only; inline `<style>` and inline `<script>` allowed.
- Scientific displayed numbers come only from `results/gate5.json`, `results/gate5b.json`, and `results/gate6.json`.

- [ ] **Step 1: Write failing static-page test**

Require:

```python
html = Path("index.html").read_text(encoding="utf-8")
assert "Hand waving ends here. Executable claims begin here." in html
for gate in ("Gate 0", "Gate 1", "Gate 2", "Gate 3", "Gate 4", "Gate 5", "Gate 5B", "Gate 6"):
    assert gate in html
assert "<script src=" not in html.lower()
assert "rel=\"stylesheet\" href=" not in html.lower()
assert "VIDEO_URL" not in html
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_index_html.py -v`

Expected: failure because `index.html` does not exist.

- [ ] **Step 3: Implement page**

Use a dark scientific-instrument visual language with a hand-drawn SVG-like neuron schematic, animated inline CSS/JS pulses, Gate 5 comparison bars, and compact gate cards. The first section may use “Wild hand waving”; after the hard divider, use restrained scientific language. Do not imply the AIS mechanism has been tested; label it a future hypothesis in the schematic.

- [ ] **Step 4: Update README and paper from frozen receipts**

Add Gate 5B/6 methods, exact results, limitations, and next step (AIS/load adaptation). Do not manually round in a way that changes comparisons; preserve enough digits to reproduce headline deltas.

- [ ] **Step 5: Extend CI**

Add Gate 5B and Gate 6 smoke execution. For normal PR CI, Gate 6 may run `n_worlds=2` through a tiny Python command to control runtime; the committed 24-world receipt remains reproducible via the experiment default and is separately checked by a deterministic receipt test that compares aggregate/per-world output to the committed JSON only when explicitly requested by the scientific script.

- [ ] **Step 6: Run full verification**

Run conceptually on GitHub Actions:

```bash
python -m pip install -e ".[test]"
pytest -q
python experiments/gate0_oja.py --out /tmp/gate0.json
python experiments/gate1_identities.py --out /tmp/gate1.json
python experiments/gate2_mode_energy.py --out /tmp/gate2.json
python experiments/gate3_temporal_modes.py --out /tmp/gate3.json
python experiments/gate4_alignment.py --out /tmp/gate4.json
python experiments/gate5_adaptive_cable.py --out /tmp/gate5.json
python experiments/gate5b_spectral_audit.py --out /tmp/gate5b.json
```

and a Gate 6 two-world smoke run. Require Python 3.11 and 3.12 success.

- [ ] **Step 7: Commit**

Commit message: `docs: publish Gate 5B and Gate 6 explainer`

---

### Task 6: Review and integration

**Files:** no new scientific code unless review finds an issue.

- [ ] **Step 1: Inspect complete PR diff against spec**

Verify Gate 5 unchanged, no positive-result CI assertions, exact shuffled multiset control, all receipt numbers trace to runner output, and page copy does not overclaim AIS/biology.

- [ ] **Step 2: Fresh final CI matrix**

Require Python 3.11 and 3.12 install, full pytest suite, Gates 0–5B smoke, and Gate 6 smoke all successful on the final PR head.

- [ ] **Step 3: Merge only the verified head**

Squash-merge into `main` only if the PR remains mergeable and head SHA is unchanged from the successful matrix.