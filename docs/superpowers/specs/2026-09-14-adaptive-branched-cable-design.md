# Adaptive Branched Cable Gate Design

## Goal

Replace the constructive two-state alignment toy with a nontrivial branched physical operator and ask a harder question: can a **local, alignment-blind structural adaptation rule** reshape branch dynamics so that Oja learning from bounded observations becomes more aligned with physically persistent modes on held-out input statistics?

The experiment must remain falsifiable. A null or negative result is scientifically valid and must not fail CI.

## Scope

This slice adds one reusable graph/cable module and one new scientific experiment, Gate 5. It does not replace Gate 4, add conductance-based ion channels, optimize morphology with a global search method, or claim biological validation.

## Physical substrate

Use an 11-node fixed tree with one root, a trunk, and three terminal branches. The topology is fixed for v1; only positive edge conductances can adapt.

For symmetric weighted adjacency `W(g)`, define the graph Laplacian

```math
L(g)=D(g)-W(g).
```

The discrete leaky cable update is

```math
d_{t+1}=A(g)d_t + Bx_t,
```

with

```math
A(g)=I-dt\,[\ell I+\kappa L(g)].
```

Parameters are chosen so `rho(A)<1`. The graph eigenvectors of `A` define the physical modes used only for evaluation.

## Bounded observation

The learner does not observe the complete state. A fixed sensor matrix `S` selects/combines four distal/trunk node voltages:

```math
z_t=S d_t \in R^4.
```

Oja receives only `z_t`. Physical-mode visibility in sensor space is

```math
p_k = S \phi_k,
```

normalized when nonzero. Evaluation alignment is the maximum sign-invariant cosine between the Oja vector and visible nontrivial physical modes. The near-uniform leak mode is excluded from the main score because trivial global smoothing should not count as successful route formation.

## Input ensembles

Use several zero-mean Gaussian input covariance conditions. Inputs enter through four distal ports via fixed `B`.

- adaptation ensemble: a sequence of covariance matrices whose principal directions are deliberately not identical to the physical modes;
- held-out ensembles: covariance rotations and spectra not used during conductance adaptation;
- fixed random seeds make all receipts deterministic.

The same input tapes must be reused across frozen/adaptive/control conditions wherever comparison requires common random numbers.

## Local structural rule

Each edge `(i,j)` has conductance `g_e` constrained to `[g_min, g_max]`.

The adaptation signal may use only activity available at that edge. Define local voltage difference energy

```math
q_e = EMA[(d_i-d_j)^2].
```

and a homeostatic target `q_target`. Conductance evolves slowly:

```math
g_e <- clip(g_e * exp(eta_g * (q_e/q_target - 1)), g_min, g_max).
```

Interpretation: edges carrying persistently larger-than-target local differences become more conductive; quiet edges weaken. The rule has no access to eigenvectors, Oja weights, held-out alignment, class labels, or a global objective.

A total-conductance normalization keeps the mean edge conductance fixed after each structural update, preventing the trivial solution of increasing every edge equally.

## Controls

Gate 5 compares at least four conditions.

1. **Frozen** — initial conductances never change.
2. **Adaptive local** — the true local edge rule above.
3. **Shuffled-credit control** — the same multiset of local update magnitudes is randomly permuted across edges before application. This preserves update scale but destroys edge-local credit.
4. **Uniform-scaling control** — all edges receive the same multiplicative update followed by mean-conductance normalization; therefore topology-relative conductances do not change.

The shuffled control is the important anti-cheating control: if any generic heterogeneity/drift gives the same result, local structural credit is not doing useful work.

## Measurements

For each condition record:

- initial and final conductance vectors;
- spectral radius of `A`;
- visible physical eigenmodes;
- Oja-to-physical alignment on adaptation data;
- mean, median, minimum, and per-ensemble alignment on held-out covariance conditions;
- change from the frozen baseline;
- conductance coefficient of variation;
- correlation between adaptation-ensemble local edge energy and final conductance change;
- state RMS/stability diagnostics.

Primary scientific metric:

```math
Delta_heldout = mean_alignment_adaptive - mean_alignment_frozen.
```

This metric is reported, not required to be positive by CI.

Secondary discrimination metric:

```math
adaptive_local - shuffled_credit.
```

Again, report rather than hard-code a desired sign.

## Hypotheses

### H1 — structural consequence

The local rule produces nonuniform conductance changes while preserving stability and mean conductance.

### H2 — generalization

If the rule captures a real structural principle rather than overfitting one covariance, adaptive-local alignment should improve relative to frozen on at least some held-out covariance conditions and preferably in their mean.

### H3 — locality matters

If edge-local credit is important, adaptive-local should differ from shuffled-credit despite matching update magnitudes.

These are scientific hypotheses, not CI requirements.

## Engineering invariants

CI **does** require:

- Laplacian symmetry and zero row sum;
- graph operator stability for supported parameters;
- conductances remain finite and within bounds;
- mean-conductance normalization is preserved to numerical tolerance;
- deterministic Gate 5 receipts for a fixed seed;
- all reported metrics are finite;
- common-tape controls receive identical external inputs;
- uniform-scaling control leaves relative conductances unchanged;
- no code path in the local adaptor accepts physical eigenvectors or alignment scores.

## Files

Create:

- `anttis_neuron/cable.py` — tree topology, Laplacian/operator construction, simulation, visible-mode utilities, local structural adaptor.
- `experiments/gate5_adaptive_cable.py` — frozen/adaptive/shuffled/uniform experiment.
- `tests/test_cable.py` — mathematical and adaptation invariants.
- `results/gate5.json` — frozen receipt after the implementation is verified.

Modify:

- `tests/test_experiments.py` — Gate 5 determinism/finiteness only; no positive scientific-result assertion.
- `.github/workflows/ci.yml` — smoke-run Gate 5.
- `README.md` and `PAPER.md` — report Gate 5 exactly as observed, including a null/negative result if that is what occurs.

## Interpretation boundary

A positive Gate 5 would establish only this:

> In a synthetic branched diffusion operator, an alignment-blind local conductance rule can alter the physical spectrum and the bounded state statistics in a way that improves subsequent Oja-to-physical-mode alignment under some held-out input statistics.

It would **not** establish that real dendrites use this rule, that morphology performs PCA, that biological neurons optimize eigenmode alignment, or that the coupled eigenproblem is solved generally.

A null Gate 5 would also be useful: it would tell us that the attractive Gate 4 effect does not survive this removal of explicit spectral construction under the tested local rule.