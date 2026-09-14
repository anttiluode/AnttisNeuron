# AnttisNeuron: spectral two-layer neuron — design

Date: 2026-09-14

## Purpose

Turn the informal neuron sketch into a falsifiable mathematical and computational object.

The project does **not** claim that the two-layer-neuron idea is new. Poirazi, Brannon & Mel (2003) explicitly modeled a pyramidal neuron as nonlinear dendritic subunits followed by somatic pooling. The narrower object studied here is:

> local Oja/Hebbian spectral learning -> branch dynamics -> branch-local nonlinearity -> somatic readout, with an explicit test of whether statistical modes can align with long-lived physical branch modes.

The repository is intentionally a small NumPy research codebase, not a biophysically complete neuron simulator.

## Mathematical object

Branch `j` observes a local vector `x_j(t) in R^{n_j}`.

### 1. Statistical mode selection

Each branch learns a normalized direction with Oja's rule:

`y_j = w_j^T x_j`

`Delta w_j = eta * y_j * (x_j - y_j w_j)`

For stationary zero-mean inputs and suitable conditions this converges toward a principal covariance direction. Multiple branches seeing identical statistics are **not** assumed to discover different PCs without competition. Experiments either give branches distinct local statistics or explicitly introduce orthogonalizing/deflation when a principal subspace is required.

### 2. Physical branch dynamics

A branch is a stable linear dynamical system

`d_j(t+1) = A_j d_j(t) + b_j y_j(t)`

with measured branch output

`r_j(t) = c_j^T d_j(t)`.

This is a causal temporal filter. The eigenmodes of `A_j` are the branch's physical modes. Differential decay can increase the relative purity of slow modes, but it does **not** imply that Oja's covariance eigenvector automatically equals a physical cable eigenmode.

### 3. Local nonlinear subunit

`q_j(t) = psi_j(r_j(t))`

where the initial study supports `identity`, `square`, and a smooth NMDA-like saturating/supralinear function.

### 4. Soma/AIS abstraction

`v(t) = a^T q(t) + bias`

`p(t) = sigmoid(v(t))` for trainable classification experiments, with a hard threshold available for exact algebraic tests.

The soma readout may be fitted. This keeps the scientific question separated: Oja determines branch receptive directions; the supervised readout only asks what becomes linearly readable after the branches.

## Claims to test

### C1 — Oja principal-mode recovery

On synthetic stationary Gaussian inputs with a known covariance spectrum, one branch recovers the leading eigenvector up to sign.

Metric: absolute cosine alignment `|w^T v_1| >= 0.98` in a deterministic seeded test.

### C2 — Linear-collapse identity

If every branch nonlinearity is identity and branch states are replaced by static effective filters `m_j`, then

`sum_j a_j m_j^T x = m^T x`, where `m = sum_j a_j m_j`.

This is an exact numerical identity. It is the negative control: morphology plus purely linear integration does not by itself buy a second computational layer.

### C3 — Two-layer equivalence

For static effective filters, AnttisNeuron

`v = sum_j a_j psi(m_j^T x) + bias`

must match a conventional one-hidden-layer network with hidden weights `m_j`, activation `psi`, and output weights `a_j` to floating-point tolerance.

This establishes architectural equivalence, not biological identity and not novelty.

### C4 — Quadratic-form identity

For `psi(z)=z^2`,

`v-bias = sum_j a_j (m_j^T x)^2 = x^T Q x`,

`Q = sum_j a_j m_j m_j^T`.

This must hold numerically to floating-point tolerance.

### C5 — Nonlinear representational gain

Construct a covariance/mode-energy classification task whose label is not linearly separable in raw input but is linearly separable in squared learned mode responses. Compare:

1. raw-input linear readout;
2. Oja branches with identity nonlinearity;
3. Oja branches with square nonlinearity.

Success criterion: the nonlinear spectral model substantially exceeds both linear controls on held-out data. This is a representational result, not a claim of universal learning superiority.

### C6 — Temporal branch selectivity

Feed mixtures of temporal components into branches with known stable eigenvalues/time constants. Verify that the measured response ratio of a slower versus faster physical mode increases with elapsed time exactly as predicted by differential decay.

This tests the narrow, defensible meaning of "mode purification."

### C7 — Statistical/physical alignment experiment

This is the central exploratory gate and is **not assumed true**.

Construct a branch where input channels couple through a known matrix `B` into physical modes of `A`. Measure alignment between the Oja-learned statistical direction and each input-visible physical mode. Then compare fixed statistics against a coupled condition in which the observed statistics depend on the branch response.

Primary quantity: maximum absolute cosine alignment before/after coupling.

Possible outcomes:

- alignment rises reproducibly: evidence for a coupled statistical/physical mode-selection mechanism in this model;
- alignment stays unchanged: the attractive coupled-eigenproblem story is not supported by this mechanism;
- alignment is parameter-sensitive: report the dependence and do not generalize.

CI must never fail simply because C7 is scientifically negative. CI fails only if the experiment cannot run, outputs non-finite values, or violates algebraic/invariant tests.

## Repository layout

- `anttis_neuron/__init__.py` — public API
- `anttis_neuron/oja.py` — Oja learner and covariance helpers
- `anttis_neuron/branch.py` — stable linear branch dynamics and nonlinearities
- `anttis_neuron/model.py` — spectral branch neuron and simple readout
- `anttis_neuron/tasks.py` — deterministic synthetic datasets
- `experiments/gate0_oja.py` — C1
- `experiments/gate1_identities.py` — C2-C4
- `experiments/gate2_mode_energy.py` — C5
- `experiments/gate3_temporal_modes.py` — C6
- `experiments/gate4_alignment.py` — C7
- `tests/` — unit and scientific invariant tests
- `results/` — committed compact deterministic receipts, not large raw arrays
- `PAPER.md` — paper-style manuscript
- `README.md` — claims, quickstart, gates, and interpretation rules
- `.github/workflows/ci.yml` — Python 3.11/3.12 tests + deterministic experiment smoke tests

## Testing philosophy

Engineering invariants are red/green. Scientific hypotheses are measured and recorded.

Hard tests include:

- deterministic Oja convergence on a well-conditioned covariance;
- branch stability validation (`spectral_radius(A) < 1`);
- linear collapse identity;
- exact shallow-network equivalence;
- exact quadratic-form equivalence;
- deterministic train/test splitting;
- finite outputs and reproducible experiment receipts.

Scientific result tests should assert only broad sanity conditions unless the result follows algebraically. In particular, the alignment experiment is allowed to return a null result.

## Paper scope

Working title:

**A Spectral Two-Layer Neuron: Local Principal-Mode Learning, Dendritic Dynamics, and Nonlinear Subunit Readout**

The manuscript separates four levels of statement:

1. established background: Oja/PCA-like learning and nonlinear dendritic subunits;
2. exact mathematics: linear collapse, shallow-network equivalence, quadratic form;
3. simulation findings from this repository;
4. hypotheses: self-consistent statistical/physical-mode alignment and AIS/load adaptation.

The first version will not claim a new biological mechanism, will not equate a generic linear state-space branch with a real dendrite, and will not claim that passive dendrites selectively purify arbitrary statistical modes.

## Literature anchors

- E. Oja (1982), *A simplified neuron model as a principal component analyzer*, Journal of Mathematical Biology 15:267-273. DOI: 10.1007/BF00275687.
- P. Poirazi, T. Brannon, B. W. Mel (2003), *Pyramidal neuron as two-layer neural network*, Neuron 37:989-999. DOI: 10.1016/S0896-6273(03)00149-1.
- M. London, M. Hausser (2005), *Dendritic computation*, Annual Review of Neuroscience 28:503-532. DOI: 10.1146/annurev.neuro.28.061604.135703.
- A. Gidon et al. (2020), *Dendritic action potentials and computation in human layer 2/3 cortical neurons*, Science 367:83-87. DOI: 10.1126/science.aax6239.
- I. Aizenbud et al. (2026), *Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons*, PNAS 123:e2533168123. DOI: 10.1073/pnas.2533168123.
- C. Leterrier (2018), *The Axon Initial Segment: An Updated Viewpoint*, Journal of Neuroscience 38:2135-2145. DOI: 10.1523/JNEUROSCI.1922-17.2018.

## Definition of done for v0

- all C1-C6 tests pass;
- C7 runs deterministically and records its result without outcome hacking;
- CI is green on supported Python versions;
- README can reproduce every gate from a fresh clone with NumPy + pytest;
- PAPER.md accurately labels prior art, exact identities, measured findings, and speculative extensions;
- no large dependency stack, no fitting hidden inside data generation, and no result numbers hand-written independently of executable receipts.
