# A Spectral Two-Layer Neuron: Local Principal-Mode Learning, Branch Dynamics, and Nonlinear Subunit Readout

Antti Luode  
Working manuscript — 14 September 2026

## Abstract

Nonlinear dendritic subunits can make the input-output function of a single pyramidal neuron resemble a small multilayer network. That architectural observation is established prior work. Here we study a narrower synthesis: what if branch receptive directions are acquired locally by an Oja-style principal-component rule, transformed by physical branch dynamics, and then combined through branch-local nonlinearities?

We separate exact algebraic consequences from synthetic experiments. Purely linear branches collapse exactly to a single linear filter (maximum numerical error `1.78e-15`), whereas nonlinear branches are exactly equivalent to a one-hidden-layer network under the static abstraction. With a square branch nonlinearity the model reduces exactly to a quadratic classifier `x^T Q x` (maximum numerical error `4.26e-14`). On a variance-only classification task, raw linear and linear-branch controls obtain `49.4%` and `50.8%` held-out accuracy, while two locally Oja-learned mode detectors followed by square responses obtain `93.5%`; mean learned-filter alignment with the generating modes is `0.999900`.

A differential-decay experiment verifies the narrow sense in which branch dynamics can increase relative mode purity. A constructive two-state recurrence then raises statistical-to-physical mode alignment from `0.722183` to `0.999010`, while a no-memory control remains at `0.722179`.

The stronger experiment replaces that hand-built two-state system with an 11-node branched graph cable. Edge conductances adapt using only local squared voltage differences; the rule never receives physical eigenvectors, Oja weights, alignment scores, labels, or held-out metrics. Across five held-out covariance rotations, mean Oja-to-physical-mode alignment rises from `0.782185` for the frozen cable to `0.900191` after local structural adaptation. An exact-multiset shuffled-credit control, which receives the same per-phase update values assigned to the wrong edges, obtains `0.759219`; the update-multiset mismatch is exactly `0.0`. The adaptive minimum held-out alignment rises from `0.606365` to `0.839369`.

These results identify a compact synthetic computational object—local spectral learning coupled to a physical substrate whose own local activity can slowly reshape that substrate. They do not establish that biological dendrites use this rule or generally optimize eigenmode alignment.

## 1. Introduction

The phrase "a single neuron is a two-layer neural network" is not a new claim. Poirazi, Brannon, and Mel showed in 2003 that the firing-rate response of a detailed CA1 pyramidal-cell model could be approximated by nonlinear dendritic subunits whose outputs were pooled at the soma. Broader work on dendritic computation has established that dendrites can perform both linear and nonlinear operations, and experiments in human layer 2/3 pyramidal neurons have demonstrated dendritic mechanisms capable of linearly nonseparable computation.

A separate classical line of work concerns local statistical learning. Oja's normalized Hebbian rule gives a simple neuron-like update whose weight vector converges, under standard assumptions, toward a leading covariance eigenvector. Oja learning therefore provides a local mechanism for constructing a direction through a multidimensional input cloud rather than selecting a single input wire.

The present study asks what happens when these ideas are connected through an explicit physical branch:

```text
local input statistics
        ↓
Oja spectral direction
        ↓
physical branch dynamics
        ↓
branch-local nonlinearity
        ↓
somatic pooled readout
```

The point is not biological completeness. It is to expose exactly which parts of the intuition are algebraic, which are prior art, and which remain scientific hypotheses.

The central negative control is simple. If every branch is linear, then multiple branches do not by themselves create a deeper computation: they collapse into one effective linear filter. Computational depth enters only when a nonlinearity acts before branch outputs are pooled.

The next question is more unusual. Physical dynamics do not merely transform a representation after it is learned. They also change the time series, covariance, and local activity from which future learning is computed. If the physical structure itself changes slowly in response to local activity, statistical learning and physical dynamics can form a feedback loop even without backpropagating an alignment objective through the substrate.

AnttisNeuron tests progressively stronger versions of that idea. Gate 4 is a deliberately constructive existence proof. Gate 5 removes the explicit spectral construction and asks whether an alignment-blind local structural rule can produce a useful statistical/physical alignment effect on a branched operator and generalize to held-out input statistics.

## 2. Related work

### 2.1 Oja learning and principal directions

Oja (1982) introduced a simplified neuron model that behaves as a principal-component analyzer. In the one-unit setting used here, the relevant statement is limited: the learned vector approaches a leading covariance eigenvector. Multiple identical Oja units observing the same distribution are not assumed to discover distinct principal components without competition or decorrelation.

### 2.2 Nonlinear dendritic subunits

Poirazi, Brannon, and Mel (2003) mapped a detailed CA1 pyramidal-cell model to a two-layer abstraction: dendritic subunits perform local nonlinear transformations and the soma pools their outputs. London and Häusser (2005) reviewed mechanisms by which dendrites contribute to neuronal computation. Gidon et al. (2020) reported dendritic calcium action potentials in human layer 2/3 cortical neurons and showed that the measured nonlinear behavior could support linearly nonseparable computation.

Aizenbud et al. (2026) introduced a Functional Complexity Index and reported that modeled human cortical pyramidal neurons were more difficult for deep networks to approximate than corresponding rat neurons, with dendritic membrane area, branching pattern, and NMDA receptor density/nonlinearity contributing strongly to functional complexity. These results motivate studying morphology/dynamics and branch nonlinearity as computational components, but they do not imply the spectral-learning mechanism studied here.

### 2.3 Axon initial segment plasticity

The axon initial segment (AIS) controls action-potential initiation and exhibits structural and functional plasticity. Leterrier (2018) reviewed the dependence of neuronal excitability on AIS composition and position. AnttisNeuron does not model AIS plasticity. The soma/AIS remains a pooled scalar readout; coupling branch load to an adaptive output boundary is left for a later gate.

## 3. Model

### 3.1 Local spectral learning

Branch `j` observes a local vector

```math
x_j(t) \in \mathbb{R}^{n_j}.
```

Its Oja unit computes

```math
y_j(t)=w_j^T x_j(t)
```

and updates

```math
\Delta w_j = \eta y_j\left(x_j-y_j w_j\right).
```

For stationary zero-mean data under the usual conditions, the expected fixed point is a covariance eigenvector. We use sign-invariant cosine alignment because eigenvectors are defined only up to sign.

### 3.2 Generic physical branch

The minimal dynamic branch is

```math
d_j(t+1)=A_j d_j(t)+b_j y_j(t),
```

with measured scalar response

```math
r_j(t)=c_j^T d_j(t).
```

A branch-local nonlinearity gives

```math
q_j(t)=\psi_j(r_j(t)),
```

and the soma-like readout is

```math
v(t)=\sum_j a_j q_j(t)+b.
```

For classification experiments, the final readout is fitted only after branch features have been formed. This separates the representation exposed by the branches from the simple question of whether a linear output can use it.

### 3.3 Static effective-filter abstraction

For the exact algebraic tests we replace temporal branch history by static effective filters `m_j` and write

```math
v(x)=\sum_j a_j\psi(m_j^T x)+b.
```

This abstraction is sufficient to prove the key controls.

### 3.4 Branched graph cable

Gate 5 uses a fixed 11-node tree with ten positive edge conductances. For symmetric weighted adjacency `W(g)`, the graph Laplacian is

```math
L(g)=D(g)-W(g).
```

The discrete leaky diffusion operator is

```math
A(g)=I-dt\,[\ell I+\kappa L(g)],
```

and state evolves as

```math
d_{t+1}=A(g)d_t+Bx_t.
```

The learner does not see the full state. A fixed sensor matrix selects four node voltages,

```math
z_t=S d_t \in \mathbb R^4,
```

and Oja learning operates only on `z_t`.

The physical modes used for evaluation are eigenvectors of the symmetric operator `A(g)` projected into the same four-dimensional sensor space. The nearly uniform global leak mode is excluded from the main alignment score. The score uses the three slowest remaining visible modes so that trivial global smoothing cannot count as successful route formation.

### 3.5 Alignment-blind local structural adaptation

For edge `e=(i,j)`, define local voltage-difference energy

```math
q_e = \mathbb E[(d_i-d_j)^2].
```

A fixed homeostatic target `q_*` is measured once from the initial frozen structure. Conductance updates as

```math
g_e \leftarrow g_e \exp\left[\eta_g\left(\frac{q_e}{q_*}-1\right)\right],
```

followed by clipping and mean-conductance renormalization.

The rule is deliberately blind to the scientific evaluation target. Its implementation accepts neither physical eigenvectors nor Oja weights, alignment values, class labels, or held-out metrics. Mean-conductance normalization also prevents the trivial solution of increasing every conductance together.

## 4. Exact results

### 4.1 Linear branches collapse

If `psi(z)=z`, then

```math
v(x)=\sum_j a_j m_j^T x+b
     =\left(\sum_j a_j m_j\right)^T x+b.
```

Thus a static multi-branch linear neuron is only one affine linear unit. The implementation verifies the identity with maximum absolute error `1.7763568394002505e-15`.

### 4.2 Nonlinear branches are a shallow network

For arbitrary pointwise `psi`, define

```math
h_j(x)=\psi(m_j^Tx).
```

Then

```math
v(x)=a^T h(x)+b,
```

which is exactly the forward equation of a one-hidden-layer network whose hidden units are branch subunits. The implementation obtains zero numerical difference from an explicit shallow-network expression.

### 4.3 Square branches are a quadratic form

For

```math
\psi(z)=z^2,
```

we have

```math
v-b=\sum_j a_j(m_j^Tx)^2
   =x^T\left(\sum_j a_jm_jm_j^T\right)x.
```

With

```math
Q=\sum_j a_jm_jm_j^T,
```

this becomes

```math
v=x^TQx+b.
```

The direct branch computation and quadratic form agree to maximum absolute error `4.263256414560601e-14`.

## 5. Experiments

All committed results use seed `17`. JSON receipts are written by the experiment scripts and checked into `results/`.

### 5.1 Gate 0 — Oja recovery

A five-dimensional Gaussian distribution is generated with known covariance spectrum `[9.0, 3.0, 1.5, 0.7, 0.3]` in a random orthonormal basis. A single Oja unit is trained on 14,000 samples for three epochs. Performance is sign-invariant cosine alignment with the analytic leading covariance eigenvector.

### 5.2 Gate 1 — algebraic identities

Random inputs, branch filters, and output weights test linear collapse, explicit shallow-network equivalence, and the square/quadratic-form identity. These are implementation checks of algebraic statements rather than empirical hypotheses.

### 5.3 Gate 2 — variance-only classification

Two orthonormal task modes are generated in six dimensions. Both classes remain centered near zero. Class 0 has high variance along mode 0 and low variance along mode 1; class 1 reverses those variances. Two branches receive different local streams, each dominated by one task mode, and learn their directions with Oja's rule without class labels.

A ridge readout is fitted on raw input, linear Oja branch coordinates, and squared Oja branch coordinates. The task is intentionally constructed so first-order sign information is unhelpful while second-order mode energy is diagnostic.

### 5.4 Gate 3 — differential decay

Two physical modes decay with rates `mu_slow=0.03` and `mu_fast=0.21`. Starting from equal amplitudes, the analytic relative amplitude is

```math
R(t)=\exp[(\mu_{fast}-\mu_{slow})t].
```

The same ratio is measured in a diagonal discrete-time system. This tests only the narrow claim that slower modes become purer *relative* to faster modes.

### 5.5 Gate 4 — constructive statistical/physical alignment

A two-state stable physical system has eigenvalues `0.97` and `0.65`. External Gaussian statistics begin halfway between its two input-visible physical modes. The recurrent state evolves as

```math
d_{t+1}=Ad_t+Bx_t,
```

and is observed back in input coordinates. The slow mode accumulates more variance; Oja follows the covariance of that recurrent observation. A no-memory control sets `A=0`.

This experiment asks whether the causal mechanism can exist. It does not ask whether alignment appears spontaneously under local structural learning.

### 5.6 Gate 5 — adaptive branched cable with matched controls

Gate 5 replaces the two-state construction with the 11-node graph cable of Section 3.4. External input enters four distal ports. Four node voltages are observed. Four adaptation covariance environments use rotations of `12°`, `31°`, `53°`, and `74°`; five held-out evaluation environments use `22°`, `42°`, `63°`, `83°`, and `103°`.

All conditions receive common random input tapes. Oja initialization is also shared per evaluation tape so differences between conditions are not random-initialization effects.

Four structural conditions are evaluated:

1. **Frozen:** all ten conductances remain one.
2. **Local adaptive:** each edge updates from its own voltage-difference energy.
3. **Shuffled credit:** on every adaptation phase, this control receives the exact multiset of update values generated by the local-adaptive condition, but the values are randomly permuted across edges before application.
4. **Uniform:** every edge receives the same phase-mean update before mean-conductance normalization, leaving relative conductances unchanged.

The shuffled condition is deliberately stronger than merely running the same learning law on a different trajectory. Its per-phase update multiset is exactly matched to the local condition; the committed receipt reports maximum mismatch `0.0`. Therefore the adaptive-versus-shuffled comparison isolates *where* local credit is assigned, not update magnitude or distribution.

No software test requires Gate 5 alignment to increase. CI checks determinism, finiteness, graph/operator invariants, conductance bounds and normalization, common input tapes, the alignment-blind adaptor interface, and the exact shuffled-update multiset.

## 6. Results

| Measurement | Result |
|---|---:|
| Gate 0 Oja↔true-PC alignment | `0.9955947725` |
| Gate 1 linear-collapse max error | `1.7763568394e-15` |
| Gate 1 two-layer equivalence max error | `0.0` |
| Gate 1 quadratic-form max error | `4.2632564146e-14` |
| Gate 2 filter 0 alignment | `0.9999145444` |
| Gate 2 filter 1 alignment | `0.9998860968` |
| Gate 2 raw linear accuracy | `0.4940` |
| Gate 2 identity-branch accuracy | `0.50775` |
| Gate 2 square-branch accuracy | `0.9350` |
| Gate 3 slow/fast ratio at `t=20` | `36.5982344437` |
| Gate 3 analytic/sim max error | `7.1054273576e-15` |
| Gate 4 fixed alignment | `0.7221832493` |
| Gate 4 no-memory alignment | `0.7221787537` |
| Gate 4 recurrent alignment | `0.9990103496` |
| Gate 4 alignment change | `+0.2768271003` |
| Gate 5 frozen held-out mean alignment | `0.7821845260` |
| Gate 5 local-adaptive held-out mean alignment | `0.9001914311` |
| Gate 5 shuffled-credit held-out mean alignment | `0.7592192537` |
| Gate 5 uniform held-out mean alignment | `0.7821845260` |
| Gate 5 adaptive − frozen | `+0.1180069051` |
| Gate 5 adaptive − shuffled | `+0.1409721775` |
| Gate 5 frozen held-out minimum | `0.6063645741` |
| Gate 5 adaptive held-out minimum | `0.8393687765` |
| Gate 5 local energy/conductance-change correlation | `0.9762404274` |
| Gate 5 frozen slow non-uniform eigenvalue | `0.9859826076` |
| Gate 5 adaptive slow non-uniform eigenvalue | `0.9881552640` |
| Gate 5 shuffled update-multiset max error | `0.0` |

### 6.1 Spectral nonlinear features solve the intended second-order task

The raw linear baseline (`49.4%`) and identity-branch model (`50.775%`) remain at chance, whereas the square-branch model reaches `93.5%`. The two Oja filters align almost perfectly with the locally dominant modes. Thus local unsupervised learning recovers useful spectral coordinates, but those coordinates become class-informative only after the nonlinear energy transform.

### 6.2 Differential decay gives a precise weak form of mode purification

The slow/fast ratio grows to `36.60` at `t=20`, and simulation matches the analytic ratio to floating-point precision. This validates only relative differential decay. A passive stable system does not choose an arbitrary desired covariance mode.

### 6.3 Physical recurrence can write its spectrum into local statistics

Gate 4 begins at alignment `0.722183`; the no-memory control is `0.722179`; recurrence raises alignment to `0.999010`. The slow physical state accumulates variance, causing the covariance seen by Oja to become dominated by the slow physical eigenmode.

Gate 4 therefore demonstrates a causal arrow:

```text
physical dynamics -> changed observed covariance -> changed local spectral learning.
```

Because the unequal decay rates are explicitly constructed, Gate 4 remains an existence proof.

### 6.4 Alignment-blind local structure improves held-out coupling

Gate 5 asks the harder question. The local structural rule is never shown an eigenvector or alignment score. Nevertheless, after four adaptation environments its mean held-out alignment is `0.900191`, compared with `0.782185` for the frozen graph, a gain of `+0.118007`.

The result is not driven only by one favorable environment. Per-held-out alignments for the local-adaptive cable are

```text
0.873230, 0.884271, 0.922112, 0.981976, 0.839369,
```

while the frozen cable gives

```text
0.606365, 0.608281, 0.982430, 0.972830, 0.741017.
```

The mean therefore increases while the worst held-out case rises from `0.606365` to `0.839369`. Two environments already aligned strongly under the frozen structure and remain strong; the largest changes occur where the original structure was poorly matched.

The exact-multiset shuffled-credit control obtains mean alignment `0.759219`, so correct local credit beats the matched wrong-edge assignment by `+0.140972`. This matters because the shuffled condition receives exactly the same update-value multiset on every phase. What differs is which physical edge receives which change.

The uniform control returns `0.782185`, numerically reproducing the frozen condition as expected after mean-conductance normalization. This verifies that a global scalar conductance change is not the source of the effect.

The final adaptive conductance coefficient of variation is `0.509652`, and adaptation-ensemble local edge energy correlates `0.976240` with conductance change. The slowest non-uniform physical eigenvalue moves from `0.985983` to `0.988155`. Together these measurements show a concrete sequence:

```text
local edge activity
        ↓
nonuniform structural change
        ↓
changed physical spectrum / state statistics
        ↓
changed Oja direction under bounded observation.
```

This is stronger than Gate 4 because no global spectral target constructs the desired physical mode. It is still a synthetic result, not a biological mechanism claim.

## 7. Discussion

AnttisNeuron now separates four conceptual levels that are easy to mix together.

First, Oja learning and fixed branch filtering are linear with respect to a fixed state. They can supply selectivity and temporal memory, but a static multi-branch neuron without local nonlinearity collapses to one linear map.

Second, branch-local nonlinearity before pooling creates genuine hidden computational subunits in the ordinary shallow-network sense. The square special case makes the computation interpretable as weighted energy in learned directions.

Third, physical dynamics can define part of the learning problem. Gate 4 shows that recurrence changes the covariance available to a local learner even when the learner itself has no model of the physical system.

Fourth, the physical structure can itself change through a local rule. Gate 5 closes one additional arrow: edge-local state differences slowly alter conductances, which alter the graph spectrum and future bounded observations. The Oja learner then adapts to statistics produced by this changed substrate.

The important point is not that the local rule "knows" about eigenmodes. It does not. The update is defined entirely by local voltage-difference energy plus a scalar homeostatic target. The positive alignment result is therefore an emergent consequence of this particular synthetic system rather than direct optimization of the reported metric.

The shuffled-credit control strengthens that interpretation. Matching the update-value multiset while destroying edge-local assignment removes the gain. In this experiment, *where* the same structural changes are applied matters more than their mere magnitude distribution.

This begins to resemble the coupled fixed-point picture that motivated the project. A full version would contain at least three mutually interacting variables:

```math
C(A,\theta)w=\lambda w,
```

with slow structural dynamics for `A` and homeostatic/output dynamics for `theta`. Gate 5 does not solve that full system, but it demonstrates two-way coupling between local activity statistics and a physical operator without using a global alignment objective.

The novelty claim should therefore remain narrow. Two-layer dendritic computation is established. Oja/PCA learning is established. Weighted graph diffusion and local homeostasis are established mathematical ingredients. What is new here, if useful, is the explicitly falsifiable synthesis and control structure: locally learned statistical modes, a physical operator that changes observed statistics, and an alignment-blind local structural rule whose edge-specific credit can be tested against exact matched controls.

## 8. Limitations and falsification criteria

### 8.1 The branched cable is still an abstraction

Gate 5 improves on the generic `LinearBranch` by introducing topology, weighted edges, bounded sensors, and a graph Laplacian. It is nevertheless a discrete linear diffusion system. It omits membrane capacitance units, compartment geometry, active ion channels, synaptic conductance kinetics, NMDA voltage dependence, stochastic channels, and realistic dendritic plasticity.

Calling it a "cable" describes the mathematical analogy, not a validated biophysical neuron.

### 8.2 The local conductance rule is a hypothesis generator

The edge rule was chosen because it is local, simple, homeostatic, and alignment-blind. The present result does not establish that biological dendrites update axial or membrane properties according to squared voltage differences, nor that such a rule is optimal.

A stronger study should test a family of local rules predeclared before outcome inspection and identify which effects survive rule changes.

### 8.3 Oja learning remains local PCA

A single Oja unit is biased toward one dominant covariance direction. Multiple branches observing the same distribution require competition, deflation, Sanger's rule, or another mechanism to reliably span multiple PCs.

### 8.4 Gate 2 is deliberately constructed

The nonlinear benchmark is designed so squared mode energy is the correct representation. Its purpose is explanatory, not to establish broad task superiority.

### 8.5 Gate 4 is an existence proof

Gate 4 explicitly builds unequal physical decay rates and then observes the recurrent state. Its positive alignment is therefore evidence that the causal path can exist, not that it self-organizes.

### 8.6 Gate 5 is stronger but not biological validation

Gate 5 predeclares an alignment-blind local rule and evaluates held-out covariance rotations with matched controls. The positive result is therefore not merely the Gate 4 construction repeated. However, one topology, one sensor placement, one parameter set, one adaptation rule, and one seed do not establish a general phenomenon.

The next falsification should sweep seeds, topologies, port/sensor arrangements, conductance bounds, and local-rule families without retuning each case. A useful phenomenon should survive a meaningful fraction of these changes and should continue to beat exact-multiset shuffled credit.

### 8.7 No AIS/output adaptation yet

The current output readout does not implement activity-dependent AIS location, length, conductance, or dendritic-load matching. The proposed statistical-mode ↔ physical-mode ↔ output-boundary fixed point therefore remains incomplete.

### 8.8 Scientific nulls remain valid

CI never asserts that Gate 4 or Gate 5 alignment must increase. It asserts determinism, finiteness, structural invariants, matched inputs/updates, and valid numerical bounds. A negative future receipt is allowed to remain negative.

This separation between engineering gates and scientific outcomes is intentional protection against tuning the code until a preferred story appears.

## 9. Next experiments

Gate 5 completes much of the previously proposed "stronger future test," so the next sequence should move from one successful construction toward generality and closure:

1. **Gate 6 — generalization matrix:** run many fixed seeds, several tree topologies, multiple port/sensor placements, and covariance families with no per-case retuning. Report the distribution of adaptive−frozen and adaptive−shuffled deltas.
2. **Rule ablation:** compare voltage-difference homeostasis with alternative local rules matched for update scale to determine whether the effect is specific or generic.
3. **Multi-mode learning:** replace the single Oja direction with Sanger-style local subspace learning and ask whether physical structure supports several persistent statistical modes simultaneously.
4. **Active branch dynamics:** introduce a minimal local nonlinear conductance only after the passive graph controls are understood, then test whether the structural effect survives.
5. **Output-boundary adaptation:** add a slow homeostatic soma/AIS-like variable and test the three-way fixed point between statistical mode, physical mode, and output boundary.
6. **Biophysical transfer:** only after the abstract controls survive, port the strongest gate to a compartmental cable/NEURON-style model and ask which conclusions remain.

The decisive next result is no longer simply "alignment rose once." It is whether the alignment-blind local mechanism survives perturbation of topology, statistics, observation, and random seed while continuing to outperform exact matched wrong-edge credit.

## References

1. Oja, E. (1982). *A simplified neuron model as a principal component analyzer*. Journal of Mathematical Biology, 15, 267–273. DOI: 10.1007/BF00275687.
2. Poirazi, P., Brannon, T., & Mel, B. W. (2003). *Pyramidal neuron as two-layer neural network*. Neuron, 37, 989–999. DOI: 10.1016/S0896-6273(03)00149-1.
3. London, M., & Häusser, M. (2005). *Dendritic computation*. Annual Review of Neuroscience, 28, 503–532. DOI: 10.1146/annurev.neuro.28.061604.135703.
4. Gidon, A., et al. (2020). *Dendritic action potentials and computation in human layer 2/3 cortical neurons*. Science, 367, 83–87. DOI: 10.1126/science.aax6239.
5. Leterrier, C. (2018). *The Axon Initial Segment: An Updated Viewpoint*. Journal of Neuroscience, 38, 2135–2145. DOI: 10.1523/JNEUROSCI.1922-17.2018.
6. Aizenbud, I., Yoeli, D., Beniaguev, D., de Kock, C. P. J., London, M., & Segev, I. (2026). *Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons*. Proceedings of the National Academy of Sciences, 123, e2533168123. DOI: 10.1073/pnas.2533168123.
