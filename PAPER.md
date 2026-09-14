# A Spectral Two-Layer Neuron: Local Principal-Mode Learning, Branch Dynamics, and Nonlinear Subunit Readout

Antti Luode  
Working manuscript — 14 September 2026

## Abstract

Nonlinear dendritic subunits can make the input-output function of a single pyramidal neuron resemble a small multilayer network. That architectural observation is established prior work. Here we study a narrower synthesis: what if each branch's receptive direction is not an arbitrary supervised weight vector, but is acquired locally by an Oja-style principal-component rule, then transformed by stable branch dynamics before a branch-local nonlinearity and somatic pooling? We define a minimal model and separate exact algebraic consequences from synthetic experiments. Purely linear branches collapse exactly to a single linear filter (maximum numerical error `1.78e-15`), whereas nonlinear branches are exactly equivalent to a one-hidden-layer network under the static abstraction. With a square branch nonlinearity the model reduces exactly to a quadratic classifier `x^T Q x` (maximum numerical error `4.26e-14`). On a variance-only classification task, raw linear and linear-branch controls obtain `49.4%` and `50.8%` held-out accuracy, while two locally Oja-learned mode detectors followed by square branch responses obtain `93.5%`; mean learned-filter alignment with the generating modes is `0.999900`. A differential-decay experiment verifies the narrow sense in which branch dynamics can increase relative mode purity. Finally, a constructive recurrent-state experiment raises statistical-to-physical mode alignment from `0.722183` to `0.999010`, while a no-memory control remains at `0.722179`. The latter result demonstrates that physical dynamics can reshape the statistics subsequently selected by Oja learning in this model; it does not establish spontaneous biological co-adaptation. The study therefore identifies a compact computational object—local spectral learning followed by physical filtering and nonlinear subunit pooling—and a set of falsifiable next questions about statistical/physical eigenmode alignment.

## 1. Introduction

The phrase "a single neuron is a two-layer neural network" is not a new claim. Poirazi, Brannon, and Mel showed in 2003 that the firing-rate response of a detailed CA1 pyramidal-cell model could be approximated by nonlinear dendritic subunits whose outputs were pooled at the soma. Broader work on dendritic computation has established that dendrites can perform both linear and nonlinear operations, and experiments in human layer 2/3 pyramidal neurons have demonstrated dendritic mechanisms capable of linearly nonseparable computation.

A separate classical line of work concerns local statistical learning. Oja's normalized Hebbian rule gives a simple neuron-like update whose weight vector converges, under standard assumptions, toward a leading covariance eigenvector. Oja learning therefore provides a local mechanism for constructing a direction through a multidimensional input cloud rather than selecting a single input wire.

The present study asks what happens when these two ideas are connected through an explicit physical branch:

```text
local input statistics
        ↓
Oja spectral direction
        ↓
stable branch dynamics
        ↓
branch-local nonlinearity
        ↓
somatic pooled readout
```

The point of the model is not biological completeness. It is to expose exactly which parts of the intuition are algebraic, which are ordinary prior art, and which remain genuine scientific hypotheses.

The central discipline is a negative control. If every branch is linear, then multiple branches do not by themselves create a deeper computation: they collapse into one effective linear filter. Any computational gain attributed to a "two-layer neuron" must therefore enter through a local nonlinearity before branch outputs are pooled. Once that is made explicit, the spectral-learning question becomes sharper: can locally learned statistical modes provide useful hidden features, and can physical branch dynamics reshape which statistical modes are learned?

## 2. Related work

### 2.1 Oja learning and principal directions

Oja (1982) introduced a simplified neuron model that behaves as a principal-component analyzer. In the one-unit setting used here, the relevant statement is limited: the learned vector approaches a leading covariance eigenvector. Multiple identical Oja units observing the same distribution are not assumed to discover distinct principal components without an additional competition or decorrelation mechanism.

### 2.2 Nonlinear dendritic subunits

Poirazi, Brannon, and Mel (2003) mapped a detailed CA1 pyramidal-cell model to a two-layer abstraction: dendritic subunits perform local nonlinear transformations and the soma pools their outputs. London and Häusser (2005) reviewed a broad set of mechanisms by which dendrites contribute to neuronal computation. Gidon et al. (2020) reported dendritic calcium action potentials in human layer 2/3 cortical neurons and showed that the measured nonlinear behavior could support linearly nonseparable computation.

More recently, Aizenbud et al. (2026) introduced a Functional Complexity Index and reported that modeled human cortical pyramidal neurons were more difficult for deep networks to approximate than corresponding rat neurons, with dendritic membrane area, branching pattern, and NMDA receptor density/nonlinearity contributing strongly to functional complexity. These results motivate studying morphology/dynamics and branch nonlinearity as computational components, but they do not imply the spectral-learning mechanism proposed here.

### 2.3 Axon initial segment plasticity

The axon initial segment (AIS) controls action-potential initiation and exhibits structural and functional plasticity. Leterrier (2018) reviewed the dependence of neuronal excitability on AIS composition and position. AnttisNeuron does not model AIS plasticity in v0. The soma/AIS is represented only as a pooled scalar readout. Coupling branch load to AIS geometry is left as a future extension rather than silently folded into the present results.

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

For stationary zero-mean data under the usual conditions, the expected fixed point is a covariance eigenvector. In the experiments below, branches either observe different local streams or the desired basis is otherwise explicitly controlled; we do not rely on identical Oja units spontaneously diversifying.

### 3.2 Physical branch

The minimal dynamic branch is

```math
d_j(t+1)=A_j d_j(t)+b_j y_j(t),
```

with measured scalar response

```math
r_j(t)=c_j^T d_j(t).
```

`A_j` is required to have spectral radius below one in the discrete-time implementation. This is a generic stable linear system, not a conductance-based dendrite.

A branch-local nonlinearity gives

```math
q_j(t)=\psi_j(r_j(t)),
```

and the soma-like readout is

```math
v(t)=\sum_j a_j q_j(t)+b.
```

For classification experiments, the final readout is fitted by ridge regression only after the branch features have been formed. This separates the question "what representation do the branches expose?" from the question "can a simple output weight use it?"

### 3.3 Static effective-filter abstraction

For the exact algebraic tests we replace temporal branch history by static effective filters `m_j` and write

```math
v(x)=\sum_j a_j\psi(m_j^T x)+b.
```

This abstraction is sufficient to prove the key controls.

## 4. Exact results

### 4.1 Linear branches collapse

If `psi(z)=z`, then

```math
v(x)=\sum_j a_j m_j^T x+b
     =\left(\sum_j a_j m_j\right)^T x+b.
```

Define

```math
m_{\rm eff}=\sum_j a_jm_j.
```

Then the entire multi-branch neuron is just one affine linear unit. The implementation verifies this identity with a maximum absolute floating-point error of `1.7763568394002505e-15` for the committed gate.

This is a useful negative result: branch multiplicity plus linear filtering alone is insufficient to justify the two-layer computational interpretation.

### 4.2 Nonlinear branches are a shallow network

For arbitrary pointwise `psi`,

```math
h_j(x)=\psi(m_j^Tx),
```

and

```math
v(x)=a^T h(x)+b.
```

This is exactly the forward equation of a one-hidden-layer network whose hidden units are branch subunits. The code compares the branch implementation to an explicit shallow-network expression and obtains zero numerical difference in the committed gate.

The equivalence is architectural. It does not claim that dendrites and artificial hidden units have identical biophysics or learning rules.

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

The implementation verifies the direct branch computation against the quadratic form with maximum absolute error `4.263256414560601e-14`.

This special case provides a transparent interpretation: the neuron can weight energy in several directions and threshold the combination.

## 5. Experiments

All committed results use seed `17`. JSON receipts are written by the experiment scripts and checked into `results/`.

### 5.1 Gate 0 — Oja recovery

A five-dimensional Gaussian distribution is generated with a known covariance spectrum `[9.0, 3.0, 1.5, 0.7, 0.3]` in a random orthonormal basis. A single Oja unit is trained on 14,000 samples for three epochs. Performance is measured by sign-invariant cosine alignment with the analytic leading covariance eigenvector.

### 5.2 Gate 1 — algebraic identities

Random inputs, branch filters, and output weights are sampled once. We measure maximum absolute errors for linear collapse, explicit shallow-network equivalence, and the square/quadratic-form identity. These are implementation checks of algebraic statements rather than empirical hypotheses.

### 5.3 Gate 2 — variance-only classification

Two orthonormal task modes are generated in six dimensions. Both classes remain centered near zero. Class 0 has high variance along mode 0 and low variance along mode 1; class 1 reverses those variances. Background energy orthogonal to the two task modes is small.

Two branches receive different local streams, each dominated by one of the task modes. Each branch learns its direction using Oja's rule without access to class labels. A ridge readout is then fitted on three representations: raw input, linear Oja branch coordinates, and squared Oja branch coordinates.

The task is intentionally constructed so first-order sign information is unhelpful while second-order mode energy is diagnostic. Its purpose is to isolate the role of local nonlinearity rather than to benchmark general learning ability.

### 5.4 Gate 3 — differential decay

Two physical modes decay with rates `mu_slow=0.03` and `mu_fast=0.21`. Starting from equal amplitudes, the analytic relative amplitude is

```math
R(t)=\exp[(\mu_{fast}-\mu_{slow})t].
```

The same ratio is measured in a diagonal discrete-time system whose eigenvalues are `exp(-mu_slow)` and `exp(-mu_fast)`. This tests only the narrow claim that slower modes become purer *relative* to faster modes.

### 5.5 Gate 4 — statistical/physical mode alignment

The exploratory alignment gate asks whether recurrent physical dynamics can alter the covariance later seen by a local Oja learner.

A two-state stable physical system has eigenvalues `0.97` and `0.65` and an orthogonal input coupling `B`. External Gaussian statistics are chosen so their leading principal direction begins halfway between the two input-visible physical modes, giving maximum initial alignment near `sqrt(1/2)`.

The recurrent state evolves as

```math
d_{t+1}=Ad_t+Bx_t,
```

and is observed back in input coordinates as

```math
z_t=B^Td_t.
```

Because the first physical mode has much longer memory, its variance is preferentially accumulated. Oja is trained on `z_t`. A no-memory control sets `A=0`, in which case the observed statistics return to the externally imposed geometry.

This experiment is deliberately constructive. It asks whether the mechanism can exist, not whether arbitrary dynamics automatically align modes.

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
| Gate 3 slow/fast ratio at `t=5` | `2.4596031112` |
| Gate 3 slow/fast ratio at `t=10` | `6.0496474644` |
| Gate 3 slow/fast ratio at `t=20` | `36.5982344437` |
| Gate 3 analytic/sim max error | `7.1054273576e-15` |
| Gate 4 fixed alignment | `0.7221832493` |
| Gate 4 no-memory alignment | `0.7221787537` |
| Gate 4 recurrent alignment | `0.9990103496` |
| Gate 4 alignment change | `+0.2768271003` |

### 6.1 Spectral nonlinear features solve the intended second-order task

The raw linear baseline (`49.4%`) and the identity-branch model (`50.775%`) remain at chance, whereas the square-branch model reaches `93.5%`. Because the two Oja filters align almost perfectly with the locally dominant modes, the result has a simple interpretation: local unsupervised learning recovers two useful spectral coordinates, but they become class-informative only after a nonlinear energy transform.

The result therefore supports the specific chain

```text
local principal-mode learning -> branch nonlinearity -> simple pooled decision.
```

It does not imply that Oja learning is generally sufficient to discover arbitrary hidden features.

### 6.2 Differential decay provides a precise weak form of mode purification

The slow/fast ratio increases from `2.46` at `t=5` to `36.60` at `t=20`, and the simulated values match the analytic ratio to floating-point precision. This validates only the relative-decay statement. A passive stable system can make one existing mode dominate relative to another because one decays more slowly; it does not choose an arbitrary desired covariance mode.

### 6.3 Physical recurrence can create a statistical alignment signal

The fixed external statistics have maximum alignment `0.722183` to the physical eigenmodes. The no-memory control is essentially identical at `0.722179`. With recurrence, alignment rises to `0.999010`.

The mechanism is not mysterious: the slow physical state accumulates variance, so the covariance of the observed recurrent state becomes dominated by the slow physical eigenmode; Oja then tracks that changed covariance. In this constructed case, physical dynamics therefore write a signature of their own eigenstructure into the statistics available to a local learner.

This is the most interesting result in the repository, but also the easiest to overstate. We engineered a stable recurrent system with unequal decay rates and observed its state. We did **not** demonstrate morphological growth, synaptic-to-dendritic co-adaptation, or a biological neuron solving a joint eigenproblem.

## 7. Discussion

AnttisNeuron clarifies a conceptual stack that is easy to blur together.

First, Oja learning and branch filtering are both linear operations with respect to a fixed state. Their composition can produce useful selectivity and temporal memory, but without a local branch nonlinearity a static multi-branch neuron still collapses to one linear map.

Second, once local nonlinearity occurs before pooling, branches genuinely become hidden computational subunits in the ordinary shallow-network sense. The square special case makes this particularly transparent: the cell is a weighted detector of energy in learned directions.

Third, physical dynamics can influence learning without backpropagating through the physical system. Gate 4 gives a simple example: recurrence changes the covariance of what is observed; a local Oja learner adapts to that changed covariance. In this sense, the physical substrate participates in defining the statistical learning problem.

That observation suggests a stronger future object. Suppose the learned weights alter branch drive, branch geometry or dynamics alter the state statistics, and an output/homeostatic boundary adapts to the resulting activity. A stationary configuration would then be a coupled fixed point, not merely a weight vector. Schematically,

```math
C(A,\theta)w=\lambda w,
```

along with constraints on branch dynamics and output activity. The current repository tests one arrow in that loop—physical dynamics changing covariance before Oja learning—not the full closed loop.

This distinction matters for novelty. The two-layer-neuron architecture is established. Oja/PCA learning is established. Stable modes and differential decay are established. The potential contribution is the explicitly testable synthesis: a neuron-like system in which local statistical modes are learned, physical dynamics reshape those statistics, and local nonlinearities make mode combinations computationally usable.

## 8. Limitations and falsification criteria

### 8.1 Minimal rather than biophysical branches

`LinearBranch` is a generic discrete-time stable state-space system. It omits cable geometry, ion channels, conductances, synaptic kinetics, spatial branch topology, and stochastic membrane processes. The `nmda` function is only a smooth supralinear toy activation. Results should not be read as predictions of a specific biological neuron.

### 8.2 Oja learning is local PCA, not a full representation learner

A single Oja unit is biased toward one dominant covariance direction. Multiple branches in Gate 2 receive different local statistical environments so that they learn different modes. A model in which several branches observe the same distribution requires competition, deflation, Sanger's rule, or another diversity mechanism if the goal is a principal subspace.

### 8.3 The nonlinear benchmark is constructed

Gate 2 is designed so that squared mode energy is the right representation. This is an explanatory benchmark, not an unbiased model-selection competition. The important controls are that linear raw and linear branch representations fail while the nonlinear mode-energy representation succeeds.

### 8.4 Gate 4 is an existence proof

The alignment experiment deliberately provides a recurrent physical state with unequal decay constants and then observes that state. Positive alignment is therefore evidence that the proposed causal path is mathematically possible in the model, not evidence that biological dendrites use it.

A stronger future test should allow several physical modes, local nonlinearities, and perhaps geometry parameters to adapt under a rule that never explicitly rewards eigenvector alignment. The key falsification question would then be whether alignment increases above matched controls without being directly optimized.

### 8.5 No AIS adaptation in v0

The current output readout does not implement activity-dependent AIS location, length, conductance, or dendritic load matching. Any proposed statistical-mode ↔ physical-mode ↔ output-boundary fixed point remains a hypothesis.

### 8.6 Predeclared interpretation rule

A negative statistical/physical alignment result is scientifically valid. CI is not permitted to encode "alignment must increase" as a software correctness criterion. The current Gate 4 test checks determinism, finiteness, and valid bounds only. This is intended to prevent result hacking as the experiment evolves.

## 9. Next experiments

The strongest next sequence is:

1. replace the two-state Gate 4 system with a branched cable/graph operator whose eigenmodes are not coordinate axes;
2. expose only bounded local measurements rather than the full recurrent state;
3. compare fixed Oja, Sanger-style multi-mode learning, and branch-specific local statistics;
4. add a genuine slow structural variable that alters `A` or the coupling matrix rather than merely filtering observations;
5. add a homeostatic output threshold/AIS-like variable and look for a coupled fixed point;
6. use held-out perturbations to distinguish real alignment from a parameter-specific covariance trick.

The decisive future result would not be "a slow mode wins"—that is expected. It would be spontaneous improvement of statistical-to-physical mode alignment under a local adaptation law that was not explicitly told to maximize alignment.

## References

1. Oja, E. (1982). *A simplified neuron model as a principal component analyzer*. Journal of Mathematical Biology, 15, 267–273. DOI: 10.1007/BF00275687.
2. Poirazi, P., Brannon, T., & Mel, B. W. (2003). *Pyramidal neuron as two-layer neural network*. Neuron, 37, 989–999. DOI: 10.1016/S0896-6273(03)00149-1.
3. London, M., & Häusser, M. (2005). *Dendritic computation*. Annual Review of Neuroscience, 28, 503–532. DOI: 10.1146/annurev.neuro.28.061604.135703.
4. Gidon, A., et al. (2020). *Dendritic action potentials and computation in human layer 2/3 cortical neurons*. Science, 367, 83–87. DOI: 10.1126/science.aax6239.
5. Leterrier, C. (2018). *The Axon Initial Segment: An Updated Viewpoint*. Journal of Neuroscience, 38, 2135–2145. DOI: 10.1523/JNEUROSCI.1922-17.2018.
6. Aizenbud, I., Yoeli, D., Beniaguev, D., de Kock, C. P. J., London, M., & Segev, I. (2026). *Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons*. Proceedings of the National Academy of Sciences, 123, e2533168123. DOI: 10.1073/pnas.2533168123.
