# A Spectral Two-Layer Neuron: Local Statistical Modes, Physical Branch Dynamics, and Falsifiable Structural Adaptation

Antti Luode  
Working manuscript — 14 September 2026

## Abstract

Nonlinear dendritic subunits can make the input-output function of a single pyramidal neuron resemble a small multilayer network. That architectural observation is established prior work. Here we study a narrower synthesis: what happens when locally learned statistical directions interact with a physical branched operator that filters activity through time and can itself change slowly from local signals?

We separate exact algebra from synthetic experiments. Purely linear branches collapse exactly to one affine filter (maximum numerical error `1.78e-15`), whereas nonlinear branches are exactly a one-hidden-layer network under the static abstraction. With a square branch nonlinearity, the model is exactly a quadratic classifier `x^T Q x` (maximum error `4.26e-14`). On a variance-only task, raw linear and linear-branch controls obtain `49.4%` and `50.8%` held-out accuracy, while two locally Oja-learned mode detectors followed by square responses obtain `93.5%`.

A differential-decay experiment verifies the narrow sense in which physical dynamics can increase relative slow-mode purity. A constructive recurrent branch raises statistical-to-physical alignment from `0.722183` to `0.999010`. Gate 5 then replaces that hand-built system with an 11-node branched graph cable. Edge conductances adapt using only local squared voltage differences. In that construction, held-out Oja-to-physical alignment rises from `0.782185` to `0.900191`, while an exact-update-multiset shuffled-credit control obtains `0.759219`.

Gate 5B asks what moved. Diagnostic hybrid operators show that adapted eigenvectors with frozen eigenvalues reproduce most of the gain (`0.892401`), whereas adapted eigenvalues with the frozen basis do not (`0.801087`). Thus the original Gate-5 effect is mainly associated with reorientation of the physical basis rather than a simple change of time constants.

Gate 6 then freezes the Gate-5 rule and evaluates 24 deterministic branched worlds. Because individual eigenvectors are not stable objects inside degenerate eigenspaces, the final Gate-6 metric is the basis-invariant projection into the span of the three slowest visible non-uniform physical modes. Under that corrected metric, local adaptation beats frozen in `11/24` worlds; mean local-minus-frozen alignment is only `+0.000498`, and mean local-minus-shuffled is `-0.003074`. The unchanged rule therefore does not generalize as a universal organizing principle. Strong positive and negative worlds remain, converting the next question from “does it work?” to “what physical conditions make it work?”

The study supports a compact synthetic mechanism—statistics can select directions, dynamics can reshape what is observable, and local structural change can sometimes reorient the operator toward those statistics. It does **not** establish that biological dendrites implement this rule or generally optimize eigenmode alignment.

## 1. Introduction

The phrase “a single neuron is a two-layer neural network” is not new. Poirazi, Brannon, and Mel (2003) showed that a detailed pyramidal-cell model could be approximated by nonlinear dendritic subunits whose outputs are pooled at the soma. Dendritic computation has since been studied across morphology, synaptic placement, active conductances, and local nonlinear events. Human cortical neurons add another important context: recent detailed modeling suggests that dendritic morphology and nonlinear synaptic integration can substantially alter single-cell input-output complexity (Aizenbud et al., 2026).

A second classical line of work concerns local statistical learning. Oja’s normalized Hebbian rule gives a simple update whose weight vector approaches a leading covariance eigenvector under standard conditions. A single Oja learner therefore does not “select one wire”; it extracts one dominant direction through a multidimensional local input space.

A third ingredient is physical dynamics. A dendritic tree is not merely a collection of arbitrary learned vectors. Geometry, passive conductance, membrane properties, synaptic nonlinearities, and observation location constrain how activity propagates. In a linearized model those constraints define an operator with modes and time scales. The operator changes what statistics are visible downstream.

This motivates a feedback picture:

```text
external statistics
        ↓
local statistical selection
        ↓
physical branched dynamics
        ↓
locally observed activity
        ↓
slow structural change
        ↺
```

The attractive intuition is that a physical branch could come to “fit” statistically important activity. That statement is easy to overstate. It could mean only that one decay rate became slower; it could mean that the physical eigenbasis actually changed; or it could be a numerical artifact caused by arbitrary eigenvector orientation in a nearly degenerate eigenspace. AnttisNeuron therefore proceeds through gates designed to separate these possibilities.

The central discipline of the repository is that a failed scientific gate is not a failed CI job. Software tests enforce algebra, determinism, stability, matched controls, and reproducibility. They do not require the preferred biological story to be true.

## 2. Biological and mathematical anchors

### 2.1 Oja learning

Oja (1982) introduced a normalized Hebbian rule that behaves as a principal-component analyzer. In the single-unit setting used here, the relevant result is limited: the weight vector approaches one leading covariance direction. Multiple identical Oja units are not assumed to discover a full PCA basis without competition or decorrelation.

### 2.2 Dendritic nonlinear subunits

Poirazi et al. (2003) mapped a detailed CA1 pyramidal neuron to a two-layer abstraction in which local dendritic nonlinearities provide hidden subunits and the soma pools them. London and Häusser (2005) reviewed a broad range of dendritic computations, and Gidon et al. (2020) reported nonlinear dendritic events in human layer 2/3 cortical neurons capable of supporting linearly nonseparable computations.

Aizenbud et al. (2026) provide a useful contemporary constraint. Their Functional Complexity Index compares how difficult detailed biophysical neuron models are for a fixed deep network to approximate. In their model set, human cortical pyramidal neurons are more functionally complex than rat counterparts. Morphology is a major contributor: total dendritic area is the strongest single morphological correlate they report, while combining area with longest bifurcation-branch length explains substantially more variance than branch count alone. They also find that stronger and steeper NMDA-mediated nonlinear integration further increases model complexity.

This matters here because it warns against a simplistic “more branches = more computation” story. Geometry, distributed extent, electrical separation, and nonlinear integration interact.

### 2.3 Dendritic load and the axon initial segment

The axon initial segment (AIS) is the specialized proximal axonal compartment at which action potentials are normally initiated. Leterrier (2018) reviews how AIS channel composition and position shape excitability and how AIS structure can adapt under developmental and physiological conditions. The effect of AIS geometry on excitability is not independent of the rest of the neuron; dendritic morphology and electrical load matter to the whole-cell boundary problem.

Aizenbud et al. also make the load connection explicit in their modeling methods: somatic and axonal active conductance densities are normalized using conductance ratios that account for the electrical load imposed by the dendritic tree. That is a modeling choice rather than evidence for a specific biological AIS learning rule, but it reinforces an important physical point: dendrite and axon cannot always be treated as independent computational modules.

AnttisNeuron does **not** yet implement AIS plasticity. These biological results motivate a later, separately controlled output-boundary experiment rather than being used as retrospective validation of Gates 0–6.

## 3. Model

### 3.1 Local statistical direction

A local input vector is

```math
x(t)\in\mathbb{R}^{n}.
```

An Oja unit computes

```math
y(t)=w^T x(t)
```

and updates

```math
\Delta w=\eta y\left(x-yw\right).
```

Alignment metrics are sign-invariant because eigenvectors are defined only up to sign.

### 3.2 Generic dynamic branch

A minimal branch state evolves as

```math
d(t+1)=A d(t)+b y(t),
```

with local response

```math
r(t)=c^T d(t)
```

and branch nonlinearity

```math
q(t)=\psi(r(t)).
```

A soma-like readout pools branch outputs,

```math
v(t)=\sum_j a_j q_j(t)+b_0.
```

### 3.3 Static branch abstraction

For exact identities we use effective branch filters `m_j`:

```math
v(x)=\sum_j a_j\psi(m_j^Tx)+b_0.
```

If `psi(z)=z`, then

```math
v(x)=\left(\sum_j a_jm_j\right)^Tx+b_0,
```

so all static linear branches collapse to one affine filter.

If `psi(z)=z^2`,

```math
v-b_0=\sum_j a_j(m_j^Tx)^2
=x^T\left(\sum_j a_jm_jm_j^T\right)x.
```

Thus the static model is exactly a quadratic form.

### 3.4 Branched graph cable

Gate 5 uses a tree with eleven nodes and ten positive edge conductances. Let `W(g)` be the weighted symmetric adjacency matrix and

```math
L(g)=D(g)-W(g)
```

its graph Laplacian. The discrete stable leaky-diffusion operator is

```math
A(g)=I-dt\,[\ell I+\kappa L(g)].
```

State evolves as

```math
d_{t+1}=A(g)d_t+Bx_t.
```

Only four node voltages are exposed to the learner:

```math
z_t=S d_t\in\mathbb{R}^4.
```

The full physical state and physical eigenvectors remain hidden from Oja learning.

### 3.5 Local structural rule

For edge `e=(i,j)`, define

```math
q_e=\mathbb E[(d_i-d_j)^2].
```

A target `q_*` is measured once from the initial frozen structure. Conductance changes by

```math
g_e\leftarrow g_e\exp\left[\eta_g\left(\frac{q_e}{q_*}-1\right)\right],
```

followed by clipping and exact mean-conductance normalization.

The adaptor receives no eigenvectors, Oja weights, labels, alignment scores, or held-out metrics.

## 4. Gates 0–4: algebra and existence tests

### Gate 0 — Oja sanity check

A five-dimensional Gaussian with known covariance eigenvectors is sampled. Oja recovers the leading principal direction with cosine alignment `0.9955947725`.

### Gate 1 — exact algebra

The implementation verifies:

- linear branch collapse, maximum error `1.7763568394e-15`;
- explicit one-hidden-layer equivalence, error `0.0`;
- square-branch quadratic-form identity, error `4.2632564146e-14`.

### Gate 2 — variance-only classification

Two centered classes differ only in mode energy. Raw linear accuracy is `0.4940`; identity-branch accuracy is `0.50775`. Two Oja-learned local filters followed by squaring expose the second-order difference and achieve `0.9350` held-out accuracy. Mean learned-filter alignment with the generating modes is `0.999900`.

The point is not that Oja solves arbitrary nonlinear tasks. It is the specific pipeline:

```text
local spectral direction → local nonlinear feature → simple pooled decision
```

### Gate 3 — differential decay

Two modes decay at rates `0.03` and `0.21`. Starting with equal amplitudes, the slow/fast relative amplitude ratio reaches `36.5982344437` at `t=20`, matching the analytic expression to `7.11e-15`.

“Purification” here means **relative** dominance of the slow component. No passive system is claimed to amplify all absolute energy.

### Gate 4 — recurrent existence proof

A deliberately constructed two-state system has eigenvalues `0.97` and `0.65`. External statistics begin between its physical modes. Recurrent observation reshapes the covariance seen by Oja. Alignment changes from `0.7221832493` to `0.9990103496`; the no-memory control remains `0.7221787537`.

Gate 4 proves only that such coupling can exist.

## 5. Gate 5: alignment-blind structural adaptation

Gate 5 removes the hand-built two-state target. The 11-node tree receives four-dimensional inputs at four distal ports and exposes only four sensor voltages. Four covariance environments drive adaptation; five different covariance rotations are held out for evaluation.

Four conditions share the same input tapes and matched Oja initialization:

1. **Frozen:** all conductances remain fixed.
2. **Local:** each edge receives its own local voltage-difference signal.
3. **Shuffled credit:** every adaptation phase receives the exact multiset of local update values, randomly reassigned to the wrong edges.
4. **Uniform:** relative structure remains exactly frozen.

The shuffled control is important. It asks whether the local *placement* of credit matters, not merely whether adaptation introduces a distribution of conductance changes. The maximum update-multiset mismatch in the committed receipt is exactly `0.0`.

Held-out mean alignment is:

| condition | alignment |
|---|---:|
| frozen | `0.7821845260` |
| shuffled | `0.7592192537` |
| local | `0.9001914311` |

The local adaptive minimum across held-out environments is `0.8393687765`, versus `0.6063645741` frozen.

This is the strongest positive construction in the repository: an alignment-blind local structural rule improves held-out statistical-to-physical alignment in one branched operator without receiving the global alignment objective.

It remains a synthetic graph-diffusion result.

## 6. Gate 5B: spectral autopsy

The phrase “the dendrite grew to fit the mode” is stronger than Gate 5 alone justifies. A conductance change can alter eigenvalues while leaving the physical basis almost unchanged, or it can rotate the basis itself. Gate 5B separates these effects.

Let

```math
A_f=Q_f\Lambda_fQ_f^T
```

be the frozen operator and

```math
A_a=Q_a\Lambda_aQ_a^T
```

be the adapted operator. Two diagnostic hybrids are constructed:

```math
A_\lambda=Q_f\Lambda_aQ_f^T
```

and

```math
A_Q=Q_a\Lambda_fQ_a^T.
```

These are symmetric diagnostic counterfactuals, not claimed to correspond to realizable positive-conductance trees.

Held-out mean alignment is:

| operator | alignment |
|---|---:|
| frozen | `0.7821845260` |
| adapted eigenvalues, frozen basis | `0.8010872578` |
| adapted basis, frozen eigenvalues | `0.8924009609` |
| fully adapted | `0.9001914311` |

The slow three-dimensional state subspaces differ by principal angles approximately `4.06°`, `5.26°`, and `16.42°`.

A four-way frame comparison is also informative:

| statistical direction | physical modes | mean alignment |
|---|---|---:|
| frozen Oja | frozen | `0.782185` |
| frozen Oja | adapted | `0.903345` |
| adapted Oja | frozen | `0.748890` |
| adapted Oja | adapted | `0.900191` |

In this construction the physical substrate moved toward the statistical frame much more strongly than the learner simply moved toward an unchanged physical frame. That makes “basis reorientation” a fair description of Gate 5. It still does not imply biological dendritic morphology performs the same operation.

## 7. Gate 6: 24-world generalization test

### 7.1 Predeclared variation

Gate 6 freezes the Gate-5 rule and hyperparameters and generates 24 deterministic worlds varying:

- tree topology;
- input-port placement;
- sensor placement;
- covariance rotations and spectra;
- random seed.

The same frozen/local/exact-multiset-shuffled/uniform controls are retained. Failures are recorded rather than retuned.

### 7.2 Why individual eigenvectors were the wrong metric

An early Gate-6 evaluation matched Oja directions to individual slow visible eigenvectors. That metric is numerically unstable when two or more eigenvalues are equal or nearly equal: an eigensolver is free to return any orthonormal basis spanning the same physical eigenspace. A microscopic perturbation can therefore rotate the reported eigenvectors while the physical subspace is unchanged.

The final Gate-6 metric uses the **span** of the three slowest visible non-uniform modes. Their sensor-space projections are orthonormalized with an SVD, and the normalized score for an Oja vector `w` is

```math
s(w,U)=\frac{\|U^Tw\|_2}{\|w\|_2},
```

where columns of `U` form an orthonormal basis for the selected visible physical subspace. The score is invariant to sign flips, mode permutations, and arbitrary rotations within that span. A regression test explicitly verifies this invariance.

Because the observation space is four-dimensional while the scored physical span is three-dimensional, absolute projection scores are expected to be high. Scientific interpretation therefore focuses on **within-world condition deltas**, not the absolute mean score.

### 7.3 Corrected result

The frozen 24-world receipt reports:

| measurement | result |
|---|---:|
| frozen mean alignment | `0.9289695876` |
| local mean alignment | `0.9294679130` |
| shuffled mean alignment | `0.9325420371` |
| local beats frozen | `11 / 24` |
| local beats shuffled | `9 / 24` |
| mean local − frozen | `+0.0004983254` |
| median local − frozen | `−0.0001175012` |
| first quartile local − frozen | `−0.0099764339` |
| minimum local − frozen | `−0.3012406816` |
| mean local − shuffled | `−0.0030741242` |
| median local − shuffled | `−0.0015638860` |
| minimum local − shuffled | `−0.3030618501` |

The aggregate effect is therefore approximately null versus frozen and slightly negative versus shuffled. The unchanged Gate-5 rule does **not** generalize as a universal organizing rule across these worlds.

The distribution is highly heterogeneous. For example, one world improves by roughly `+0.344` versus frozen, while another degrades by roughly `−0.301`. This heterogeneity is scientifically more interesting than forcing the mean positive. It provides a compact dataset for asking which structural and electrical conditions permit or prevent useful operator reorientation.

## 8. What the gates establish

The sequence now supports the following bounded claims:

1. A locally learned statistical direction plus branch-local nonlinearity can expose second-order information unavailable to a linear readout in the constructed task.
2. Physical recurrence can reshape the covariance seen by a local learner.
3. A purely local structural rule can, in at least one branched system, improve held-out statistical-to-physical alignment without being given that objective.
4. In the Gate-5 construction, most of that improvement is associated with physical-basis reorientation rather than eigenvalue change alone.
5. The same local rule does **not** robustly improve an arbitrary family of branched worlds.
6. Near degenerate spectra, the physically meaningful comparison is an eigenspace/subspace comparison rather than arbitrary individual eigenvector labels.

The sequence does **not** establish:

- that biological dendrites use Oja’s rule;
- that biological structural plasticity follows the Gate-5 conductance update;
- that real dendrites explicitly optimize eigenmodes;
- that the synthetic graph cable captures NMDA spikes or detailed compartmental biophysics;
- that Gate-5 adaptation is universally beneficial;
- that AIS plasticity implements a load-matching algorithm.

## 9. Next experiments

### 9.1 World anatomy: explain the Gate-6 heterogeneity

The immediate next step should not change the learner. It should characterize the worlds already produced. Candidate explanatory variables include graph depth, branching geometry, port-to-sensor distances, effective electrical distances, observable slow-mode conditioning, spectral gaps, baseline ceiling, local energy heterogeneity, and conductance-change heterogeneity.

This analysis is necessarily **post hoc** because the Gate-6 outcomes have already been observed. It should therefore be used to generate a small number of candidate mechanisms, followed by a fresh confirmatory world suite.

The Aizenbud et al. morphology result suggests a useful caution: branch count alone is unlikely to be the right explanatory variable. Extent, area/load proxies, and how branching is distributed may matter more.

### 9.2 Output-boundary/AIS surrogate

Only after the dendritic-world analysis should the model add an AIS-like boundary. A minimal future experiment would introduce a slowly adapting output excitability or threshold variable driven by a dendritic-load signal and compare it against a fixed-boundary control across a fresh morphology suite.

This would test a narrow physical question inspired by real AIS plasticity and dendritic-load coupling: can slow adaptation at the output boundary stabilize useful input-output behavior as the upstream branched load changes? It should not be called a detailed AIS model.

### 9.3 More realistic dendritic nonlinearities

The current graph cable is linear between structural updates. The repository’s earlier static `NMDA` activation is only a toy supralinear function. A later biophysical gate should distinguish passive cable effects from voltage-dependent local nonlinearities rather than mixing them prematurely.

## 10. Reproducibility and scientific policy

All random generators are seeded. Gate 6 worlds are deterministic. Common tapes are used across controls. The shuffled structural condition receives the exact local update-value multiset on every phase. Operators are checked for finiteness and stability.

CI treats engineering failures and scientific outcomes differently:

- nondeterminism, invalid operators, broken controls, mismatched tapes, and algebraic failures are **red**;
- a negative scientific delta is still a valid, recorded scientific result.

This distinction is why Gate 6 is useful: the code passed while the broad hypothesis weakened.

## 11. Conclusion

The useful object in AnttisNeuron is not “a neuron secretly does PCA” and not “a dendrite is literally an eigenmode purifier.” The more defensible object is a feedback system in which statistics, physical dynamics, bounded observation, local nonlinearities, and slow structural change can alter one another.

Gate 5 shows that local structural activity can produce a strong alignment effect in one constructed branched substrate. Gate 5B shows that this effect is dominated by physical-basis reorientation. Gate 6 then demonstrates the limitation: the same rule is not generally beneficial across arbitrary branched worlds.

That failure sharpens the research program. The next problem is not to search for another rule that makes the average green. It is to identify the physical conditions under which local structural credit becomes meaningful, then test those conditions on fresh worlds. Only after that should dendritic load be coupled to an adaptive AIS-like output boundary.

## References

1. Oja E. (1982). A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology* 15:267–273. DOI: `10.1007/BF00275687`.
2. Poirazi P, Brannon T, Mel BW. (2003). Pyramidal neuron as two-layer neural network. *Neuron* 37:989–999. DOI: `10.1016/S0896-6273(03)00149-1`.
3. London M, Häusser M. (2005). Dendritic computation. *Annual Review of Neuroscience* 28:503–532. DOI: `10.1146/annurev.neuro.28.061604.135703`.
4. Gidon A, et al. (2020). Dendritic action potentials and computation in human layer 2/3 cortical neurons. *Science* 367:83–87. DOI: `10.1126/science.aax6239`.
5. Leterrier C. (2018). The Axon Initial Segment: An Updated Viewpoint. *Journal of Neuroscience* 38:2135–2145. DOI: `10.1523/JNEUROSCI.1922-17.2018`.
6. Aizenbud I, Yoeli D, Beniaguev D, de Kock CPJ, London M, Segev I. (2026). Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons. *PNAS* 123:e2533168123. DOI: `10.1073/pnas.2533168123`.
