# AnttisNeuron

> Wheel re-invented, but with the axle instrumented.

**AnttisNeuron** is a small falsifiable study of one specific synthesis: a local Oja learner selects a statistical input mode, stable branched dynamics filter those statistics through time, branch-local nonlinearities provide computational subunits, and slow local structural changes can alter the physical operator that the learner observes.

> **When can local statistics and a physical branched operator reshape one another, and when does that idea fail?**

See [PAPER.md](PAPER.md) for the Gates 0–6 argument, [[index.html](index.html)](https://anttiluode.github.io/AnttisNeuron/) for the visual companion, and `results/gate*.json` for frozen receipts. Gate 7 is currently an executable constructive extension of Gate 3 rather than a claim about a biological growth sensor.

## Current result spine

| Gate | Result |
|---|---:|
| 0 | Oja leading-PC alignment **0.995595** |
| 1 | linear collapse error **1.78e-15**; quadratic identity error **4.26e-14** |
| 2 | raw linear **0.4940**; square-branch **0.9350** |
| 3 | slow/fast relative ratio at t=20 **36.5982×** |
| 4 | recurrent alignment **0.722183 → 0.999010** |
| 5 | frozen → local-adaptive **0.782185 → 0.900191**; shuffled **0.759219** |
| 5B | eigenvalue-only **0.801087**; basis-only **0.892401**; full **0.900191** |
| 6 | local beats frozen **11/24**; mean Δ **+0.000498** |
| 6 | local beats shuffled **9/24**; mean Δ **−0.003074** |
| 7 | modal gap **0.02 → 0.16** gives required growth **74 → 10**; equal-persistence control never purifies |

Gate 5 is a strong synthetic success. Gate 5B shows that most of its gain follows **reorientation of the physical basis**, not an eigenvalue/time-scale change alone. Gate 6 then freezes the same rule across 24 deterministic branched worlds and finds essentially no aggregate advantage. That failure is retained rather than tuned away.

Gate 6 uses a basis-invariant score: normalized projection of the learned Oja direction into the span of the three slowest visible non-uniform physical modes. This replaced an unstable individual-eigenvector score after a near-degenerate eigenspace exposed arbitrary basis rotation. Because the scored subspace is 3-D inside a 4-D sensor space, interpret Gate 6 from **within-world deltas**, not the high absolute projection values.

Gate 7 returns to the original mode-purification intuition and asks a smaller question: if a designated target mode is more persistent than a distractor, can added serial path be treated as a finite purification budget? With a 95% target, doubling the modal gap repeatedly reduces the required integer path from **74 → 37 → 19 → 10**. At fixed gap, increasing distractor amplitude **0.5 → 1 → 2** increases required path **10 → 19 → 28**. When both modes have exactly equal persistence, purity remains **0.5 after 300 path units**. These results match the closed-form differential-decay law exactly by construction. The target identity and purity readout are oracle/developmental inputs, so Gate 7 does **not** yet explain biological growth or learning.

## Run it

```bash
python -m pip install -e ".[test]"
pytest -q
python experiments/gate5b_spectral_audit.py --out results/gate5b.json
python experiments/gate6_many_worlds.py --worlds 24 --out results/gate6.json
python experiments/gate7_growth_to_purity.py --out results/gate7.json
```

CI tests determinism, stability, common tapes, exact shuffled-credit update multisets, basis-rotation invariance, frozen Gate-7 receipt identity, and earlier gate regressions. CI never requires a preferred scientific delta where a gate is empirical.

## Biological anchors, not validation

Aizenbud et al. (2026) report that dendritic morphology and nonlinear synaptic integration jointly shape modeled single-neuron functional complexity; geometry carries information beyond branch count alone. Leterrier (2018) reviews the AIS as a plastic determinant of excitability whose position and composition adapt to physiological conditions. These papers motivate asking whether real morphology, membrane area, electrical load, and an AIS-like output boundary can eventually replace Gate 7's dimensionless path and oracle purity signal. They do not validate the synthetic Gate-5 rule or the Gate-7 growth interpretation.

## Interpretation rules

1. Oja is PCA-like, not ICA.
2. Linear branches do not create nonlinear computational depth.
3. “Mode purification” means relative differential decay here.
4. Gate 5 is synthetic and local-credit dependent.
5. Gate 5B’s hybrid operators are diagnostics, not literal cables.
6. Gate 6 is a generalization failure, not a software failure.
7. Near degeneracy, subspaces are the stable physical object—not arbitrary eigenvector labels.
8. Gate 7 is a constructive growth-depth law with dimensionless path units and oracle target purity, not a learned or biological growth mechanism.
9. AIS/load adaptation and a biophysical mapping from path to membrane area remain **future work**.

## Literature anchors

- Oja E. (1982), *A simplified neuron model as a principal component analyzer*. DOI `10.1007/BF00275687`.
- Poirazi P, Brannon T, Mel BW. (2003), *Pyramidal neuron as two-layer neural network*. DOI `10.1016/S0896-6273(03)00149-1`.
- London M, Häusser M. (2005), *Dendritic computation*. DOI `10.1146/annurev.neuro.28.061604.135703`.
- Gidon A, et al. (2020), *Dendritic action potentials and computation in human layer 2/3 cortical neurons*. DOI `10.1126/science.aax6239`.
- Leterrier C. (2018), *The Axon Initial Segment: An Updated Viewpoint*. DOI `10.1523/JNEUROSCI.1922-17.2018`.
- Aizenbud I, et al. (2026), *Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons*. DOI `10.1073/pnas.2533168123`.
