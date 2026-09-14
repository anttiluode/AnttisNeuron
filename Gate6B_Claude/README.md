# Gate 6B (Claude) — what explains Gate 6, and what predicts it

> Two different questions. Kept separate on purpose.

This folder is a holding pen, not yet integrated into the package. It was
built and verified in a separate sandboxed clone of `main` at `6727d06`,
against the real `anttis_neuron` / `experiments.gate6_many_worlds` code —
imported and called as-is, nothing reimplemented.

## Status: not yet real

Dropped in here as four loose files, these don't run. `gate6b_feature_ranking.py`
imports `anttis_neuron.*` and `experiments.gate6_many_worlds`, neither of which
exist relative to this folder. To make it real, either:

```bash
git am Gate6B_Claude/gate6b.patch     # applies cleanly on top of main at 6727d06
```

or move the three files by hand to where the patch puts them:

```
Gate6B_Claude/gate6b_feature_ranking.py  →  experiments/gate6b_feature_ranking.py
Gate6B_Claude/test_gate6b.py             →  tests/test_gate6b.py
Gate6B_Claude/gate6b.json                →  results/gate6b.json
```

Then `pytest -q` and `python experiments/gate6b_feature_ranking.py --worlds 24 --out results/gate6b.json`.

## What's here

| file | what it is |
|---|---|
| `gate6b_feature_ranking.py` | the experiment: `run(seed, n_worlds)`, same shape as Gate 6/5B |
| `test_gate6b.py` | determinism, bounds, and a frozen-receipt test |
| `gate6b.json` | the frozen output — real, not illustrative |
| `gate6b.patch` | `git format-patch` output; applies the above three files in one shot |

## Precondition, checked before trusting anything else

`adaptive_minus_frozen` and `adaptive_minus_shuffled` were re-derived for all
24 worlds from scratch and required to match `results/gate6.json` exactly.
They do — `reproduces_gate6_receipt` in `gate6b.json` records
`max_mismatch_delta_frozen: 0.0`, `max_mismatch_delta_shuffled: 0.0`. Found and
fixed one real bug getting there (wrong seed into `_train_conditions`, caught
because the shuffled comparison stopped matching while frozen still did).

## Result spine

**Explains the outcome** — Gate 5B's eigenvalue/eigenvector hybrid-operator
split, run for the first time across all 24 worlds instead of one hand-built
tree:

| | R² vs `Δ_frozen` | R² vs `Δ_shuffled` |
|---|---:|---:|
| eigenvector-only hybrid gain | **0.989** | **0.852** |
| eigenvalue-only hybrid gain | 0.302 | 0.177 |

Sign of the eigenvector-only gain matches `Δ_frozen`'s sign in **21/24**
worlds; eigenvalue-only matches in 13/24 — a coin flip. Mean
`|eigenvector-only gain|` is 5.6× mean `|eigenvalue-only gain|`. Holds
excluding the three biggest-swing worlds (R²=0.851 remaining). Gate 5B's
mechanism — local growth rotates the eigenbasis, that's most of the effect —
was not a property of the one lucky world it was built on.

**Predicts the outcome in advance** — six features knowable before running any
adaptation, ranked against `Δ_frozen` the way Aizenbud et al. rank
morphological features against FCI:

| feature | R² |
|---|---:|
| initial alignment (frozen operator) | 0.384 |
| tree depth | 0.076 |
| max port→sensor path length | 0.069 |
| mean port→sensor path length | 0.056 |
| spectral gap (λ₁−λ₂, frozen) | 0.002 |
| spectral gap ratio (λ₁/λ₂, frozen) | 0.002 |

Best pair (spectral gap + initial alignment): R²=0.451 — barely above initial
alignment alone. Spectral gap, the nearest analogue to "total dendritic area,"
carries essentially nothing here. The capacity/depth pair that ranks Aizenbud's
morphological features does not replicate as a predictor of which of these 24
worlds Gate 5's rule will help.

## Interpretation rules

1. The two tables above answer different questions. High R² in the first does
   not make the second table's low numbers a mistake.
2. Post-adaptation features (both tables' non-bracketed rows above the split)
   require having already run the adaptation. They explain; they cannot be
   used to screen a new world in advance.
3. Initial alignment predicting `Δ_frozen` is not a structural discovery —
   its slope is negative (−0.42): less-aligned starting points have more room
   to move, in either direction. Closer to regression toward the mean than to
   a finding.
4. n=24 worlds and six candidate features is close to the edge of what will
   not overfit. The best pair barely beating the best single feature is the
   evidence that it isn't hiding a stronger story, not a reason to try a
   third feature.
5. This receipt was produced outside the repo's own CI, in a sandboxed clone.
   Treat it as reproducible-once, not yet frozen the way Gates 0–7 are, until
   it's actually merged and re-run through GitHub Actions.

## Literature anchor

Aizenbud I, et al. (2026), *Dendritic morphology and synaptic nonlinearities
enhance functional complexity in human cortical neurons*. DOI
`10.1073/pnas.2533168123` — the source of the feature-ranking method this
borrows (single-feature R² first, pairs checked but not chased).
