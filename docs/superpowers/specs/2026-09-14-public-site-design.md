# AnttisNeuron public site design

Date: 2026-09-14
Status: proposed for implementation after user review
Branch: `site/public-front-door-v1`

## Purpose

Build a public-facing GitHub Pages site for AnttisNeuron that can receive viewers from videos and explain the project in under a minute without weakening the scientific discipline of the repository.

The site is not a replacement for `README.md` or `PAPER.md`. The README remains the compact research ledger; the paper remains the detailed argument. The site is the visual front door.

The site must preserve a strict distinction between:

- established neuroscience or prior work;
- exact identities proved by the code;
- synthetic experimental results produced by AnttisNeuron;
- open hypotheses that have not been established.

## Audience

Primary audience:

- curious viewers arriving from Antti's videos;
- technically interested readers who may not know the mathematics;
- researchers or engineers who want to see the actual receipts after the conceptual picture catches their interest.

Secondary audience:

- GitHub visitors who encounter the repository directly;
- people following the broader series of experimental repos.

The site should work for a reader who understands the words “neuron”, “signal”, “filter”, and “spike” but does not know linear algebra.

## Voice

The public page should retain the project's informal personality without making scientific claims casually.

The framing line is:

> **Wild hand waving. Executable receipts underneath.**

The visual layer may be playful and provocative. The evidence layer must be precise.

The site must never imply that the strongest biological hypothesis is already established.

## Core explanatory model

The primary visual object is a left-to-right neuron-as-circuit flow:

```text
incoming statistics
      ↓
synapses / local selectivity
      ↓
dendritic branch dynamics
      ↓
branch-local nonlinearity
      ↓
soma / cable load
      ↓
AIS trigger boundary
      ↓
spike / no spike
```

Each stage has a short plain-English explanation and one of the four site status tags: `prior work`, `exact`, `synthetic`, or `open`.

### Stage 1 — incoming statistics

Plain English: repeated structure in the world produces recurring patterns in local input.

Status: `open` — this is the conceptual input side of the proposed model rather than a claim that AnttisNeuron has modeled a full sensory pathway.

### Stage 2 — synapses / Oja learning

Plain English: a population of synaptic weights can adapt toward a dominant covariance direction rather than selecting one physical input wire.

Status: `prior work` plus an executable sanity check in AnttisNeuron.

Do not describe one isolated synapse as independently performing PCA.

### Stage 3 — dendritic branch dynamics

Plain English: physical branch dynamics can preserve some components longer than others and reshape the statistics that remain observable.

Status: `synthetic`; the biological eigenmode-growth claim remains `open`.

Do not state that real dendrites are known to grow toward covariance eigenmodes.

### Stage 4 — branch-local nonlinearity

Plain English: nonlinearity before pooling creates genuine hidden computational subunits. Without it, static linear branches collapse to one effective linear filter.

Status: `exact` for the algebraic collapse/equivalence; biological dendritic nonlinearity itself is `prior work`.

### Stage 5 — soma / load

Plain English: branch events arrive through the physical cable with geometry-dependent transfer and jointly create the electrical state presented to the trigger zone.

Status: `open` in AnttisNeuron v1 because explicit soma/AIS load coupling has not yet been implemented; the general cable/circuit interpretation is prior work.

### Stage 6 — AIS trigger

Plain English: the axon initial segment is a stateful regenerative trigger region, not merely a memoryless ReLU. The public site may use the phrase “analog-to-event boundary” rather than “digital transistor”.

Status: `prior work` for the AIS's established biological role; explicit AnttisNeuron AIS dynamics remain `open`.

### Stage 7 — spike

Plain English: the continuous internal state becomes a discrete event that travels down the axon.

Status: `prior work`.

## Information architecture

The site is one scrollable page with five primary sections.

### 1. Hero: the mental picture

Content:

- project name `AnttisNeuron`;
- framing line `Wild hand waving. Executable receipts underneath.`;
- a one-sentence serious subtitle:
  `A falsifiable study of local spectral learning, physical dendritic dynamics, nonlinear branch computation, and the path toward a stateful spike trigger.`;
- the interactive neuron-as-circuit diagram;
- two calls to action: `See the evidence` and `Read the paper`.

The hero should communicate the whole conceptual chain without requiring equations.

### 2. What is actually being claimed?

A four-column or four-card claim ledger:

- Established biology / prior work
- Exact in this model
- Synthetic result here
- Open hypothesis

Representative entries:

Established biology / prior work:
- Oja-style learning can recover a leading covariance direction.
- Nonlinear dendritic subunits can support richer-than-linear computation.
- The AIS initiates and shapes action potentials and can adapt in composition and morphology.

Exact in this model:
- linear static branches collapse to one affine filter;
- square branch nonlinearities give a quadratic form.

Synthetic result here:
- nonlinear Oja-mode features solve the variance-only benchmark where linear controls remain near chance;
- recurrent physical dynamics can reshape observed statistics;
- Gate 5 local conductance adaptation improves held-out Oja-to-physical-mode alignment relative to frozen and exact-multiset shuffled-credit controls.

Open hypothesis:
- real dendritic morphology grows or reorganizes toward input statistical eigenmodes;
- dendrite/AIS co-adaptation forms a self-consistent input-to-spike circuit around recurring statistics.

### 3. Executable receipts

Show three headline experimental cards drawn from committed JSON data.

Gate 2:
- raw linear accuracy;
- identity-branch accuracy;
- square-branch accuracy;
- mean Oja filter alignment.

Gate 4:
- fixed alignment;
- no-memory alignment;
- recurrent alignment;
- alignment change.

Gate 5:
- frozen held-out alignment;
- local-adaptive held-out alignment;
- exact-multiset shuffled-credit alignment;
- adaptive-minus-frozen;
- adaptive-minus-shuffled;
- worst held-out frozen → adaptive;
- update-multiset mismatch.

The values must be loaded at runtime from `results/gate2.json`, `results/gate4.json`, and `results/gate5.json` copied into the published artifact. They must not be duplicated as authoritative constants in the site JavaScript.

If a result file or required field is missing, the affected card must visibly say `receipt unavailable` rather than displaying stale fallback numbers.

Each card links to the corresponding experiment source and JSON receipt on GitHub.

### 4. Wild hand waving / serious ledger

Purpose: connect the public video persona to the scientific artifact without making the paper sound apologetic.

Two adjacent panels:

`The hand-waving picture`
- synapses select recurring structure;
- branches act like physical filters;
- nonlinear convergence creates logic-like combinations;
- the soma presents a load;
- the AIS turns analog state into an event.

`What the repo has actually earned`
- exact algebraic identities;
- deterministic synthetic experiments;
- matched controls;
- machine-readable receipts;
- explicit negative claims and open questions.

This section may link to the video if a canonical video URL is present in repository metadata at implementation time. If there is no canonical URL, v1 simply omits the video button rather than displaying an empty placeholder.

### 5. Next falsifications

Display the research trajectory as a compact sequence:

```text
Gate 5B
mechanism audit
    ↓
eigenvalue/eigenvector swap controls
    ↓
environment-specificity matrix
    ↓
Gate 6
multi-seed / multi-topology / multi-sensor robustness
    ↓
AIS/load gate
explicit trigger node + transfer impedance + adaptive firing boundary
```

This section must clearly use future tense.

A final button links to `PAPER.md` on GitHub.

## Visual design

The site should look technical, dark, and slightly strange rather than corporate.

Constraints:

- pure HTML/CSS/vanilla JavaScript;
- no framework and no package manager;
- responsive down to phone width;
- no external web fonts required for correct rendering;
- no stock brain imagery;
- no fake microscopy or fake biological diagrams;
- reduced-motion mode must disable decorative animation;
- high contrast sufficient for normal reading;
- semantic headings and keyboard-focus states.

Suggested visual language:

- near-black background;
- pale neutral text;
- one electric accent and one warm accent;
- thin circuit traces connecting sections;
- the neuron diagram built from CSS/SVG-like HTML elements or inline SVG contained directly in the page;
- subtle pulse animations confined to decorative signal markers.

The neuron diagram should be abstract enough not to pretend anatomical precision.

## Interaction design

The primary interaction is inspection, not gamification.

On hover/focus/tap of a neuron stage:

- highlight that stage;
- display its plain-English explanation;
- display one of four status classes: `prior work`, `exact`, `synthetic`, `open`.

On Gate receipt cards:

- show values loaded from JSON;
- provide a short `what this means` sentence;
- provide a direct GitHub link to the experiment and result file.

No user data is collected and no analytics code is added in v1.

## Repository layout

Add:

```text
site/
  index.html
  styles.css
  app.js

.github/workflows/pages.yml

tests/test_site.py
```

The published Pages artifact contains:

```text
index.html
styles.css
app.js
results/gate2.json
results/gate4.json
results/gate5.json
```

Do not publish internal `docs/superpowers/` content as part of the Pages artifact.

## Deployment

Use GitHub Actions Pages deployment triggered on pushes to `main` and manual `workflow_dispatch`.

Workflow responsibilities:

1. check out the repository;
2. set up Python and install the project with test dependencies;
3. run `pytest -q` so publication is blocked by repository or site test failures;
4. run `actions/configure-pages`;
5. create a temporary Pages artifact directory;
6. copy `site/*` into its root;
7. copy only `results/gate2.json`, `results/gate4.json`, and `results/gate5.json` into `results/`;
8. upload the artifact with `actions/upload-pages-artifact`;
9. deploy with `actions/deploy-pages` using the required Pages permissions/environment.

The site must work when hosted under the repository path `/AnttisNeuron/`, so all internal asset and receipt paths must be relative rather than root-absolute.

## Testing

Add Python tests that do not require a browser or Node.

Required tests:

1. `site/index.html`, `site/styles.css`, and `site/app.js` exist.
2. The HTML references CSS/JS with relative paths.
3. The HTML contains the required page sections and claim-status language.
4. JavaScript names the three receipt files and does not contain hard-coded authoritative Gate 5 result values.
5. Gate 2, Gate 4, and Gate 5 receipts contain the exact fields consumed by the site.
6. The Pages workflow copies only the intended receipts and site assets.
7. No absolute `/styles.css`, `/app.js`, or `/results/...` paths are introduced.

Existing scientific tests remain unchanged except where a site-related regression requires shared fixture access.

## Source links and scientific grounding

The site should link to:

- `README.md`;
- `PAPER.md`;
- each highlighted experiment source;
- each highlighted JSON receipt.

The public copy must not introduce claims stronger than the README or paper.

For the AIS text, use conservative wording supported by Leterrier 2018 and the existing paper references: AIS ion channels generate and shape action potentials; excitability depends on AIS composition/position; AIS morphology can adapt; the consequences of AIS geometry depend on the broader somatodendritic morphology.

Do not describe chandelier-cell input as a proven Boolean veto gate.

Do not describe the 190 nm periodic scaffold as a demonstrated computation mechanism.

## OpenGraph and sharing metadata

Add static metadata for social sharing:

Title:
`AnttisNeuron — Wild hand waving, executable receipts`

Description:
`A visual, falsifiable study of spectral learning, dendritic computation, physical structure, and the path from analog branch dynamics to a spike.`

Include canonical GitHub repository URL as the fallback site identity until the deployed Pages URL is known.

A custom image is optional and is not required for v1.

## Failure behavior

The public site must fail visibly rather than silently when evidence cannot be loaded.

- JSON fetch failure: show `receipt unavailable` for that gate.
- malformed/missing field: show `receipt incomplete` for the affected card.
- JavaScript disabled: the page remains readable; the claim ledger and conceptual diagram remain present, while dynamic receipt values are replaced by explanatory static text.

## Non-goals for v1

Do not add:

- analytics;
- comments;
- login;
- backend services;
- WebGL;
- a scientific simulation in the browser;
- live editing of experiment parameters;
- an AI chat box;
- a fake interactive neuron whose visuals imply biological validation;
- a full redesign of the repository README;
- automatic literature fetching at page load.

## Success criteria

The v1 site is successful when:

1. a nontechnical viewer can state the proposed input→branch→trigger story after one minute;
2. every strong visual claim has an explicit status class;
3. the strongest numerical claims come directly from committed JSON receipts;
4. a technical reader can reach the experiment, receipt, README, and paper in one click;
5. the page deploys correctly under GitHub Pages repository-path hosting;
6. the existing scientific test suite remains green;
7. site tests verify evidence-loading and deployment invariants;
8. the site remains useful without JavaScript, while enhanced receipt values load when JavaScript is available.
