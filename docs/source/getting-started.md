# Getting started

## Install

```bash
git clone https://github.com/agpoks/neurelux.git
cd neurelux
uv venv && source .venv/bin/activate      # or: python3 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"                # or: pip install -e ".[dev]"
```

PyTorch is the primary framework throughout. If your `pip` is old enough to lack full PEP 660 editable-install support, add `--no-build-isolation` — the project's `setuptools` requirement is already satisfied locally, so build isolation just gets in the way.

```bash
python -m pytest tests/ -q     # ~40s, ~50 tests
jupyter lab notebooks/
```

## Why small notebooks, not one large model

The long-term goal is one combined ATLAS model, but that model was never built directly. Each physical sub-problem — skin effect, hysteresis, saturation, motion-induced eddy currents, energy-consistent force, friction, thermal feedback — was isolated, tested on synthetic data and/or a public benchmark, and understood on its own first. [Notebook 09](notebooks/09_atlas_small_combined_model) only combines everything, and only after every other piece existed and worked independently. See {doc}`Method landscape <methods/overview>` for why each piece was chosen, and the [notebooks](notebooks/00_overview_methods) themselves for how each one was actually built and verified.

## A note on the data

Nothing here trains on real ATLAS data, because none has surfaced anywhere in this project's workspace (see {doc}`Background <background>`). Notebook 01 uses purely synthetic data with known ground truth. Notebooks 02, 05, and 06 use **real** public data — ferrite B-H measurements, and real geometry/material parameters and a fully-transcribed 174-point experimental measurement from two public TEAM Workshop electromagnetic benchmarks, respectively (see each method's page for exactly what was and wasn't used, and why). The remaining notebooks use synthetic data generated to match the *shape* of the physics, not measured values. If real ATLAS measurements ever surface, that's what Notebook 09's data loader is meant to be extended for.

## Repository layout

```
neurelux/
├── notebooks/                one tutorial notebook per method
├── src/atlas_physics/        reusable PyTorch modules imported by the notebooks
├── data/{raw,processed,public}  never committed except small processed artifacts
├── papers/                   literature (references.bib) backing the methods
├── docs/                     this documentation site
├── scripts/                  download/prepare scripts for public datasets
├── tests/                    unit tests for src/atlas_physics
└── interfaces/simulink/      MATLAB/Simulink deployment contract (no MATLAB needed to build/run)
```

## Notebook conventions

Every experiment notebook (everything except the method-overview notebook) follows the same structure: motivation, physical problem, governing equations, what's learned vs. what stays physical, dataset description, preprocessing, architecture, training, plots, evaluation, discussion, ATLAS relevance, next step. Every model states explicitly:

```
PHYSICS-GUIDED:   ...
PHYSICS-INFORMED: ...
PHYSICS-ENCODED:  ...
```

per the definitions in {doc}`Background <background>`. No notebook is code-only — markdown and equations carry the explanation, not just code comments.

## Simulink / MATLAB integration

Not required to build or run anything today. `interfaces/simulink/` defines the eventual deployment contract — `state = model.initial_state()`, `outputs, next_state = model.step(inputs, state, dt)`, SI units, deterministic golden test vectors — so that any `src/atlas_physics` submodel can later be swapped in and out of a MATLAB/Simulink benchmark harness against the existing ATLAS model, without touching the harness itself.
