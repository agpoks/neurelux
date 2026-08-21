# NeuRelux

<p align="center"><img src="branding/logo-icon.svg" alt="NeuRelux logo" width="140"></p>

**Physics-Guided, Physics-Informed & Physics-Encoded Neural Networks for the ATLAS Magnetic Track Brake**

NeuRelux is a research template for progressively replacing/augmenting the simplified equivalent-magnetic-circuit models used in ATLAS (layered skin-effect networks, lumped reluctances, empirical friction) with neural components — while keeping as much physical structure (topology, conservation, positivity) intact as possible.

**Start here → [`PLAN.md`](PLAN.md).** It defines what "physics-guided", "physics-informed" and "physics-encoded" mean in this project, summarizes the (currently assumed — see PLAN.md §0) ATLAS model, lists candidate methods, and lays out the notebook-by-notebook execution plan. Do not skip it.

## Why "small notebooks first"

The long-term goal is one combined ATLAS model, but that model is **not** being built directly. Each physical sub-problem (skin effect, hysteresis, saturation, motion-induced eddy currents, energy-consistent force, friction, thermal feedback) is first isolated, tested on synthetic data and/or a public benchmark, and understood on its own in a small tutorial notebook — see `PLAN.md` §7 for the full list, rationale, and build order. Only Notebook 09 combines everything, and only after 01–08 exist and work.

### A note on the data

Nothing here trains on real ATLAS data yet, because none has surfaced in this workspace (see `PLAN.md` §0). Notebook 01 uses purely synthetic data with known ground truth, generated to match the *shape* of the physics (not measured values). Notebooks 02+ pull small public datasets — magnetic material B-H curves, the TEAM Workshop eddy-current benchmarks, and similar — via `scripts/download_*.py`; see the table in "Public datasets" below for what each one is and how it's used. If real ATLAS measurements ever land in `data/raw/atlas/`, Notebook 09 is where they plug in.

## Repository layout

```
neurelux/
├── PLAN.md                 <- read first
├── REFERENCES.md            external code repos consulted, what was reused vs. inspiration-only
├── notebooks/                one tutorial notebook per method (see PLAN.md §7)
├── src/atlas_physics/        reusable PyTorch modules imported by the notebooks
├── data/{raw,processed,public}  never committed except small processed artifacts; see data/README.md
├── papers/                   literature (references.bib) backing the methods; see papers/README.md
├── docs/                     Sphinx/ReadTheDocs documentation site; see "Documentation" below
├── scripts/                  download/prepare scripts for public datasets
├── tests/                    unit tests for src/atlas_physics (conservation, positivity, shape checks)
└── interfaces/simulink/      MATLAB/Simulink deployment contract (no MATLAB dependency to build/run)
```

## Status

Working and verified: the plan, the method-overview notebook, and Notebooks 01–08 — each built, executed end to end, and checked before moving to the next, per `PLAN.md` §7. Only the final combined model remains.

- [x] Plan and method overview (`PLAN.md`, Notebook 00)
- [x] Notebook 01 — synthetic 1D Cauer skin effect
- [x] Notebook 02 — hysteresis on real ferrite B-H data (UPB materialdatabase)
- [x] Notebook 03 — neural reluctance circuit
- [x] Notebook 04 — surface×depth Graph-Cauer
- [x] Notebook 05 — TEAM Workshop Problem 7 (real material/frequency parameters)
- [x] Notebook 06 — TEAM Workshop Problem 28 (real 174-point measured trajectory)
- [x] Notebook 07 — co-energy force
- [x] Notebook 08 — friction + thermal
- [ ] Notebook 09 — combined ATLAS model

Real ATLAS documentation/measurements haven't surfaced yet (see `PLAN.md` §0). Drop them into `data/raw/atlas/` when available; `PLAN.md` §1–2 get rewritten from the real source instead of the current assumed placeholder.

## Setup

```bash
cd neurelux
uv venv && source .venv/bin/activate    # or: python3 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"              # or: pip install -e ".[dev]"
jupyter lab notebooks/
```

If your `pip` is old enough to lack full PEP 660 editable-install support, add `--no-build-isolation` to the install command (setuptools already satisfies `pyproject.toml`'s `build-system.requires`, so isolation just gets in the way).

PyTorch is the primary ML framework throughout (see `pyproject.toml`). Every notebook checks for required data locally first and prints the exact `scripts/download_*.py` command if missing — notebooks never silently fail on missing data, and never require a fresh download after the first run.

## Documentation

The project also has a Sphinx documentation site at `docs/`, built for [Read the Docs](https://readthedocs.org/) (config at `.readthedocs.yaml`). It republishes `PLAN.md`, `README.md`, `REFERENCES.md`, and `papers/README.md` as linked pages, renders every notebook from its already-executed output (no re-training on the docs server), generates an API reference for `src/atlas_physics` via static analysis (`sphinx-autoapi` — no need to install PyTorch just to build docs), and renders `papers/references.bib` as a bibliography page.

Build locally:

```bash
pip install -r docs/requirements.txt --no-build-isolation   # see the pip note above
cd docs && make html   # output in docs/_build/html/index.html
```

`nbsphinx` shells out to `pandoc` for parts of the markdown-to-HTML conversion; if it isn't already on your `PATH`, `pip install pypandoc_binary` and symlink its bundled binary onto your `PATH` (this environment needed exactly that — see the pandoc note if `make html` fails with `PandocMissing`).

To publish: connect the repository on readthedocs.org — it picks up `.readthedocs.yaml` automatically and needs no further configuration.

## Public datasets

See `PLAN.md` §6 for the full table (URL, variables, license, auto-download status, which notebook uses it, ATLAS relevance). Highlights:

- **Princeton MagNet** / **MagNet Challenge 2** — ferrite B-H/loss data, used to validate hysteresis *methods* (Notebook 02); ATLAS uses steel, so a steel B-H dataset is preferred where one can be identified and is used in preference to ferrite when available.
- **UPB Material Database** / **FEM Magnetics Toolbox** — structured material lookups and FEM cross-checks.
- **HystRNN** / **Magnetic Hysteresis Neural Operator** — reference implementations for the physics-aware recurrent and neural-operator hysteresis baselines.
- **TEAM Workshop Problems 7 & 28** — public electromagnetic benchmarks for eddy-current skin effect (7) and moving-conductor force (28); closest public analogues to ATLAS's velocity-dependent eddy-current problem.
- **Kaggle Electric Motor Temperature** — used *only* to exercise the thermal submodel's training loop; explicitly **not** an ATLAS or friction dataset.

## Notebook conventions

Every experiment notebook (not the overview) follows the same 14-part structure: motivation, physical problem, governing equations (LaTeX), what's learned vs. what stays physical, dataset description, preprocessing, architecture, training, plots, evaluation, discussion, ATLAS relevance, next step. Every model states explicitly:

```
PHYSICS-GUIDED:  ...
PHYSICS-INFORMED: ...
PHYSICS-ENCODED:  ...
```

per the definitions in `PLAN.md` §3–5. No notebook is code-only — markdown and equations carry the explanation.

## Simulink / MATLAB integration

Not required to build or run anything in this repo today. `interfaces/simulink/` defines the eventual deployment contract (`state = model.initial_state()`, `outputs, next_state = model.step(inputs, state, dt)`, SI units, deterministic golden test vectors) so that any `src/atlas_physics` submodel can later be swapped in/out of a MATLAB/Simulink benchmark harness against the existing ATLAS model without touching the harness itself. See `PLAN.md` §10.
