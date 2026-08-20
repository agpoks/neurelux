# interfaces/simulink/

Deployment contract for later MATLAB/Simulink integration and benchmarking against the existing ATLAS model — see `PLAN.md` §10. **No MATLAB dependency is required to build or run anything in this repository today.** This directory defines the interface every `src/atlas_physics` submodel should already satisfy, so a future Simulink S-function port is a mechanical translation rather than a redesign.

## Contract

```python
state = model.initial_state()
outputs, next_state = model.step(inputs, state, dt)
```

- `step()` is a pure function of `(inputs, state, dt)` — no internal adaptive-step solver, no hidden state outside what `initial_state()`/`step()` expose. `CauerLadder1D.step()` in `src/atlas_physics/cauer.py` is the first implementation of this contract (fixed-step explicit Euler, matching `simulate()`'s integrator exactly).
- All physically-meaningful internal states (Cauer ladder states, later: hysteresis state, thermal state) must be explicit fields of `state`, not buried inside an opaque RNN hidden vector.
- SI units at every interface boundary — see `model_io.yaml` (to be filled in as each submodel is built; today it documents the *planned* signal set from `PLAN.md` §2, not an implemented I/O spec).
- Submodels stay swappable: a benchmark harness should be able to substitute the neural reluctance graph with the original linear circuit, or the Graph-Cauer block with a classical fixed Cauer ladder, without touching the surrounding harness — this is why every submodel exposes the same `step()`-shaped interface regardless of whether it is "classical" or "neural".

## Files

| File | Purpose | Status |
|---|---|---|
| `model_io.yaml` | Signal names, units, directions (in/out/state) for each submodel | stub — documents planned ATLAS signal set only |
| `reference_step.py` | Runs `initial_state()`/`step()` for a given submodel over a fixed input trajectory and dumps outputs+states | stub |
| `export_parameters.py` | Exports a trained submodel's parameters (e.g. `CauerLadder1D.C()`, `.G()`) to a MATLAB-loadable format | stub |
| `test_vectors/` | Deterministic golden input/output/state trajectories generated from the PyTorch reference, for later `y_PyTorch ≈ y_Simulink` / `x_PyTorch ≈ x_Simulink` checks | empty — populated once a submodel is trained and frozen |

None of these are implemented yet — this directory is scaffolding created alongside `PLAN.md`/`README.md` (Step 1), to be filled in incrementally as each `src/atlas_physics` submodel is built, not as a separate late-stage effort.
