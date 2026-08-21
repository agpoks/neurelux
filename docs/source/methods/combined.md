(combined)=
# Small combined ATLAS model

The culmination notebook: chains the magnetic circuit, Cauer skin-effect ladder, co-energy force, and friction/thermal feedback loop into one coupled synthetic braking simulation.

## The chain

```
I, U, g, v, T
      |
Nonlinear Magnetic Circuit  ---->  Cauer skin states (parallel, driven by the same B(t))
      |
Co-energy / attraction force
      |
friction model
      |
F_R,total  ---->  thermal dynamics  ---->  feeds back into circuit + friction
```

A deliberate, stated scope decision: the skin-effect ladder runs in parallel with the force/friction/thermal chain here, driven by the same computed flux density, rather than *feeding into* the force calculation — the full target architecture's remaining piece, not a small extension.

## What's composed, and what isn't trained

This notebook trains nothing new. Every submodule uses the **reference physics function** established when it was individually validated — `SaturatingMuR`, `true_coenergy`, `true_mu`, a chosen-parameter `LumpedThermal` — not the specific trained weights from each notebook's own comparison (not persisted to disk). This builds the synthetic combined system a real ablation study would train learned submodels against, rather than re-running eight training loops in one notebook.

## Results

A synthetic braking event: current ramps to 2A and holds, velocity decelerates from 30 m/s to zero over 5s, temperature is a genuine feedback state throughout.

```{image} ../_static/results/nb09_1.png
:alt: Six-panel simulation results: B, F_A, F_R, T, mu, skin-effect states
:width: 100%
```

Peak attraction force reaches roughly 7000N and braking force roughly 1500N while current is still rising; temperature climbs smoothly to about 112°C. The friction coefficient traces a **U-shape** over the event — not scripted, a direct consequence of composing the pieces: falling velocity would *raise* $\mu$ (the Stribeck term in `true_mu`), rising temperature *lowers* it, and which one dominates changes over the course of the event.

### Does it conserve energy the way its own equation says it should?

$C_{th}\,dT/dt = P_{\text{loss}} - hA(T-T_{\text{ambient}})$ implies an exact energy balance, checked directly rather than assumed:

```text
total friction heat generated:        72579.91 J
heat stored (C_th * delta T):          68816.92 J
heat lost to ambient (integrated):      3780.21 J
stored + lost:                         72597.13 J
relative imbalance: 0.0237%
```

The residual is integration error (explicit Euler, discrete Riemann-sum accounting), not a physics bug.

## Discussion

Two honest limitations, stated rather than glossed over. First, nothing here was trained — building this same pipeline from the *learned* submodels and comparing it against this reference simulation's output is exactly the ablation ladder (M0-M8) this project's plan calls the natural next step, not done here. Second, the skin-effect ladder's states don't feed into the force calculation; doing that properly means redefining the co-energy potential as a function of the Graph-Cauer state rather than raw current — a real architectural change.

## Where this leaves the project

Every notebook the project's build order called for now exists and runs. What's next isn't more notebooks — it's training each submodule for real, assembling the learned pipeline, and comparing it against both this synthetic reference and, eventually, real ATLAS data.

## Try it

Full walkthrough: {doc}`../notebooks/09_atlas_small_combined_model`.
