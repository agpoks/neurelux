# Results so far

All nine implemented methods are built, trained, and evaluated — three on real public data (N87 ferrite, TEAM7, TEAM28), six on synthetic ground truth. Read {doc}`Method landscape <methods/overview>` for what each method *is*; this page is what each one actually *showed*, with the real numbers, so the throughline across notebooks is visible without reading all nine.

## The catalogue

| method | category | data | headline result | page |
|---|---|---|---|---|
| Cauer ladder | encoded | synthetic | time-domain generalization RMSE 0.038 (sine), 0.016 (multi-freq); deep-layer params only weakly identifiable from surface data | {doc}`cauer <methods/cauer>` |
| Hysteresis, JA+UDE residual | encoded | **real** N87 (25°C/100°C) + synthetic | 0.0377 MSE on real 100°C extrap. (best) **but** 7.83 on synthetic amplitude extrap. (worst) | {doc}`hysteresis <methods/hysteresis>` |
| Neural reluctance circuit | encoded | synthetic | wins all 3 extrapolation axes: 0.00326 (current) / 0.00000 (gap) / 0.00000 (temperature) MSE | {doc}`neural_circuit <methods/neural_circuit>` |
| Graph-Cauer | encoded | synthetic | 0.000001 MSE at unseen velocity, same 68 params as a topology-poor baseline; beats a 720-param free-linear model 285x | {doc}`graph_cauer <methods/graph_cauer>` |
| TEAM7 diffusion | encoded / **informed** / guided (3-way) | **real** geometry & frequencies | Cauer ladder 0.01145 MSE at real 200Hz (held out); PINN 0.17414 despite physics residual $\approx 2\times10^{-9}$ | {doc}`team7 <methods/team7>` |
| TEAM28 levitation | encoded | **real** 174-pt measured trajectory | 0.85mm RMSE reproducing the full underdamped transient (real settle: 11.35mm, model: 11.18mm) | {doc}`team28 <methods/team28>` |
| Co-energy network | encoded | synthetic | reciprocity violation 0.00005 vs. 421.0 for two independent networks (7 orders of magnitude) | {doc}`coenergy <methods/coenergy>` |
| Friction + thermal | guided / encoded | synthetic | 0.0011 vs. 0.0099 MSE extrapolated normal force; energy balance closes to 0.0237% | {doc}`friction_thermal <methods/friction_thermal>` |
| Combined model | encoded (composition) | synthetic | 4 modules chained; U-shaped $\mu(t)$ emerges unscripted; energy balance closes to 0.0237% | {doc}`combined <methods/combined>` |

*Category* follows the {doc}`background <background>` definitions exactly: **guided** = physics shapes inputs/targets only, nothing enforced; **informed** = a governing equation as an extra *loss term*, still violable at inference; **encoded** = the equation/topology is built into the forward computation graph, unviolable up to integration error. TEAM7 is the project's one page carrying all three side by side — see {doc}`its "Three ways to build it" section <methods/team7>` for the guided-MLP / physics-informed-PINN code next to the physics-encoded Cauer ladder.

## Where physics-encoding wins cleanly

```{image} _static/results/nb04_1.png
:alt: Graph-Cauer extrapolation to unseen velocity, probe-node comparison
:width: 75%
```

{doc}`**Graph-Cauer** <methods/graph_cauer>` beats an unconstrained model with 10x the parameters on velocity extrapolation, using the *same* parameter count as a topology-poorer baseline — the cleanest result in the project (table above).

```{image} _static/results/nb06_1.png
:alt: TEAM28 real measurement vs. both fitted models over the full transient
:width: 75%
```

{doc}`**TEAM28** <methods/team28>`: the physics-encoded model (solid line above) tracks the real 174-point measured trajectory through its full underdamped overshoot; the black-box model, given the identical real data, settles at the wrong equilibrium.

{doc}`**Co-energy network** <methods/coenergy>` and {doc}`**neural reluctance circuit** <methods/neural_circuit>` round this out — both cases where a structural constraint (a shared potential; a hard-coded exact reluctance term) produces a guarantee no amount of extra loss-shaping reproduces, checked directly rather than assumed (see their tables above).

## Where it's more nuanced

- {doc}`**Hysteresis** <methods/hysteresis>`: a neural residual embedded in a physics ODE (a Universal Differential Equation) wins on real temperature extrapolation (0.0377, best of 4 models) but **loses badly** on amplitude extrapolation (7.83, worst of 4) — an unconstrained additive correction has no guarantee outside its training domain, and actively damages the otherwise-correct bare physics there. Being physics-**encoded** bounds *where* the physics lives, not what an unbounded embedded network does with it.
- {doc}`**Neural reluctance circuit** <methods/neural_circuit>` is the direct counterpoint: a **bounded** multiplicative correction (±30% modulation via `tanh`) doesn't share that failure mode and wins on all 3 axes. The difference is architectural (bounded vs. unbounded correction), not incidental — worth reading both pages together.
- {doc}`**Cauer ladder** <methods/cauer>` (Notebook 01, the first result): generalizes well across excitation *type* it never trained on (RMSE 0.038/0.016), but individual deep-layer parameters are only weakly identifiable from surface-only, low-frequency data (~37% error above 1Hz vs. ~2.5% in-band) — a genuine structural-identifiability limit, not a training failure. The same class of finding recurs in the thermal model ({doc}`friction_thermal <methods/friction_thermal>`) with a 2-parameter system.
- {doc}`**TEAM7** <methods/team7>`: even a PINN with its physics-residual loss driven to near-zero ($\approx2\times10^{-9}$) fails to extrapolate to a real, benchmark-specified second frequency (0.174 MSE, barely better than a plain MLP's 0.187) — its collocation points never covered that timescale, so a soft penalty only shaped behavior where it was evaluated. The Cauer ladder, given the same real 50Hz training data, extrapolates to real 200Hz at 0.0115 — 15x better.

## What composing everything showed

```{image} _static/results/nb09_1.png
:alt: Six-panel combined simulation results: B, F_A, F_R, T, mu, skin-effect states
:width: 100%
```

{doc}`**The combined model** <methods/combined>` chains four of these pieces (using each one's reference physics, not retrained weights) into one coupled synthetic braking simulation with genuine temperature feedback. It surfaces an emergent, unscripted behavior — the friction coefficient traces a U-shape from two competing real effects (falling velocity raises $\mu$ via Stribeck, rising temperature lowers it) — and its own energy balance closes to 0.0237% (72597.13 J vs. 72579.91 J generated — the residual is explicit-Euler integration error, not a physics bug), a direct correctness check rather than just "the plots look right."

## What's next

Every notebook the project's build order called for now exists (see {doc}`Getting started <getting-started>`). What's next is the ablation study — training each submodule for real (not the reference functions the combined model used) and comparing the assembled learned pipeline against both the synthetic reference and, eventually, real ATLAS data.
