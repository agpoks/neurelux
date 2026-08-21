# Results so far

All nine implemented methods are built, trained, and evaluated — three on real public data, six on synthetic ground truth. Read {doc}`Method landscape <methods/overview>` for what each method *is*; this page is what each one actually *showed*, in one place, so the throughline across notebooks is visible without reading all nine.

## Where physics-encoding wins cleanly

- {doc}`**Graph-Cauer** <methods/graph_cauer>`: beats an unconstrained model with 10x the parameters on velocity extrapolation, using the *same* parameter count as a topology-poorer baseline — the cleanest result in the project.
- {doc}`**TEAM28** <methods/team28>`: a physics-encoded transformer + co-energy model reproduces a real 174-point measured trajectory closely (0.85mm RMSE, including the full underdamped overshoot); a black-box model trained on the same real data settles at the wrong equilibrium entirely.
- {doc}`**Co-energy network** <methods/coenergy>`: the Maxwell-reciprocity consistency violation is ~7 orders of magnitude smaller than two independently-trained networks — a structural guarantee, not a trained-in property.
- {doc}`**Neural reluctance circuit** <methods/neural_circuit>` and {doc}`**friction** <methods/friction_thermal>`: hard-coding an exact relation (air-gap reluctance; $F_R=\mu F_N$) and learning only the genuinely uncertain part wins on every extrapolation axis tested.

## Where it's more nuanced

- {doc}`**Hysteresis** <methods/hysteresis>`: a neural residual embedded in a physics ODE (a Universal Differential Equation) wins on real temperature extrapolation but **loses badly** on amplitude extrapolation — an unconstrained additive correction has no guarantee outside its training domain, and actively damages the otherwise-correct bare physics there.
- {doc}`**Neural reluctance circuit** <methods/neural_circuit>` again, contrasted directly against hysteresis's finding: a **bounded** multiplicative correction (±30% modulation) doesn't share that failure mode. The difference is architectural, not incidental — worth reading both pages together.
- {doc}`**Cauer ladder** <methods/cauer>` (Notebook 01, the first result): generalizes well across excitation *type* it never trained on, but individual deep-layer parameters are only weakly identifiable from surface-only, low-frequency data — a genuine structural-identifiability limit, not a training failure. The same class of finding recurs in the thermal model (`friction_thermal`, §12) with a 2-parameter system.
- {doc}`**TEAM7** <methods/team7>`: even a PINN with its physics-residual loss driven to near-zero fails to extrapolate to a real, benchmark-specified second frequency, because its collocation points never covered that timescale — a soft penalty only shapes behavior where it's evaluated.

## What composing everything showed

{doc}`**The combined model** <methods/combined>` chains four of these pieces (using each one's reference physics, not retrained weights) into one coupled synthetic braking simulation with genuine temperature feedback. It surfaces an emergent, unscripted behavior — the friction coefficient traces a U-shape from two competing real effects — and its own energy balance closes to 0.02% (integration error, not a bug), a direct correctness check rather than just "the plots look right."

## What's next

Every notebook the project's build order called for now exists (see {doc}`Getting started <getting-started>`). What's next is the ablation study — training each submodule for real (not the reference functions the combined model used) and comparing the assembled learned pipeline against both the synthetic reference and, eventually, real ATLAS data.
