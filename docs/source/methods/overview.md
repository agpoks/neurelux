# Method landscape

Before building anything, it's worth surveying what a "physics + neural network" model can actually mean for this problem, and picking which combination is worth building first. This page is the survey; every method marked *implemented* below now has its own page with the equation, the code, and real results.

## Candidates

| # | Method | Learned variables | Encoded physics | Advantages | Disadvantages | ATLAS suitability |
|---|---|---|---|---|---|---|
| A | Black-box MLP | direct map $x\to y$ | none | simple, fast to fit | no extrapolation, no interpretability | poor — baseline only |
| B | Physics-guided residual NN | residual on top of a cheap baseline | none (baseline fixed, not enforced) | keeps baseline behavior far from data | residual can still misbehave outside baseline's validity | good as a baseline improvement |
| C | PINN (generic net + PDE loss) | full field, generic weights | equation as a *soft* penalty | works with little labeled data | equation can still be violated; hard to make fixed-step for Simulink | moderate — reference/validation only, {doc}`tested <team7>` |
| D | Neural equivalent magnetic circuit | $R(B,T)$ or $\mu(B,T)$ only | circuit topology, Kirchhoff's laws | directly interpretable, minimal parameters | still lumped, no spatial resolution | **high** — {doc}`implemented <neural_circuit>` |
| E | Neural reluctance graph | edge reluctances/permeances | graph = Kirchhoff laws | generalizes D to multi-pole topologies | topology must be specified by hand | **high** |
| F | Cauer ladder (1D) | layer $C_i, G_i > 0$ | ladder topology = discretized 1D diffusion | matches ATLAS's existing layered network shape; passive by construction | 1D only — no lateral redistribution | **high** — {doc}`implemented <cauer>` |
| G | Surface × depth Graph-Cauer | local $C(x,T), G(x,T,v)$ | 2D graph, same conservation form | adds lateral redistribution + velocity dependence | more parameters, needs validation | **high** — central candidate for the rail, {doc}`implemented <graph_cauer>` |
| H/I | Hysteresis: recurrent / Jiles-Atherton + residual | ODE parameters + optional residual | JA ODE structure (fixed) | classical, well-understood; captures memory | JA has known limitations (minor loops) | moderate — {doc}`implemented <hysteresis>` |
| J | FNO/RIFNO material operator | operator weights (spectral) | none built-in | strong generalization across waveforms | opaque, hard to keep as explicit Simulink state | low-moderate, optional reference |
| K | Co-energy network | single $W'_\theta(I,g,T)$ | flux/force are *exact* derivatives of $W'$ | flux and force can't become inconsistent | needs both measured together | **high** — {doc}`implemented <coenergy>` |
| L | Port-Hamiltonian NN | $J, R, H$ structure | energy + dissipation, passivity guaranteed | strongest stability guarantee | more implementation overhead | speculative — later hardening step |
| M | Full Maxwell/vector-potential PINN | vector potential field | Maxwell's equations, soft penalty | most physically complete | expensive, not real-time | low for deployment — reference only, {doc}`tested <team7>` |

Friction and the thermal feedback loop (method B applied to $F_R=\mu F_N$) and the small combined system chaining everything together are {doc}`implemented <friction_thermal>` and {doc}`implemented <combined>` too, though neither is a numbered row above — they're the closing pieces of the architecture, not separate candidate representations of a field/circuit quantity.

**Guided:** B, and the input-feature choices underlying D–L. **Informed:** C, M, and any consistency loss layered on top of an encoded model. **Encoded:** D, E, F, G, K, L.

## The recommended combination

$$
\underbrace{\text{E}}_{\text{reluctance graph}} + \underbrace{\text{G}}_{\text{graph-Cauer skin effect}} + \underbrace{\text{[H/I optional]}}_{\text{hysteresis}} + \underbrace{\text{K}}_{\text{energy-consistent force}} + \underbrace{\text{B-style friction}}_{} + \text{thermal feedback}
$$

```{eval-rst}
.. plot:: _diagrams/recommended_pipeline.py
```

Why this combination, method by method, and why the *most complicated* model is not assumed to be best, is in the repository's `PLAN.md` (§4-5 and §8-9, the ablation ladder M0-M8). Every piece now exists — see [Results](../results) for what each one actually showed.
