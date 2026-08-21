(neural-circuit)=
# Neural reluctance circuit

Compares five ways to predict flux density from current, air gap, and temperature — ATLAS's assumed linear circuit, a black-box MLP, a physics-informed soft-residual model, a physics-constrained hard-capped model, and a physics-encoded neural circuit that replaces *only* the uncertain permeability term inside the exact circuit equation — on extrapolation to unseen current, air gap, and temperature. The magnetic-equivalent-circuit form itself follows the standard lumped-parameter treatment of electromagnets ({cite:t}`kallenbach2018elektromagnete`).

## The equation

$$
\Theta = N I, \qquad \Theta = H_{\text{core}}(B,T)\, l_{\text{core}} + \frac{B}{\mu_0} g, \qquad \mu_r(B,T) = \frac{\mu_{r0}(T)}{1 + \left(B/B_{\text{sat}}(T)\right)^n}
$$

The air-gap term is exact geometry, never learned. Given $(\Theta, g, T)$, $B$ is found by **fixed-point (successive-substitution) iteration**, not a black-box root finder — substitute the current $B$ estimate's permeability into the linear form and repeat, converging in a handful of steps for the linear and neural-circuit models below (the informed and constrained models predict $B$ directly instead — see "How it's built").

## Five ways to build it

**PHYSICS-ENCODED (linear, too simple):** a single learned scalar $\mu_r$, routed through the exact circuit equation via `solve_B`. This *is* physics-encoded, but the learned quantity can't represent saturation. What ATLAS's assumed model does.

**PHYSICS-GUIDED** (black-box baseline): raw $(I,g,T)$ into a free MLP, $B$ out — no circuit structure anywhere.

**PHYSICS-INFORMED:** `PhysicsInformedB` — the *same* free-MLP architecture as the black-box, trained on data plus a soft Ampère's-law residual loss evaluated at collocation points, using a second, unconstrained permeability network only to compute that residual:

```python
def residual(self, I, g, T, N, l_core):
    B = self.forward(I, g, T)               # free prediction, never routed through solve_B
    H_core = B / (MU0 * self.mu_r(B, T))     # unconstrained permeability, used only here
    Theta = N * I
    return Theta - H_core * l_core - (B / MU0) * g
```

**PHYSICS-CONSTRAINED:** `PhysicsConstrainedB` — another free MLP, no circuit equation anywhere, but its output is hard-capped at a *known* (not exactly measured) material saturation value:

```python
def forward(self, I, g, T):
    x = torch.stack([I / self.I_scale, g / self.g_scale, T / self.T_scale], dim=-1)
    raw = self.net(x).squeeze(-1)
    return self.B_cap * torch.tanh(raw)   # structurally can't exceed B_cap, any input
```

**PHYSICS-ENCODED (neural circuit):** `NeuralMuR`'s permeability, `mu_r0 * (1 + 0.3 * tanh(NN(B,T)))`, is a small network **bounded to a ±30% multiplicative modulation** of a learned base value by construction. Ampère's law and the exact air-gap reluctance stay fixed around it; only this bounded correction is learned, and it's routed through the same `solve_B` as the linear model.

## Results

Trained on a moderate operating range (sub-saturation current, small gap, moderate temperature — a small air gap deliberately, so the core rather than the gap dominates total reluctance and saturation is visible at all). Evaluated on three separate extrapolation holdouts:

| set | linear (encoded) | MLP (guided) | informed | constrained | neural circuit (encoded) |
|---|---|---|---|---|---|
| train | 0.00003 | 0.00001 | 0.00000 | 0.00000 | 0.00000 |
| extrap. current (saturation) | 0.01251 | 0.00400 | 0.00307 | **0.00031** | 0.00421 |
| extrap. air gap | 0.00001 | 0.00187 | 0.00117 | 0.00060 | **0.00000** |
| extrap. temperature | 0.00029 | 0.00009 | **0.00000** | 0.00001 | **0.00000** |

```{image} ../_static/results/nb03_0.png
:alt: Saturation curve, true vs. all five models
:width: 85%
```

## Discussion

Not the same winner on every axis, and now with five models the pattern is sharper than a simple "more physics is better" story:

- **Extrapolating current (deeper saturation):** the *physics-constrained* model wins outright — better even than the fully physics-encoded neural circuit. Its hard `B_cap` bound is exactly the right prior for this failure mode: pushed past its training range, a free or under-penalized model can keep extrapolating upward without limit, while `PhysicsConstrainedB` structurally cannot, regardless of the input. It doesn't know the *shape* of saturation, but knowing the *ceiling* alone beats a model with more physics but no ceiling.
- **Extrapolating air gap:** the neural circuit is essentially exact, because $R_{\text{gap}}=g/(\mu_0 A)$ is exact geometry baked into its forward pass. Neither the soft residual nor the hard cap encodes that specific relationship, so both trail behind it here.
- **Extrapolating temperature:** informed and neural-circuit effectively tie for best. The physics-informed model's Ampère's-law residual couples $T$ through its own learned $\mu_r(B,T)$, and that soft coupling happens to generalize across temperature about as well as the exactly-encoded version — a genuinely different outcome from {doc}`Notebook 05's PINN <team7>`, where the same *kind* of soft residual did **not** help extrapolation at all. The difference: this residual is an *algebraic* constraint over collocation points drawn from the same input box the data lives in, so extrapolating past it is extrapolating a smooth, low-order relationship — not evaluating a PDE at a frequency an integral never sampled.

The broader lesson, now confirmed on two independent axes (this page and {doc}`velocity-dependent flux weakening <velocity_eddy_weakening>`): **the type of physics constraint matters as much as whether physics is present at all.** An unconstrained additive correction ({doc}`Notebook 02's UDE <hysteresis>`) can actively hurt extrapolation; a well-scaled but still-unconstrained soft residual can help on some axes and not others; a single hard bound with zero equation knowledge wins the one axis its bound is actually relevant to; full structural encoding wins wherever the *exact* relationship it encodes is the one being tested. Which pattern to reach for depends on what you actually know for certain — not on maximizing how much physics gets used.

## Try it

Full walkthrough: {doc}`../notebooks/03_neural_reluctance_circuit`. Source: [`src/atlas_physics/magnetic_circuit.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/magnetic_circuit.py).
