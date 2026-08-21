(neural-circuit)=
# Neural reluctance circuit

Compares ATLAS's assumed linear circuit, a black-box MLP, and a physics-encoded neural circuit that replaces *only* the uncertain permeability term — on extrapolation to unseen current, air gap, and temperature simultaneously.

## The equation

$$
\Theta = N I, \qquad \Theta = H_{\text{core}}(B,T)\, l_{\text{core}} + \frac{B}{\mu_0} g, \qquad \mu_r(B,T) = \frac{\mu_{r0}(T)}{1 + \left(B/B_{\text{sat}}(T)\right)^n}
$$

The air-gap term is exact geometry, never learned. Given $(\Theta, g, T)$, $B$ is found by **fixed-point (successive-substitution) iteration**, not a black-box root finder — substitute the current $B$ estimate's permeability into the linear form and repeat, converging in a handful of steps for any of the three permeability representations below.

## How it's built

```python
def solve_B(Theta, g, T, mu_r_fn, l_core, n_iter=8):
    mu_r0_guess = mu_r_fn(torch.zeros_like(Theta), T)
    B = MU0 * mu_r0_guess * Theta / (l_core + mu_r0_guess * g)
    for _ in range(n_iter):
        mu_r = mu_r_fn(B, T)
        B = MU0 * mu_r * Theta / (l_core + mu_r * g)
    return B
```

**PHYSICS-ENCODED:** `NeuralMuR`'s permeability, `mu_r0 * (1 + 0.3 * tanh(NN(B,T)))`, is a small network **bounded to a ±30% multiplicative modulation** of a learned base value by construction — `tanh` makes the bound structural, not merely typical. Ampère's law and the exact air-gap reluctance stay fixed around it; only this bounded correction is learned. That bound is the entire point, as the results below show.

## Results

Trained on a moderate operating range (sub-saturation current, small gap, moderate temperature — a small air gap deliberately, so the core rather than the gap dominates total reluctance and saturation is visible at all). Evaluated on three separate extrapolation holdouts:

| set | linear (ATLAS-style) | black-box MLP | neural circuit |
|---|---|---|---|
| train | 0.00003 | 0.00001 | 0.00000 |
| extrap. current (saturation) | 0.01251 | 0.00400 | **0.00326** |
| extrap. air gap | 0.00001 | 0.00187 | **0.00000** |
| extrap. temperature | 0.00029 | 0.00009 | **0.00000** |

```{image} ../_static/results/nb03_0.png
:alt: Saturation curve, true vs. all three models
:width: 75%
```

## Discussion

Not the same winner on every axis, and that's informative. Extrapolating the **air gap** is essentially exact for both circuit-based models and not for the MLP — because $R_{\text{gap}} = g/(\mu_0 A)$ is exact geometry encoded identically in both, never learned; the MLP has to infer that dependence from data and pays for it outside its training range. Extrapolating **current** (deeper saturation) and **temperature** both favor the neural circuit clearly.

The neural circuit wins on all three axes here — a cleaner result than {doc}`the hysteresis notebook's <hysteresis>`, where an analogous "physics-encoded + learned component" model won on temperature extrapolation but badly *lost* on amplitude extrapolation. The difference is architectural: that residual was an unconstrained additive correction on a derivative; this correction is a bounded multiplicative modulation, structurally incapable of moving permeability more than 30% from its base value regardless of what the network computes for an out-of-distribution input. Bounding what an embedded network is allowed to do is a concrete, checkable design choice — not automatic just because the network sits inside a physics-encoded equation.

## Try it

Full walkthrough: {doc}`../notebooks/03_neural_reluctance_circuit`. Source: [`src/atlas_physics/magnetic_circuit.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/magnetic_circuit.py).
