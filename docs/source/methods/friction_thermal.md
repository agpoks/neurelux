(friction-thermal)=
# Friction and thermal feedback

A physics-guided friction coefficient compared against a black-box alternative, closed with a lumped thermal model demonstrating real feedback from temperature back into friction. The Coulomb/Stribeck-type relation kept fixed here is the standard friction-modeling family surveyed in {cite:t}`armstrong1994survey`.

## The equations

$$
F_{R,i} = \mu_i F_{A,i} \quad \text{(hard-coded, never learned)}, \qquad \mu = \mathrm{softplus}\!\left(\mu_\theta(v, F_N, T)\right)
$$

$$
C_{th}\frac{dT}{dt} = P_{\text{loss}} - hA\,(T - T_{\text{ambient}}), \qquad P_{\text{loss}} = |F_R \cdot v|
$$

**PHYSICS-GUIDED:** `GuidedFriction` keeps the Coulomb relation $F_R=\mu F_A$ as a fixed Python function (`friction_force`), never inside any trainable module — only the coefficient $\mu_\theta$ is a generic network, with nothing structural stopping it from returning a physically implausible value outside its training range (a softplus keeps it non-negative, no more). `LumpedThermal` is physics-encoded — same fixed-ODE-structure pattern as every other dynamic model in this project — so `BlackBoxFriction` below is the pure black-box control, not `LumpedThermal`.

## Results

Both friction models trained on a moderate normal-force range, evaluated on an extrapolated range never seen during training:

| model | train MSE | extrapolated $F_N$ MSE |
|---|---|---|
| `GuidedFriction` | **0.000134** | **0.001097** |
| `BlackBoxFriction` | 0.000718 | 0.009941 |

```{image} ../_static/results/nb08_0.png
:alt: Friction force vs. normal force, training range shaded
:width: 75%
```

## Thermal feedback, closing the loop

A sustained-braking scenario drives the trained friction model, whose output heats the thermal model, whose rising temperature feeds back into the friction coefficient — a genuinely coupled simulation:

```{image} ../_static/results/nb08_1.png
:alt: Temperature rising under sustained braking, and mu responding
:width: 100%
```

Temperature rises from 25°C to about 207°C over the simulated event while $\mu$ drifts downward in response — but that final temperature is far outside `GuidedFriction`'s 20-80°C training range, so this specific numeric trajectory is itself an extrapolation, not a validated prediction; the qualitative self-limiting shape (rising $T$ lowers $\mu$, which lowers heat generation, which slows further heating) is a believable structural consequence, not something to trust past 80°C without recalibrating.

## Discussion

The friction comparison repeats a pattern seen throughout this project: hard-coding an exact, known relation and learning only the genuinely uncertain part costs nothing on the training range and wins clearly on extrapolation, here to normal-force values twice the training range. `BlackBoxFriction` has no structural reason to know friction force should scale with normal force at all — it has to infer that from data, and outside the range it saw, it doesn't reliably. A separate attempt to fit the thermal parameters ($C_{th}$, $hA$) themselves to a reference trajectory ran into a genuine identifiability issue — a single constant-power step response under-determines the two independently (their ratio, the thermal time constant, is much better constrained than either alone), the same class of finding as {doc}`the Cauer ladder's <cauer>` parameter-recovery result, here for a 2-parameter system instead of a 12-parameter one.

## Try it

Full walkthrough: {doc}`../notebooks/08_friction_temperature_model`. Source: [`src/atlas_physics/friction.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/friction.py) · [`src/atlas_physics/thermal.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/thermal.py).
