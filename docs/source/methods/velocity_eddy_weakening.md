(velocity-eddy-weakening)=
# Velocity-dependent flux weakening

A magnetic track brake's assumed model is usually quasi-static — attraction force from instantaneous current, gap, and temperature, as if the vehicle stood still. A real, current (2025) measurement from TU Wien's Institute of Mechanics and Mechatronics says that assumption is systematically wrong once the vehicle is moving, and this page tests four ways to represent the effect it measured.

## The real measurement behind this page

{cite:t}`ebner2025stability` instrumented a magnetic track brake with a secondary coil around each brake element and correlated the induced voltage with the magnetic flux through the pole, $U_{\text{ind}} = -N_{\text{sec}}\,d\phi_p/dt$, during real braking manoeuvres from 100km/h to standstill. Their finding: **the magnetic flux is significantly reduced at higher vehicle velocities** — attributed to motion-induced eddy currents in the rail opposing the driving magnetomotive force (MMF), the same mechanism that produces "flux expulsion" in linear and rotating eddy-current machines ({cite:t}`wang2018simple`; {cite:t}`gholizad2009direct`). They explicitly note that deriving this from first principles for a real, segmented 3D geometry needs a full 3D eddy-current solver, and instead fit an analytical function to the measured velocity dependence, folded directly into a magnetic-equivalent-circuit (MEC) — the same lumped-parameter family {doc}`neural_circuit` already uses ({cite:t}`kallenbach2018elektromagnete`).

## The equation

Starting from {doc}`neural_circuit`'s circuit equation, the driving MMF is reduced by a velocity-dependent loss term before it reaches the core/gap reluctance split:

$$
\Theta_{\text{eff}}(v) = \Theta - \text{MMF}_{\text{loss}}(v), \qquad \Theta_{\text{eff}} = H_{\text{core}}(B,T)\, l_{\text{core}} + \frac{B}{\mu_0} g
$$

This notebook's synthetic "true" system uses an exponential-saturation form for $\text{MMF}_{\text{loss}}$ — deliberately different from the functional form the physics-encoded model below assumes, so recovering it is a genuine test:

$$
\text{MMF}_{\text{loss,true}}(v) = \text{MMF}_{\max}\left(1 - e^{-v/v_c}\right)
$$

Illustrative values only — no real ATLAS or MTB velocity-dependence data is available in this environment; only the qualitative saturating-monotonic shape is grounded in {cite:t}`ebner2025stability`'s measurements, not these specific numbers.

## Four ways to build it

**PHYSICS-ENCODED but incomplete:** no velocity term at all ($\text{MMF}_{\text{loss}}\equiv 0$) — what a standard quasi-static model assumes.

**PHYSICS-GUIDED** (baseline): a free MLP, $v \to \text{MMF}_{\text{loss}}$, no constraint — could even predict flux *increasing* with velocity if that fit training noise.

**PHYSICS-CONSTRAINED:** `VelocityMMFLossConstrained` — a free MLP again, hard-bounded to $[0, \text{MMF}_{\text{cap}}]$ via a sigmoid:

```python
class VelocityMMFLossConstrained(nn.Module):
    def forward(self, v):
        x = (v / self.v_scale).unsqueeze(-1)
        raw = self.net(x).squeeze(-1)
        return self.MMF_cap * torch.sigmoid(raw)   # structurally in [0, MMF_cap], any input
```

**PHYSICS-ENCODED:** `VelocityMMFLoss` — the two-parameter saturating functional form, following {cite:t}`ebner2025stability`'s own phenomenological-fit strategy:

```python
class VelocityMMFLoss(nn.Module):
    def forward(self, v):
        MMF_max = torch.nn.functional.softplus(self.raw_MMF_max)
        v_c = torch.nn.functional.softplus(self.raw_v_c) + 1e-3
        return MMF_max * v / (v + v_c)   # exactly 0 at v=0, saturates to MMF_max
```

All four route their loss term through the same `solve_B` fixed-point solver {doc}`neural_circuit` uses, via $\Theta_{\text{eff}}$ — only the loss term itself differs.

## Results

Current, gap, and temperature held at one representative operating point; trained on velocities from standstill to a moderate range, evaluated on a held-out **higher-velocity extrapolation** — an emergency/high-speed braking regime, directly what {cite:t}`ebner2025stability` motivate this problem with (mainline MTBs operate above 140km/h).

| set | no velocity term | black-box (guided) | physics-constrained | physics-encoded |
|---|---|---|---|---|
| train | 0.00290 | 0.00001 | 0.00001 | 0.00001 |
| extrap. v | 0.00898 | 0.00020 | 0.00005 | **0.00002** |

```{image} ../_static/results/nb10_0.png
:alt: Flux density vs. velocity, true vs. all four models, training range shaded
:width: 90%
```

## Discussion

The ranking holds on *both* rows, not just extrapolation: even on the training range, the no-velocity-term baseline is already two orders of magnitude worse than any model that represents the effect at all (0.00290 vs. ~0.00001) — a quasi-static model is measurably wrong during ordinary moving operation, well before any extrapolation question is asked, exactly what {cite:t}`ebner2025stability` report from real measurements.

Past the training range the three velocity-aware models separate cleanly by how much freedom they have: the unconstrained black-box degrades the most (0.00020, 20x its own training error); the hard-bounded physics-constrained model degrades far less (0.00005), unable to predict a runaway loss even with the wrong shape; the physics-encoded model — given the right qualitative structure (zero at $v{=}0$, saturating) but the *wrong* specific functional family (rational vs. the true exponential) — wins outright (0.00002), recovering $v_c=15.73$ against a true $15$ (5% error) from data that only reached $v{=}20$m/s. The one imperfect recovery is itself informative: fitted $\text{MMF}_{\max}=51.31$ overshoots the true $40$ by ~28%, because training data reaching only $\approx 1.3\,v_c$ pins down *how fast* the loss saturates far better than *how large* the plateau is — the same partial-range identifiability pattern as {doc}`the Cauer ladder's <cauer>` deep-layer parameters and {doc}`the thermal model's <friction_thermal>` two-parameter fit, here for a third, unrelated physical quantity.

## Try it

Full walkthrough: {doc}`../notebooks/10_velocity_dependent_eddy_weakening`. Source: [`src/atlas_physics/magnetic_circuit.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/magnetic_circuit.py).
