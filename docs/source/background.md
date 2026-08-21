# Background: the ATLAS magnetic track brake

## What a magnetic track brake does

A magnetic track brake is a non-contact-actuated friction brake used on rail vehicles: an electromagnet is lowered onto the rail head, current through its coil generates an attractive normal force that clamps the magnet body against the rail, and the resulting friction as the magnet slides along the moving rail produces the braking force. Unlike a wheel brake, it doesn't rely on wheel-rail adhesion, which is why it's used as a supplementary brake for emergency and high-deceleration scenarios.

```{eval-rst}
.. plot:: _diagrams/track_brake_schematic.py
```

Three physical effects make this hard to model well:

1. **Saturation.** The steel pole and rail head don't have constant permeability -- `permeability` drops as flux density approaches saturation, so the simple relation `flux = MMF / reluctance` becomes nonlinear.
2. **Skin effect.** A solid steel rail doesn't respond to a time-varying or motion-induced field uniformly through its depth -- the field diffuses in from the surface, more slowly at higher frequency (or higher relative velocity). See [Cauer ladder](methods/cauer.md) below for exactly how this is modeled.
3. **Motion-induced eddy currents.** Relative velocity between the magnet and the rail changes the effective field distribution and adds its own retarding force component (the same reason a magnet dropped down a copper pipe falls slowly) -- not yet covered by an implemented notebook, see `PLAN.md` for the plan (Notebooks 05-06, TEAM Workshop benchmarks).

## The governing equations, briefly

**Magnetic circuit (lumped).** Magnetomotive force `Theta = N I` drives flux `Phi` through a loop of reluctances -- air gap `R_gap = g / (mu_0 A)`, steel `R_Fe(B)` (nonlinear once saturation matters):

$$
\Phi = \frac{\Theta}{R_{\text{total}}(B, T)}
$$

**Magnetic diffusion (skin effect), 1D through depth $z$:**

$$
\mu \frac{\partial H}{\partial t} = \frac{\partial}{\partial z}\!\left(\rho \frac{\partial H}{\partial z}\right), \qquad \rho = \frac{1}{\sigma}
$$

This is the equation [`methods/cauer.md`](methods/cauer.md) discretizes into a layered network and actually trains -- see that page for the full derivation.

**Attraction force**, from magnetic co-energy `W'(I, g, T)`:

$$
F_A = -\frac{\partial W'}{\partial g}, \qquad \lambda = \frac{\partial W'}{\partial I}
$$

**Friction and thermal feedback:**

$$
F_{R,i} = \mu_i F_{A,i}, \qquad C_{th}\frac{dT}{dt} = P_{\text{loss}} - hA(T - T_{\text{ambient}})
$$

None of these last two are implemented yet (Notebooks 07-08 in the plan) -- they're here so the whole physical picture is visible from the start, not assembled piecemeal.

## Three ways physics can enter a neural model

This project is deliberately built around a three-way distinction, used consistently across every method and every notebook:

```{eval-rst}
.. plot:: _diagrams/physics_taxonomy.py
```

**PHYSICS-GUIDED** — physical variables or a cheap physical baseline shape the *input* or *training target* of an otherwise generic learner. Nothing prevents the network from predicting something physically impossible.

**PHYSICS-INFORMED** — the architecture is still generic, but a governing equation is added as an *extra loss term*. The network can still violate the physics at inference time; it's only discouraged from doing so where the loss was evaluated during training.

**PHYSICS-ENCODED** — the equation or topology is built directly into the forward computation graph, so it *cannot* be violated by construction (up to numerical integration error). Only physically-meaningful, positivity-constrained quantities are learned; the equations connecting them are fixed.

These aren't mutually exclusive within one architecture — see [`methods/overview.md`](methods/overview.md) for how the candidate methods combine all three at different levels, and [`methods/cauer.md`](methods/cauer.md) for the first one actually built and trained.
