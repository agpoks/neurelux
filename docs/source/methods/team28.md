(team28)=
# TEAM Workshop Problem 28

The closest public analogue to ATLAS's own motion-coupled eddy-current problem used in this project: a real, fully-transcribed 174-point experimental measurement of an electrodynamic levitation device, from Karl, Fetzer, Kurz, Lehner & Rucker, "Description of TEAM Workshop Problem 28" (Universität Stuttgart).

## The physical setup

A cylindrical aluminum plate ($\sigma = 3.40\times10^7$ S/m, 0.107 kg) rests above two concentric coils (960 + 576 turns, opposed) carrying a 50 Hz, 20 A current. The resulting eddy currents produce a repulsive force; released from rest, the plate rises, overshoots, oscillates, and settles at a measured 11.3 mm. All of this — geometry, excitation, and the full 174-point measured trajectory — is real (`scripts/download_team28.py`).

## The equation

The coil-plate system is reduced to an equivalent transformer: a mutual inductance $M(z)$ couples the coil current to an induced eddy-current loop ($R_2$, $L_2$). Cycle-averaging over one 20 ms period gives a closed-form, co-energy-consistent repulsive force:

$$
F_{\text{avg}}(z) = \frac{1}{2}\, \hat{\imath}^2\, \omega^2\, L_2\, \frac{M(z)^2}{z_0 \left(R_2^2 + \omega^2 L_2^2\right)}, \qquad M(z) = M_0\, e^{-z/z_0}
$$

with mechanics $m\ddot{z} = F_{\text{avg}}(z) - mg - c\,\dot{z}$ — the velocity-proportional damping term $c\dot z$ is the same eddy-current braking mechanism this project's own ATLAS track brake relies on, applied here to a levitating plate instead of a moving rail.

## The damping term was not optional

The first version of this model used only $F_{\text{avg}}(z)$, with no velocity-dependent term — a conservative system, like a ball in a potential well, that cannot dissipate the kinetic energy the plate gains on its way up. It never settled: released from rest, it oscillated indefinitely around whatever equilibrium the force curve implied. This was a **qualitative** failure, not an accuracy gap, and it was fixable for an identifiable reason (a missing dissipation term) — see the notebook for the full account.

## Results

Both models fit to the entire real 174-point trajectory — the same mechanical backbone ($m\ddot z = F(z,\dot z) - mg$), differing only in how the force term is realized:

| model | RMSE | final height |
|---|---|---|
| physics-encoded (transformer + co-energy force + damping) | **0.85 mm** | 11.18 mm |
| black-box force MLP | 9.64 mm | 2.07 mm (wrong equilibrium entirely) |
| real measurement | — | 11.35 mm |

```{image} ../_static/results/nb06_1.png
:alt: Real measurement vs. both fitted models over the full transient
:width: 90%
```

## Discussion

The physics-encoded model reproduces the **entire** underdamped transient closely — not just the equilibrium, but the ~18mm overshoot peak, the undershoot, and the decaying oscillation timing — a stronger result than the model's own cycle-averaging assumption obviously guarantees, since that assumption is weakest during the fast initial lift-off. The black-box model, given the same real data and the same mechanical structure, settles at the wrong height entirely: with no physical form to fall back on, it has to learn essentially the entire nonlinear coupled dynamics — including a stable equilibrium and appropriate damping — from a single 174-point trajectory, and it doesn't.

## Try it

Full walkthrough: {doc}`../notebooks/06_moving_conductor_team28`. Source: [`src/atlas_physics/eddy_current.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/eddy_current.py).
