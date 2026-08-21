(team7)=
# TEAM Workshop Problem 7

The first method tested against parameters this project didn't choose: real geometry, material, and excitation frequencies from the official TEAM Workshop Problem 7 ("Asymmetrical Conductor with a Hole") benchmark specification.

## An honest scoping note

TEAM7's real geometry is a solid aluminum plate with an eccentric rectangular hole, excited by an offset racetrack coil — genuinely 3D and asymmetric, which is the point of the benchmark. Reproducing its full 3D field solution needs a 3D FEM/BEM solver, out of scope for this project's reduced-order approach. Its actual reference field measurements are published in a companion results paper (COMPEL, 1990) not freely accessible from this environment — **no values from that paper are used or invented here**; see `REFERENCES.md` in the repository for exactly what was and wasn't obtained.

What *is* used, taken directly from the official spec PDF (`scripts/download_team7.py`; [compumag.org](https://www.compumag.org/wp/wp-content/uploads/2018/06/problem7.pdf)):

| Parameter | Value |
|---|---|
| Plate conductivity $\sigma$ | $3.526\times10^7$ S/m |
| Plate thickness | 19 mm |
| Coil | 2742 ampere-turns |
| Excitation frequencies | 50 Hz and 200 Hz |

## The question asked instead

With these real numbers, the 1D diffusion equation has a known closed-form solution for a semi-infinite conductor:

$$
H(z,t) = H_0\, e^{-z/\delta} \cos\!\left(\omega t - \frac{z}{\delta}\right), \qquad \delta = \sqrt{\frac{2}{\omega \mu_0 \sigma}}
$$

At TEAM7's real $\sigma$ and thickness: $\delta(50\text{Hz}) \approx 12.0$mm, $\delta(200\text{Hz}) \approx 6.0$mm — thickness/$\delta$ of 1.6 and 3.2 respectively, so the semi-infinite approximation is more accurate at the higher frequency. **Does {doc}`the Cauer ladder <cauer>`, reused unmodified, reproduce this analytical solution when calibrated at one real frequency and extrapolated to the other?** — the closest question this project can honestly ask of TEAM7 without a 3D solver or the inaccessible reference table.

## Results

Trained on sparse samples of the analytical solution at 50 Hz only; 200 Hz (TEAM7's second real, specified frequency) held out entirely.

| model | 50Hz (train) | 200Hz (held out) |
|---|---|---|
| Cauer ladder | 0.00878 | **0.01145** |
| MLP | 0.00064 | 0.18683 |
| PINN (physics residual $\approx 2\times10^{-9}$) | 0.00108 | 0.17414 |

```{image} ../_static/results/nb05_0.png
:alt: Depth profile, analytical vs. all three models, both frequencies
:width: 100%
```

## Discussion

The Cauer ladder extrapolates cleanly to the real 200 Hz case, because its topology is the correct diffusion equation at *any* frequency, not just the one it was calibrated at. The more interesting result is the PINN: even with the diffusion residual driven down to near-zero at 400 collocation points, it extrapolates about as poorly as the plain MLP. The reason is specific — collocation points were sampled over the *same* time span as training (two periods of the 50 Hz signal); satisfying the PDE well *inside* that domain doesn't force the learned function to behave correctly at a timescale it was never evaluated at. A soft penalty shapes behavior only where it's evaluated; a structural constraint holds everywhere, including at a frequency nobody sampled.

## Try it

Full walkthrough: {doc}`../notebooks/05_eddy_current_team7`. Source: [`src/atlas_physics/eddy_current.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/eddy_current.py).
