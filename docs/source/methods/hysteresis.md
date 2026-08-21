(hysteresis)=
# Hysteresis — Jiles-Atherton on real data

The first method in this project trained on real (not synthetic) data: a Jiles-Atherton hysteresis model calibrated to real ferrite B-H measurements, compared against generic MLP/GRU baselines and a Universal-Differential-Equation variant on both real and synthetic extrapolation tests.

## The equation

$$
M_{an}(H_e) = M_s\left[\coth\!\left(\frac{H_e}{a}\right) - \frac{a}{H_e}\right], \qquad H_e = H + \alpha M
$$

$$
\frac{dM}{dH} = \frac{(1-c)\dfrac{M_{an}-M}{k\delta - \alpha(M_{an}-M)} + c\dfrac{dM_{an}}{dH_e}}{1+c}, \qquad \delta = \mathrm{sign}\!\left(\frac{dH}{dt}\right)
$$

Five physical parameters ($M_s$, $a$, $\alpha$, $k$, $c$). `src/atlas_physics/hysteresis.py::JilesAtherton` implements this as an explicit-Euler rollout in $H$, not the classical implicit formulation — the same fixed-step-integrator philosophy as {doc}`the Cauer ladder <cauer>`.

## How it's built

```python
def dM_dH(self, He, M, delta):
    Ms, a, alpha, k, c = self.params()          # all softplus/sigmoid-constrained
    x = He / a
    Man = Ms * langevin(x)
    dMan_dHe = (Ms / a) * dlangevin(x)
    denom = k * delta - alpha * (Man - M)
    dMirr_dHe = (Man - M) / denom
    dM_dH = ((1 - c) * dMirr_dHe + c * dMan_dHe) / (1 + c)
    if self.residual is not None:                # the UDE variant
        dM_dH = dM_dH + self.residual(torch.stack([He, M, delta], dim=-1)).squeeze(-1)
    return dM_dH
```

Passing `residual=` a small MLP turns this into a **Universal Differential Equation** (Rackauckas et al., 2020) — the known ODE stays exactly in the forward graph, only an unknown correction term is replaced by a network.

## Real data

Ferrite N87 (Epcos) B-H major-loop measurements at 25°C and 100°C, extracted from the UPB `materialdatabase` package (`scripts/download_materialdatabase.py`) — real, digitized manufacturer datasheet points, 21 and 22 of them respectively. Ferrite, not steel: no open electrical-steel B-H dataset has been identified yet, so this is the documented method-validation fallback, not a stand-in for ATLAS's actual material.

```{image} ../_static/results/nb02_0.png
:alt: Real N87 major loops at 25C and 100C
:width: 85%
```

## Results

Trained on the real 25°C loop plus physics-generated synthetic minor loops (the real dataset alone is too sparse to train a recurrent model or test amplitude generalization at all). Evaluated on the **real** 100°C loop (temperature extrapolation) and a synthetic amplitude beyond training range:

| Model | real 25°C (train) | real 100°C (extrap.) | synthetic amp=0.95 (extrap.) |
|---|---|---|---|
| JA-only | 0.0205 | 0.0999 | 0.0000 (generated this target) |
| JA+residual (UDE) | 0.0144 | **0.0377 (best)** | 7.83 (far worse than JA-only) |
| MLP | 0.0607 | 0.0786 | 1236 (catastrophic) |
| GRU | **0.0066 (best)** | 0.1651 (worst) | 1177 (catastrophic) |

```{image} ../_static/results/nb02_3.png
:alt: Temperature and amplitude extrapolation comparison
:width: 100%
```

## Discussion

Two results, not one, and they point in different directions. The neural-residual model wins clearly on **real** temperature extrapolation — the residual corrects a systematic model-form error visible in the plain fit's steep-knee region, and that correction generalizes across temperature. The same residual **loses badly** on amplitude extrapolation, worse than the pure physics it was meant to only correct — trained only up to amplitude 0.7, an unconstrained MLP has no guarantee outside that range, and it actively damages the bare equation's otherwise-correct extrapolation there. This is a genuine, checkable limitation of embedding a neural correction inside a physics ODE: it does not automatically inherit the equation's extrapolation guarantees. Compare against {doc}`the neural reluctance circuit <neural_circuit>`, where a *bounded* correction avoids exactly this failure mode.

## Try it

Full walkthrough: {doc}`../notebooks/02_hysteresis_material_model`. Source: [`src/atlas_physics/hysteresis.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/hysteresis.py) · [`src/atlas_physics/materials.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/materials.py).
