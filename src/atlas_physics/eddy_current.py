"""Moving-conductor / eddy-current physics for Notebooks 05 (TEAM7) and 06 (TEAM28).

Governing physics: J = sigma*(-grad(V) + v x B), div(J) = 0. Two very
different reduced representations of this are used here, matched to what
each TEAM benchmark actually needs:

- `skin_depth` / `analytical_semi_infinite_H`: the classical closed-form 1D
  diffusion solution for a semi-infinite conductor under sinusoidal surface
  excitation -- used in Notebook 05 as an analytically-known reference
  (TEAM7's real geometry is 3D and asymmetric; this is the reduced-order
  analytical solution its real material/frequency parameters imply for the
  1D skin-effect problem specifically, not a full reproduction of TEAM7's
  own field measurements -- see the notebook for why).
- `EddyLevitationAveraged`: PHYSICS-ENCODED cycle-averaged (quasi-static)
  coupled electromechanical model for Notebook 06 -- a mutual inductance
  M(z) between the exciting coil and the plate's eddy-current loop, treated
  as a transformer secondary (R2, L2), with force from time-averaged
  co-energy over one excitation cycle, plus a velocity-proportional
  eddy-current damping term (see the notebook: dropping this term entirely,
  as a first cut, gave a non-dissipative model that could not reproduce the
  real system settling to its measured equilibrium height at all).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

MU0 = 4 * np.pi * 1e-7


def skin_depth(f: float, sigma: float, mu: float = MU0) -> float:
    """Classical skin depth delta = sqrt(2 / (omega * mu * sigma))."""
    omega = 2 * np.pi * f
    return np.sqrt(2.0 / (omega * mu * sigma))


def analytical_semi_infinite_H(z, t, f: float, sigma: float, mu: float = MU0, H0: float = 1.0):
    """H(z,t) = H0 exp(-z/delta) cos(omega t - z/delta): the steady-state solution
    of mu dH/dt = d/dz(rho dH/dz) for a semi-infinite conductor driven by
    H0 cos(omega t) at z=0. Accepts numpy arrays or floats for z, t.
    """
    omega = 2 * np.pi * f
    delta = skin_depth(f, sigma, mu)
    return H0 * np.exp(-z / delta) * np.cos(omega * t - z / delta)


class EddyLevitationAveraged(nn.Module):
    """Physics-encoded, cycle-averaged coupled model for TEAM28 (Notebook 06).

    Learned (all positivity-constrained via softplus): M0, z0 (mutual
    inductance M(z) = M0 exp(-z/z0)), R2, L2 (equivalent eddy-current loop
    resistance/inductance), damp (velocity-proportional eddy-current
    damping coefficient). The mechanical integration (Newton's second law)
    and the co-energy force formula are fixed structure -- never learned.
    """

    def __init__(
        self,
        m_plate: float,
        I_hat: float,
        f0: float,
        g: float = 9.81,
        M0_init: float = 0.02,
        z0_init: float = 0.02,
        R2_init: float = 2e-3,
        L2_init: float = 0.5,
        damp_init: float = 0.5,
    ):
        super().__init__()
        self.m_plate, self.I_hat, self.omega, self.g = m_plate, I_hat, 2 * np.pi * f0, g

        def inv_softplus(y: float) -> torch.Tensor:
            y_t = torch.tensor(float(y))
            return y_t + torch.log(-torch.expm1(-y_t))

        self.raw_M0 = nn.Parameter(inv_softplus(M0_init))
        self.raw_z0 = nn.Parameter(inv_softplus(z0_init))
        self.raw_R2 = nn.Parameter(inv_softplus(R2_init))
        self.raw_L2 = nn.Parameter(torch.tensor(float(L2_init)))
        self.raw_damp = nn.Parameter(inv_softplus(damp_init))

    def params(self):
        M0 = torch.nn.functional.softplus(self.raw_M0)
        z0 = torch.nn.functional.softplus(self.raw_z0) + 1e-4
        R2 = torch.nn.functional.softplus(self.raw_R2) + 1e-6
        L2 = torch.nn.functional.softplus(self.raw_L2) + 1e-9
        damp = torch.nn.functional.softplus(self.raw_damp)
        return M0, z0, R2, L2, damp

    def force(self, z: torch.Tensor) -> torch.Tensor:
        """Cycle-averaged repulsive (co-energy) force, from the mutual-inductance
        transformer model in sinusoidal steady state at fixed z (quasi-static:
        valid when z varies slowly relative to one excitation period).
        """
        M0, z0, R2, L2, _ = self.params()
        M = M0 * torch.exp(-z / z0)
        return 0.5 * self.I_hat**2 * self.omega**2 * L2 * M**2 / (z0 * (R2**2 + self.omega**2 * L2**2))

    def simulate(self, t_grid: np.ndarray, z0_init: float, dt: float) -> torch.Tensor:
        """Explicit-Euler rollout of m*z'' = F(z) - m*g - damp*v, evaluated at
        (and only stored at) the times in `t_grid`; sub-steps at `dt` in between.
        """
        _, _, _, _, damp = self.params()
        z = torch.tensor(float(z0_init))
        v = torch.tensor(0.0)
        t = 0.0
        zs = [z]
        for k in range(1, len(t_grid)):
            t_target = float(t_grid[k])
            while t < t_target - 1e-9:
                step = min(dt, t_target - t)
                a = self.force(z) / self.m_plate - self.g - damp * v / self.m_plate
                v = v + step * a
                z = z + step * v
                t += step
            zs.append(z)
        return torch.stack(zs)
