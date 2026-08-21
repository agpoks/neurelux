"""Lumped equivalent magnetic circuit for notebooks/03_neural_reluctance_circuit.ipynb.

Classical series circuit: MMF Theta = N*I drives flux through core + air gap
reluctance. The air-gap term is always exact geometry (R_gap = g/(mu0*A)) and
is never learned. The uncertain term is the core's relative permeability
mu_r(B, T), which saturates nonlinearly -- this module provides three ways
to represent it:

- `FixedMuR`: a single constant (what ATLAS's assumed linear circuit does,
  PLAN.md §1) -- cannot represent saturation at all.
- `SaturatingMuR`: the closed-form "true" model used to generate synthetic
  data (a Froehlich-style saturating permeability), not a fitted model.
- `NeuralMuR`: PHYSICS-ENCODED. `mu_r0 * (1 + 0.3 * tanh(NN(B, T)))` --
  topology (the circuit equation) stays fixed; a small network only
  *modulates* the base permeability, bounded to +-30%, rather than adding an
  unconstrained correction. This bound is deliberate -- see
  notebooks/03_neural_reluctance_circuit.ipynb's discussion of why an
  unconstrained additive residual (Notebook 02's UDE) extrapolated badly
  while this bounded multiplicative one does not.

`solve_B` inverts Theta -> B for any of the three via fixed-point
(successive-substitution) iteration on the saturable circuit equation,
avoiding the need for a hand-derived or autograd-based Newton step -- the
same fixed-iteration-count-instead-of-adaptive-solver philosophy as
`cauer.py`'s explicit Euler and `hysteresis.py`'s explicit-stepped ODE.
"""

from __future__ import annotations

import torch
from torch import nn

MU0 = 4 * torch.pi * 1e-7


def solve_B(Theta: torch.Tensor, g: torch.Tensor, T: torch.Tensor, mu_r_fn, l_core: float, n_iter: int = 8) -> torch.Tensor:
    """Solve Theta = H_core(B,T)*l_core + (B/mu0)*g for B, given any mu_r_fn(B,T).

    Fixed-point iteration: substitute the current B estimate's permeability
    into the (locally linear) circuit equation and repeat. Converges quickly
    because mu_r varies smoothly and slowly relative to B for any reasonable
    saturation curve -- the same "successive substitution" technique used in
    magnetic-circuit CAD tools for saturable cores.
    """
    mu_r0_guess = mu_r_fn(torch.zeros_like(Theta), T)
    B = MU0 * mu_r0_guess * Theta / (l_core + mu_r0_guess * g)
    for _ in range(n_iter):
        mu_r = mu_r_fn(B, T)
        B = MU0 * mu_r * Theta / (l_core + mu_r * g)
    return B


class FixedMuR(nn.Module):
    """Single learned scalar permeability -- ATLAS's assumed linear circuit (PLAN.md §1).
    Cannot represent saturation or temperature dependence at all, by construction.
    """

    def __init__(self, mu_r_init: float):
        super().__init__()
        self.raw_mu_r = nn.Parameter(torch.tensor(float(mu_r_init)))

    def forward(self, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_mu_r) * torch.ones_like(B)


class SaturatingMuR(nn.Module):
    """Closed-form Froehlich-style saturating permeability, used only to define the
    synthetic "true" system (not fit to data): mu_r(B,T) = mu_r0(T) / (1 + (B/Bsat(T))^n).
    """

    def __init__(self, mu_r0_ref: float, Bsat_ref: float, n_exp: float, T_ref: float = 25.0):
        super().__init__()
        self.mu_r0_ref, self.Bsat_ref, self.n_exp, self.T_ref = mu_r0_ref, Bsat_ref, n_exp, T_ref

    def mu_r0_T(self, T: torch.Tensor) -> torch.Tensor:
        return self.mu_r0_ref * (1 - 0.15 * (T - self.T_ref) / 100.0)

    def Bsat_T(self, T: torch.Tensor) -> torch.Tensor:
        return self.Bsat_ref * (1 - 0.08 * (T - self.T_ref) / 100.0)

    def forward(self, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self.mu_r0_T(T) / (1 + (torch.abs(B) / self.Bsat_T(T)) ** self.n_exp)

    def H_core(self, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return B / (MU0 * self.mu_r0_T(T)) * (1 + (B / self.Bsat_T(T)) ** self.n_exp)


class NeuralMuR(nn.Module):
    """PHYSICS-ENCODED: circuit topology (series R_Fe + R_gap) is fixed (in `solve_B`);
    only this permeability term is learned, and even then only as a bounded (+-30%)
    multiplicative modulation of a learned base mu_r0, not an unconstrained function.
    """

    def __init__(self, mu_r0_init: float, B_scale: float, T_scale: float = 100.0, hidden: int = 24):
        super().__init__()
        self.raw_mu_r0 = nn.Parameter(torch.tensor(float(mu_r0_init)))
        self.B_scale, self.T_scale = B_scale, T_scale
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        x = torch.stack([B / self.B_scale, T / self.T_scale], dim=-1)
        correction = self.net(x).squeeze(-1)
        mu_r0 = torch.nn.functional.softplus(self.raw_mu_r0)
        return torch.nn.functional.softplus(mu_r0 * (1.0 + 0.3 * torch.tanh(correction)))
