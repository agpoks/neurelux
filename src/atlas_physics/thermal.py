"""Lumped thermal model for notebooks/08_friction_temperature_model.ipynb.

PHYSICS-ENCODED: C_th dT/dt = P_loss - hA*(T - T_ambient) is a fixed ODE
structure; only the two physical parameters (thermal capacitance C_th,
heat-transfer coefficient-area product hA) are learned, positivity
constrained via softplus. Explicit-Euler rollout, the same fixed-step
philosophy as every other dynamic model in this project.
"""

from __future__ import annotations

import torch
from torch import nn


class LumpedThermal(nn.Module):
    def __init__(self, C_th_init: float = 500.0, hA_init: float = 5.0):
        super().__init__()

        def inv_softplus(y: float) -> torch.Tensor:
            y_t = torch.tensor(float(y))
            return y_t + torch.log(-torch.expm1(-y_t))

        self.raw_C_th = nn.Parameter(inv_softplus(C_th_init))
        self.raw_hA = nn.Parameter(inv_softplus(hA_init))

    def params(self):
        C_th = torch.nn.functional.softplus(self.raw_C_th) + 1e-3
        hA = torch.nn.functional.softplus(self.raw_hA) + 1e-6
        return C_th, hA

    def rhs(self, T: torch.Tensor, P_loss: torch.Tensor, T_ambient: torch.Tensor) -> torch.Tensor:
        C_th, hA = self.params()
        return (P_loss - hA * (T - T_ambient)) / C_th

    def simulate(self, P_loss_seq: torch.Tensor, T0: torch.Tensor, T_ambient: torch.Tensor, dt: float) -> torch.Tensor:
        """P_loss_seq: (T,). Returns T_seq: (T+1,), including the initial value."""
        T = T0
        Ts = [T]
        for k in range(len(P_loss_seq)):
            T = T + dt * self.rhs(T, P_loss_seq[k], T_ambient)
            Ts.append(T)
        return torch.stack(Ts)
