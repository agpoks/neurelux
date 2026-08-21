"""Friction model for notebooks/08_friction_temperature_model.ipynb.

PHYSICS-GUIDED: the Coulomb friction relation F_R = mu * F_N is hard-coded
and never learned (`friction_force`). Only the coefficient mu itself is
learned, as a function of (v, F_N, T), always through softplus so mu stays
positive.

`GuidedFriction` keeps this structure explicit: predict mu, then apply the
fixed formula. `BlackBoxFriction` is the contrasting baseline: predict F_R
directly from (v, F_N, T) with no hard-coded relation to F_N at all.
"""

from __future__ import annotations

import torch
from torch import nn


def friction_force(mu: torch.Tensor, F_N: torch.Tensor) -> torch.Tensor:
    """F_R = mu * F_N -- hard-coded, never learned."""
    return mu * F_N


def true_mu(v: torch.Tensor, T: torch.Tensor, mu0: float = 0.35, a: float = 0.15, v0: float = 5.0, b: float = -0.0015) -> torch.Tensor:
    """Illustrative synthetic friction-coefficient trend (not measured -- see PLAN.md §0):
    Stribeck-like decrease with velocity, mild linear decrease with temperature.
    """
    return mu0 - a * v / (v + v0) + b * (T - 25.0)


class GuidedFriction(nn.Module):
    """PHYSICS-GUIDED: learns mu(v, F_N, T); F_R = mu * F_N stays hard-coded (`friction_force`)."""

    def __init__(self, hidden: int = 32, v_scale: float = 50.0, FN_scale: float = 2000.0, T_scale: float = 150.0):
        super().__init__()
        self.v_scale, self.FN_scale, self.T_scale = v_scale, FN_scale, T_scale
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def mu(self, v: torch.Tensor, F_N: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        x = torch.stack([v / self.v_scale, F_N / self.FN_scale, T / self.T_scale], dim=-1)
        return torch.nn.functional.softplus(self.net(x).squeeze(-1))

    def forward(self, v: torch.Tensor, F_N: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return friction_force(self.mu(v, F_N, T), F_N)


class BlackBoxFriction(nn.Module):
    """No hard-coded relation to F_N at all -- predicts F_R directly."""

    def __init__(self, hidden: int = 32, v_scale: float = 50.0, FN_scale: float = 2000.0, T_scale: float = 150.0):
        super().__init__()
        self.v_scale, self.FN_scale, self.T_scale = v_scale, FN_scale, T_scale
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, v: torch.Tensor, F_N: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        x = torch.stack([v / self.v_scale, F_N / self.FN_scale, T / self.T_scale], dim=-1)
        return self.net(x).squeeze(-1) * self.FN_scale
