"""Magnetic hysteresis models for notebooks/02_hysteresis_material_model.ipynb.

Three model families, compared on the same data:

- ``MLPHysteresis``: memoryless baseline. PHYSICS-GUIDED at most (if fed
  physically-motivated features); cannot represent path-dependence at all,
  since it has no state.
- ``GRUHysteresis``: generic recurrent baseline. No physics anywhere -- a
  black box that *can* learn memory from data, unlike the MLP.
- ``JilesAtherton``: PHYSICS-ENCODED. The state recursion for the
  magnetization M is exactly the (simplified, explicit-stepped) classical
  Jiles-Atherton ODE (Jiles & Atherton, 1986 -- see papers/references.bib),
  with only the five physical parameters (Ms, a, alpha, k, c) learned,
  always through a positivity-preserving parameterization. Optionally wraps
  a small neural residual added to dM/dH -- a Universal Differential
  Equation (Rackauckas et al., 2020, papers/references.bib): the known ODE
  structure is kept exactly, and only an unknown correction term is
  replaced by a network, trained end-to-end through the same explicit-Euler
  rollout used everywhere else in this project.
"""

from __future__ import annotations

import torch
from torch import nn


def langevin(x: torch.Tensor) -> torch.Tensor:
    """coth(x) - 1/x, the anhysteretic (Langevin) magnetization shape, safe at x=0."""
    small = x.abs() < 1e-4
    x_safe = torch.where(small, torch.ones_like(x), x)
    val = 1.0 / torch.tanh(x_safe) - 1.0 / x_safe
    return torch.where(small, x / 3.0, val)


def dlangevin(x: torch.Tensor) -> torch.Tensor:
    """d/dx of `langevin`, safe at x=0."""
    small = x.abs() < 1e-4
    x_safe = torch.where(small, torch.ones_like(x), x)
    csch2 = 1.0 / torch.sinh(x_safe) ** 2
    val = 1.0 / x_safe ** 2 - csch2
    return torch.where(small, torch.full_like(x, 1.0 / 3.0), val)


MU0 = 4 * torch.pi * 1e-7


class JilesAtherton(nn.Module):
    """Physics-encoded (optionally UDE) scalar Jiles-Atherton hysteresis model.

    Operates in *normalized* units: H is expected pre-divided by a field
    scale ``Hs`` and M by a magnetization scale ``Ms_scale`` chosen from the
    data (see notebooks/02 for how these are picked) -- this keeps the five
    physical parameters O(1) and the optimizer well-conditioned, avoiding
    the overflow/instability that fitting in raw A/m, A/m units produces.

    ``residual`` (if given) is a small `nn.Module` mapping
    ``(He_n, M_n, delta) -> correction`` added directly to ``dM/dH`` inside
    the rollout -- the Universal Differential Equation variant. Leave it
    ``None`` for the pure physics-encoded model.
    """

    def __init__(self, residual: nn.Module | None = None):
        super().__init__()
        self.raw_Ms = nn.Parameter(torch.tensor(0.0))
        self.raw_a = nn.Parameter(torch.tensor(0.0))
        self.raw_alpha = nn.Parameter(torch.tensor(-3.0))
        self.raw_k = nn.Parameter(torch.tensor(0.0))
        self.raw_c = nn.Parameter(torch.tensor(0.0))
        self.residual = residual

    def params(self):
        Ms = torch.nn.functional.softplus(self.raw_Ms) + 0.3
        a = torch.nn.functional.softplus(self.raw_a) + 0.05
        alpha = torch.nn.functional.softplus(self.raw_alpha)
        k = torch.nn.functional.softplus(self.raw_k) + 0.05
        c = torch.sigmoid(self.raw_c)
        return Ms, a, alpha, k, c

    def dM_dH(self, He: torch.Tensor, M: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        Ms, a, alpha, k, c = self.params()
        x = He / a
        Man = Ms * langevin(x)
        dMan_dHe = (Ms / a) * dlangevin(x)
        denom = k * delta - alpha * (Man - M)
        dMirr_dHe = (Man - M) / denom
        dM_dH = ((1 - c) * dMirr_dHe + c * dMan_dHe) / (1 + c)
        if self.residual is not None:
            feat = torch.stack([He, M, delta], dim=-1)
            dM_dH = dM_dH + self.residual(feat).squeeze(-1)
        return dM_dH

    def simulate(self, H_seq: torch.Tensor, M0: torch.Tensor) -> torch.Tensor:
        """H_seq: (T,) normalized field trajectory. Returns M: (T,) normalized magnetization,
        including the initial value (so len(M) == len(H_seq)).
        """
        _, _, alpha, _, _ = self.params()
        M = M0
        out = [M]
        for i in range(1, len(H_seq)):
            Hk = H_seq[i]
            dH = H_seq[i] - H_seq[i - 1]
            delta = torch.sign(dH) if dH.item() != 0 else torch.tensor(1.0, dtype=H_seq.dtype)
            He = Hk + alpha * M
            M = M + self.dM_dH(He, M, delta) * dH
            out.append(M)
        return torch.stack(out)


class MLPHysteresis(nn.Module):
    """Memoryless baseline: normalized H_k -> normalized B_k, pointwise. No state at all,
    so it structurally cannot represent path-dependence (the defining feature of hysteresis).
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, H_seq: torch.Tensor) -> torch.Tensor:
        return self.net(H_seq.unsqueeze(-1)).squeeze(-1)


class GRUHysteresis(nn.Module):
    """Generic recurrent baseline: (H_k, dH_k) -> B_k via a GRU. No physics anywhere;
    memory (if any) is whatever the GRU's hidden state learns from data.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, H_seq: torch.Tensor) -> torch.Tensor:
        dH = torch.diff(H_seq, prepend=H_seq[:1])
        x = torch.stack([H_seq, dH], dim=-1).unsqueeze(0)  # (1, T, 2)
        h, _ = self.gru(x)
        return self.readout(h).squeeze(0).squeeze(-1)
