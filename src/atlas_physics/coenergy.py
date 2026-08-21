"""Energy-consistent force models for notebooks/07_energy_consistent_force.ipynb.

PHYSICS-ENCODED: `CoenergyNet` learns a single scalar potential
$W'_\\theta(I,g,T)$; flux linkage and force are obtained as its *exact*
partial derivatives via autograd ($\\lambda = \\partial W'/\\partial I$,
$F = -\\partial W'/\\partial g$), so they cannot become mutually
inconsistent -- a Maxwell-type reciprocity relation
$\\partial\\lambda/\\partial g = -\\partial F/\\partial I$ holds automatically
because mixed partial derivatives of the same smooth function commute.

`IndependentNets` is the contrasting baseline: two separate networks for
flux and force, with nothing enforcing that relation -- included so the
notebook can check it directly rather than assume it.

`true_coenergy` defines a synthetic ground-truth potential (illustrative,
not measured -- see PLAN.md §0) whose flux/force pair is generated the
same way (autograd of one function), so the "true" data handed to every
model is itself, by construction, perfectly self-consistent.
"""

from __future__ import annotations

import torch
from torch import nn

MU0 = 4 * torch.pi * 1e-7


def true_coenergy(I: torch.Tensor, g: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Illustrative saturating co-energy: A(g,T) * Isat(g,T)^2 * log(cosh(I / Isat(g,T))).

    log(cosh(x)) ~ x^2/2 for small x (linear-inductor limit) and ~|x| for
    large x (saturating flux linkage, since d/dx[log cosh(x)] = tanh(x) is
    bounded) -- a smooth, differentiable stand-in for real magnetic
    saturation, not derived from measured ATLAS data.
    """
    A = 1e-3 / g * (1.0 - 0.05 * (T - 25.0) / 100.0)
    I_sat = 2.0 * (1.0 + 0.5 * g / 2e-3)
    x = I / I_sat
    return A * I_sat**2 * torch.log(torch.cosh(x))


def flux_and_force_from_potential(w_fn, I: torch.Tensor, g: torch.Tensor, T: torch.Tensor):
    """lambda = dW'/dI, F = -dW'/dg, both via autograd of the same scalar function.
    `create_graph=True` so a training loss built on the returned tensors can
    still be backpropagated through (needed for CoenergyNet; harmless for
    `true_coenergy`, which has no trainable parameters).
    """
    I = I.requires_grad_(True)
    g = g.requires_grad_(True)
    W = w_fn(I, g, T)
    dWdI, dWdg = torch.autograd.grad(W.sum(), [I, g], create_graph=True)
    return dWdI, -dWdg


class CoenergyNet(nn.Module):
    """Physics-encoded: a single learned potential; flux and force are its exact
    partial derivatives (autograd), not two independently-fit quantities.
    """

    def __init__(self, hidden: int = 32, I_scale: float = 5.0, g_scale: float = 3e-3, T_scale: float = 100.0):
        super().__init__()
        self.I_scale, self.g_scale, self.T_scale = I_scale, g_scale, T_scale
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def potential(self, I: torch.Tensor, g: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        x = torch.stack([I / self.I_scale, g / self.g_scale, T / self.T_scale], dim=-1)
        return self.net(x).squeeze(-1)

    def flux_and_force(self, I: torch.Tensor, g: torch.Tensor, T: torch.Tensor):
        return flux_and_force_from_potential(self.potential, I, g, T)


class IndependentNets(nn.Module):
    """Baseline: flux and force from two separate networks -- nothing ties them
    to a common potential, so the Maxwell reciprocity relation is not enforced.
    """

    def __init__(self, hidden: int = 32, I_scale: float = 5.0, g_scale: float = 3e-3, T_scale: float = 100.0):
        super().__init__()
        self.I_scale, self.g_scale, self.T_scale = I_scale, g_scale, T_scale

        def make_net():
            return nn.Sequential(
                nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
            )

        self.flux_net = make_net()
        self.force_net = make_net()

    def _inputs(self, I, g, T):
        return torch.stack([I / self.I_scale, g / self.g_scale, T / self.T_scale], dim=-1)

    def flux_and_force(self, I: torch.Tensor, g: torch.Tensor, T: torch.Tensor):
        x = self._inputs(I, g, T)
        return self.flux_net(x).squeeze(-1), self.force_net(x).squeeze(-1)
