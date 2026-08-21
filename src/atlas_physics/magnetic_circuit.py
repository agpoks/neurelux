"""Lumped equivalent magnetic circuit for notebooks/03_neural_reluctance_circuit.ipynb
and notebooks/10_velocity_dependent_eddy_weakening.ipynb.

Classical series circuit: MMF Theta = N*I drives flux through core + air gap
reluctance. The air-gap term is always exact geometry (R_gap = g/(mu0*A)) and
is never learned. The uncertain term is the core's relative permeability
mu_r(B, T), which saturates nonlinearly -- this module provides three ways
to represent it:

- `FixedMuR`: a single constant (what ATLAS's assumed linear circuit does)
  -- cannot represent saturation at all.
- `SaturatingMuR`: the closed-form "true" model used to generate synthetic
  data (a Froehlich-style saturating permeability), not a fitted model.
- `NeuralMuR`: PHYSICS-ENCODED. `mu_r0 * (1 + 0.3 * tanh(NN(B, T)))` --
  topology (the circuit equation) stays fixed; a small network only
  *modulates* the base permeability, bounded to +-30%, rather than adding an
  unconstrained correction. This bound is deliberate -- see
  notebooks/03_neural_reluctance_circuit.ipynb's discussion of why an
  unconstrained additive residual (Notebook 02's UDE) extrapolated badly
  while this bounded multiplicative one does not.
- `PhysicsInformedB`: PHYSICS-INFORMED. A free MLP `B_theta(I,g,T)` -- not
  routed through `solve_B` at all -- trained on data plus a soft Ampere's-law
  residual loss evaluated at (I,g,T) collocation points, using a second,
  *unconstrained* permeability network only to compute that residual.
  Nothing in the forward pass stops `B_theta` from violating the circuit
  equation anywhere the residual wasn't evaluated during training.
- `PhysicsConstrainedB`: PHYSICS-CONSTRAINED, a distinct middle ground from
  both of the above. A free MLP with no circuit equation anywhere, but its
  output is passed through `B_cap * tanh(...)`, a hard bound at a known
  physical saturation cap -- structurally incapable of ever predicting a
  flux density the material can't support, regardless of what the network
  computes, even though the rest of the I/g/T dependence is fully free.

`solve_B` inverts Theta -> B for any of `FixedMuR`/`SaturatingMuR`/`NeuralMuR`
via fixed-point (successive-substitution) iteration on the saturable circuit
equation, avoiding the need for a hand-derived or autograd-based Newton step
-- the same fixed-iteration-count-instead-of-adaptive-solver philosophy as
`cauer.py`'s explicit Euler and `hysteresis.py`'s explicit-stepped ODE.

Two further classes represent the *velocity*-dependent MMF loss from
motion-induced eddy currents (Notebook 10), reducing the effective MMF as
Theta_eff = Theta - MMF_loss(v):

- `VelocityMMFLoss`: PHYSICS-ENCODED (functional form). A saturating,
  two-parameter analytical function, MMF_loss(0)=0 exactly, following the
  phenomenological magnetic-equivalent-circuit approach of a real, current
  study of exactly this effect on a magnetic track brake (Ebner, Ploechl &
  Edelmann, 2025, Nonlinear Dynamics).
- `VelocityMMFLossConstrained`: PHYSICS-CONSTRAINED. A free MLP, no assumed
  functional form, hard-bounded to `[0, MMF_cap]` via a sigmoid.
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
    """Single learned scalar permeability -- what ATLAS's assumed linear circuit does.
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


class PhysicsInformedB(nn.Module):
    """PHYSICS-INFORMED: a free B predictor, `B_theta(I,g,T)`, evaluated directly --
    never through `solve_B` -- trained on data *plus* a soft Ampere's-law residual
    loss (`.residual()`) computed with a second, unconstrained permeability network
    used only for that residual. Structurally identical in spirit to a PINN: the
    equation shapes the loss, not the forward computation graph, so nothing prevents
    `B_theta` from disagreeing with Ampere's law anywhere the residual wasn't
    evaluated during training.
    """

    def __init__(
        self, I_scale: float, g_scale: float, T_scale: float, B_scale: float, mu_r0_init: float = 2000.0, hidden: int = 32
    ):
        super().__init__()
        self.I_scale, self.g_scale, self.T_scale, self.B_scale = I_scale, g_scale, T_scale, B_scale
        self.B_net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        # raw_mu_r0 only sets a sane initialization scale (a real core's mu_r is O(1e3),
        # not O(1)) -- the correction added on top is unconstrained, unlike NeuralMuR's
        # bounded +-30% modulation, so training is free to move mu_r arbitrarily far from
        # this starting point in either direction.
        self.raw_mu_r0 = nn.Parameter(torch.tensor(float(mu_r0_init)))
        self.mu_r_net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, I: torch.Tensor, g: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        x = torch.stack([I / self.I_scale, g / self.g_scale, T / self.T_scale], dim=-1)
        return self.B_net(x).squeeze(-1) * self.B_scale

    def mu_r(self, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Unconstrained additive correction on an initial guess -- used only inside
        the residual, never to predict B directly, unlike `NeuralMuR`."""
        x = torch.stack([B / self.B_scale, T / self.T_scale], dim=-1)
        correction = self.mu_r_net(x).squeeze(-1)
        return torch.nn.functional.softplus(self.raw_mu_r0 + correction)

    def residual(self, I: torch.Tensor, g: torch.Tensor, T: torch.Tensor, N: float, l_core: float) -> torch.Tensor:
        """Theta - H_core(B_theta, T)*l_core - (B_theta/mu0)*g, evaluated with this
        model's own current prediction and permeability estimate -- zero exactly
        when B_theta satisfies Ampere's law for the mu_r this model has learned."""
        B = self.forward(I, g, T)
        H_core = B / (MU0 * self.mu_r(B, T))
        Theta = N * I
        return Theta - H_core * l_core - (B / MU0) * g


class PhysicsConstrainedB(nn.Module):
    """PHYSICS-CONSTRAINED: a free black-box map from (I,g,T) to B -- no circuit
    equation anywhere, unlike `PhysicsInformedB` or `NeuralMuR` -- but its output is
    passed through `B_cap * tanh(...)`, a hard bound at a known physical saturation
    cap. Distinct from both other patterns: not a soft loss term (the bound holds
    for *any* input, in or out of distribution, by construction) and not a full
    equation (nothing about *how* B depends on I, g, T below the cap is constrained
    at all). `B_cap` only needs to be known approximately -- e.g. a material
    datasheet's saturation flux density -- not the exact saturation curve.
    """

    def __init__(self, I_scale: float, g_scale: float, T_scale: float, B_cap: float, hidden: int = 32):
        super().__init__()
        self.I_scale, self.g_scale, self.T_scale, self.B_cap = I_scale, g_scale, T_scale, B_cap
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, I: torch.Tensor, g: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        x = torch.stack([I / self.I_scale, g / self.g_scale, T / self.T_scale], dim=-1)
        raw = self.net(x).squeeze(-1)
        return self.B_cap * torch.tanh(raw)


class VelocityMMFLoss(nn.Module):
    """PHYSICS-ENCODED (functional form): motion-induced eddy-current MMF loss for
    notebooks/10_velocity_dependent_eddy_weakening.ipynb, following the phenomenological
    magnetic-equivalent-circuit approach of Ebner, Ploechl & Edelmann (2025, Nonlinear
    Dynamics) -- a real 2025 TU Wien study of exactly this effect on a magnetic track
    brake. As velocity increases, motion-induced eddy currents in the rail oppose the
    driving MMF, reducing the effective flux; a first-principles derivation for a real,
    segmented 3D geometry is intractable in a lumped model (as that paper notes), so an
    analytical function is fit to the velocity dependence instead, folded into the
    circuit as Theta_eff = Theta - MMF_loss(v).

    Saturating form -- consistent with a magnetic-Reynolds-number argument (loss grows
    with v but the rail cannot expel more flux than the source drives): MMF_loss(0)=0
    exactly (structural, not fit), MMF_loss(v) -> MMF_max as v -> infinity, set by a
    single characteristic velocity v_c. Only MMF_max and v_c are learned.
    """

    def __init__(self, MMF_max_init: float, v_c_init: float):
        super().__init__()
        self.raw_MMF_max = nn.Parameter(torch.tensor(float(MMF_max_init)))
        self.raw_v_c = nn.Parameter(torch.tensor(float(v_c_init)))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        MMF_max = torch.nn.functional.softplus(self.raw_MMF_max)
        v_c = torch.nn.functional.softplus(self.raw_v_c) + 1e-3
        return MMF_max * v / (v + v_c)


class VelocityMMFLossConstrained(nn.Module):
    """PHYSICS-CONSTRAINED: a free MLP mapping v -> MMF loss, no assumed functional
    form (unlike `VelocityMMFLoss`) -- but hard-bounded to `[0, MMF_cap]` via a
    sigmoid, so it can never drive the effective MMF negative or remove more MMF than
    a known physical cap allows, regardless of the input. Not guaranteed monotonic in
    v, unlike `VelocityMMFLoss`'s saturating form -- only the range is constrained.
    """

    def __init__(self, v_scale: float, MMF_cap: float, hidden: int = 32):
        super().__init__()
        self.v_scale, self.MMF_cap = v_scale, MMF_cap
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        x = (v / self.v_scale).unsqueeze(-1)
        raw = self.net(x).squeeze(-1)
        return self.MMF_cap * torch.sigmoid(raw)
