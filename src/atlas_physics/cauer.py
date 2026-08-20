"""1D Cauer (RC-ladder) network for magnetic diffusion / skin effect.

Physics being encoded (see notebooks/01_skin_effect_cauer_synthetic.ipynb for the
full derivation): the 1D magnetic diffusion equation

    mu(z) dH/dt = d/dz( rho(z) dH/dz ),   rho = 1/sigma

discretized through depth into N layers is exactly a series-conductance /
shunt-capacitance ladder network (a Cauer network):

    C dx/dt = -D^T G D x + B_u u

x in R^N is the (depth-averaged) field per layer, layer 0 nearest the surface.
D is the FIXED chain incidence matrix of the ladder topology (PHYSICS-ENCODED:
it is a constant buffer, never a trained parameter). Only the per-node
capacitances C_i = mu_i * dz_i and per-edge conductances G_i = 1/(rho_i * dz_i)
are learned, always through softplus so that C_i > 0, G_i > 0 (this preserves
passivity/dissipativity of the ladder for any trained value).
"""

from __future__ import annotations

import torch
from torch import nn


def chain_incidence_matrix(n_nodes: int) -> torch.Tensor:
    """Fixed incidence matrix D (n_nodes x n_nodes) of a 1D chain ladder.

    Edge 0: external port -> node 0 (surface coupling). Row 0 carries only the
    "head" (-1) entry for node 0; the port itself is not a state and is folded
    into B_u via ``port_vector`` below instead of being a column of D.
    Edge e (e = 1 .. n_nodes-1): node e-1 -> node e, tail=+1 at e-1, head=-1 at e.
    The last node has no outgoing edge -> insulated / zero-flux far boundary,
    i.e. no leakage path to a return/ground beyond the deepest layer.
    """
    D = torch.zeros(n_nodes, n_nodes)
    D[0, 0] = -1.0
    for e in range(1, n_nodes):
        D[e, e - 1] = 1.0
        D[e, e] = -1.0
    return D


def port_vector(n_nodes: int) -> torch.Tensor:
    """d0: the port is the tail of edge 0 only -> unit vector e_0."""
    d0 = torch.zeros(n_nodes)
    d0[0] = 1.0
    return d0


class CauerLadder1D(nn.Module):
    """Physics-encoded 1D Cauer ladder for magnetic skin-effect diffusion.

    Only ``raw_C`` and ``raw_G`` are trainable; the topology (``D``, ``d0``) is
    a fixed buffer and can never be changed by gradient descent.
    """

    def __init__(self, n_layers: int, c_init: float = 1.0, g_init: float = 1.0):
        super().__init__()
        self.n_layers = n_layers
        self.register_buffer("D", chain_incidence_matrix(n_layers))
        self.register_buffer("d0", port_vector(n_layers))

        def inv_softplus(y: float) -> torch.Tensor:
            y_t = torch.tensor(float(y))
            return y_t + torch.log(-torch.expm1(-y_t))

        self.raw_C = nn.Parameter(inv_softplus(c_init) * torch.ones(n_layers))
        self.raw_G = nn.Parameter(inv_softplus(g_init) * torch.ones(n_layers))

    def C(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_C)

    def G(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_G)

    def system_matrices(self):
        """Return (A, B_u, C) with A = -D^T G D, B_u = -D^T G d0 (both derived
        directly from KCL, not hand-simplified), C the nodal capacitance vector.
        """
        D = self.D
        G = torch.diag(self.G())
        C = self.C()
        A = -(D.T @ G @ D)
        B_u = -(D.T @ G @ self.d0)
        return A, B_u, C

    def rhs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """dx/dt = C^-1 (A x + B_u u).  x: (..., N), u: (..., 1)."""
        A, B_u, C = self.system_matrices()
        dx = x @ A.T + u * B_u
        return dx / C

    def simulate(
        self, u: torch.Tensor, dt: float, x0: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Explicit-Euler rollout (fixed-step, Simulink-compatible integrator).

        u: (T,) or (B, T) excitation sequence (scalar surface drive per step).
        Returns x: (T+1, N) or (B, T+1, N), including the initial state.
        """
        squeeze_batch = u.dim() == 1
        if squeeze_batch:
            u = u.unsqueeze(0)
        B, T = u.shape
        x = (
            torch.zeros(B, self.n_layers, dtype=u.dtype, device=u.device)
            if x0 is None
            else x0
        )
        xs = [x]
        for t in range(T):
            dx = self.rhs(x, u[:, t : t + 1])
            x = x + dt * dx
            xs.append(x)
        xs = torch.stack(xs, dim=1)
        return xs.squeeze(0) if squeeze_batch else xs

    def initial_state(self, batch_shape: tuple[int, ...] = ()) -> torch.Tensor:
        """Simulink-interface-style explicit initial state (see interfaces/simulink)."""
        return torch.zeros(*batch_shape, self.n_layers)

    def step(self, u_t: torch.Tensor, x: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """One fixed-step Euler update: (outputs, next_state).

        Matches the deployment contract in interfaces/simulink/README.md so this
        module can later be driven from a Simulink S-function without change.
        """
        dx = self.rhs(x, u_t)
        x_next = x + dt * dx
        outputs = x_next[..., :1]  # surface-node observable by convention
        return outputs, x_next
