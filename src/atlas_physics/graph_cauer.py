"""Surface x depth Graph-Cauer network for notebooks/04_graph_cauer_surface_depth.ipynb.

Generalizes cauer.py's 1D depth-only ladder to a 2D grid: `n_surface` columns
(positions along the rail) x `n_depth` layers (depth into the rail), each
column independently driven at its surface node (depth 0) by its own port.

- Vertical edges (within a column): identical physics to cauer.py -- 1D
  diffusion / skin-effect penetration.
- Horizontal edges (across columns, at a fixed depth): field/eddy-current
  redistribution *along* the rail surface -- the dimension a 1D ladder
  cannot represent at all.

Same conservation form as cauer.py, just with a multi-port drive:

    C dx/dt = -D^T G D x + B_u u,   B_u = -D^T G Port

`D` (topology) and `Port` (which edge is which column's boundary drive) are
FIXED buffers -- PHYSICS-ENCODED, exactly as in cauer.py. Only per-node `C_i`
and per-edge `G_e` are learned, always through softplus.

Velocity dependence is introduced through the *excitation* `u(t)` (a source
pattern that sweeps across surface positions as a function of velocity),
not through the graph's conductances -- see the notebook for why: this
keeps the state matrix exactly as provably passive/negative-semidefinite as
cauer.py's (symmetric G on every edge), rather than needing directional
conductances whose passivity would have to be argued for separately.
"""

from __future__ import annotations

import torch
from torch import nn


def build_grid_graph(n_surface: int, n_depth: int):
    """Build the fixed topology of an n_surface x n_depth grid.

    Node index: idx(i, j) = i * n_depth + j  (i = surface position, j = depth,
    j=0 is the surface-adjacent layer for that column).

    Returns
    -------
    D : (M, N) incidence matrix, same tail=+1/head=-1 convention as cauer.py.
        Row order: n_surface boundary edges first, then n_surface*(n_depth-1)
        vertical edges (column-major), then n_depth*(n_surface-1) horizontal
        edges (depth-major).
    port : (M, n_surface) -- port[e, k] = 1 iff edge e is column k's boundary
        edge (tail = the external port for that column), else 0. Generalizes
        cauer.py's single `d0` vector to one column per surface port.
    edge_kind : list[str] of length M, "vertical" or "horizontal", for
        plotting/inspection.
    """

    def idx(i, j):
        return i * n_depth + j

    N = n_surface * n_depth
    rows_D = []
    rows_port = []
    edge_kind = []

    # boundary edges: one per surface column, tail = external port k, head = idx(k, 0)
    for k in range(n_surface):
        d_row = torch.zeros(N)
        d_row[idx(k, 0)] = -1.0
        rows_D.append(d_row)
        p_row = torch.zeros(n_surface)
        p_row[k] = 1.0
        rows_port.append(p_row)
        edge_kind.append("vertical")

    # vertical internal edges: idx(i,j-1) -> idx(i,j), within each column
    for i in range(n_surface):
        for j in range(1, n_depth):
            d_row = torch.zeros(N)
            d_row[idx(i, j - 1)] = 1.0
            d_row[idx(i, j)] = -1.0
            rows_D.append(d_row)
            rows_port.append(torch.zeros(n_surface))
            edge_kind.append("vertical")

    # horizontal edges: idx(i,j) -> idx(i+1,j), at each depth level
    for j in range(n_depth):
        for i in range(n_surface - 1):
            d_row = torch.zeros(N)
            d_row[idx(i, j)] = 1.0
            d_row[idx(i + 1, j)] = -1.0
            rows_D.append(d_row)
            rows_port.append(torch.zeros(n_surface))
            edge_kind.append("horizontal")

    D = torch.stack(rows_D)
    port = torch.stack(rows_port)
    return D, port, edge_kind


class GraphCauer(nn.Module):
    """Physics-encoded surface x depth Graph-Cauer network.

    Only `raw_C` (per node) and `raw_G` (per edge) are trainable; `D` and
    `port` are fixed buffers, exactly as in `cauer.CauerLadder1D`.
    """

    def __init__(self, n_surface: int, n_depth: int, c_init: float = 1.0, g_init: float = 1.0):
        super().__init__()
        self.n_surface, self.n_depth = n_surface, n_depth
        self.n_nodes = n_surface * n_depth
        D, port, edge_kind = build_grid_graph(n_surface, n_depth)
        self.register_buffer("D", D)
        self.register_buffer("port", port)
        self.edge_kind = edge_kind
        n_edges = D.shape[0]

        def inv_softplus(y: float) -> torch.Tensor:
            y_t = torch.tensor(float(y))
            return y_t + torch.log(-torch.expm1(-y_t))

        self.raw_C = nn.Parameter(inv_softplus(c_init) * torch.ones(self.n_nodes))
        self.raw_G = nn.Parameter(inv_softplus(g_init) * torch.ones(n_edges))

    def C(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_C)

    def G(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_G)

    def vertical_edge_mask(self) -> torch.Tensor:
        return torch.tensor([k == "vertical" for k in self.edge_kind])

    def system_matrices(self):
        D = self.D
        G = torch.diag(self.G())
        C = self.C()
        A = -(D.T @ G @ D)
        B_u = -(D.T @ G @ self.port)  # (N, n_surface)
        return A, B_u, C

    def rhs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """x: (..., N), u: (..., n_surface)."""
        A, B_u, C = self.system_matrices()
        dx = x @ A.T + u @ B_u.T
        return dx / C

    def simulate(self, u: torch.Tensor, dt: float, x0: torch.Tensor | None = None) -> torch.Tensor:
        """u: (T, n_surface) or (B, T, n_surface). Returns x: (T+1, N) or (B, T+1, N)."""
        squeeze_batch = u.dim() == 2
        if squeeze_batch:
            u = u.unsqueeze(0)
        B, T, _ = u.shape
        x = torch.zeros(B, self.n_nodes, dtype=u.dtype, device=u.device) if x0 is None else x0
        xs = [x]
        for t in range(T):
            dx = self.rhs(x, u[:, t, :])
            x = x + dt * dx
            xs.append(x)
        xs = torch.stack(xs, dim=1)
        return xs.squeeze(0) if squeeze_batch else xs

    def initial_state(self, batch_shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.zeros(*batch_shape, self.n_nodes)

    def step(self, u_t: torch.Tensor, x: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        dx = self.rhs(x, u_t)
        x_next = x + dt * dx
        return x_next, x_next
