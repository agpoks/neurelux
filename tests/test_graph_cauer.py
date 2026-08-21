import torch

from atlas_physics.graph_cauer import GraphCauer, build_grid_graph


def test_grid_graph_shapes():
    n_surface, n_depth = 5, 4
    D, port, edge_kind = build_grid_graph(n_surface, n_depth)
    n_nodes = n_surface * n_depth
    n_edges = n_surface + n_surface * (n_depth - 1) + n_depth * (n_surface - 1)
    assert D.shape == (n_edges, n_nodes)
    assert port.shape == (n_edges, n_surface)
    assert len(edge_kind) == n_edges
    assert port.sum().item() == n_surface  # exactly one boundary edge per column


def test_topology_is_fixed_buffer_not_parameter():
    model = GraphCauer(n_surface=4, n_depth=3)
    params = dict(model.named_parameters())
    assert "D" not in params and "port" not in params
    assert "raw_C" in params and "raw_G" in params


def test_C_and_G_positive_for_any_raw_value():
    model = GraphCauer(n_surface=3, n_depth=3)
    with torch.no_grad():
        model.raw_C.copy_(torch.randn_like(model.raw_C) * 10)
        model.raw_G.copy_(torch.randn_like(model.raw_G) * 10)
    assert torch.all(model.C() > 0)
    assert torch.all(model.G() > 0)


def test_A_matrix_negative_semidefinite():
    model = GraphCauer(n_surface=4, n_depth=3, c_init=0.8, g_init=1.5)
    A, _, _ = model.system_matrices()
    x = torch.randn(300, model.n_nodes)
    quad_form = (x @ A.T * x).sum(dim=1)
    assert torch.all(quad_form <= 1e-6)


def test_uniform_step_on_all_ports_converges_to_uniform_state():
    # driving every column identically with the same constant input must settle to a
    # spatially uniform steady state equal to that input everywhere -- no lateral
    # current can flow if there's no lateral difference to drive it, and the vertical
    # chains are individually insulated at the far end (same argument as cauer.py).
    n_surface, n_depth = 3, 4
    model = GraphCauer(n_surface, n_depth, c_init=1.0, g_init=3.0)
    T, dt = 60000, 1e-3
    u = torch.ones(T, n_surface)
    xs = model.simulate(u, dt=dt)
    assert torch.allclose(xs[-1], torch.ones(model.n_nodes), atol=3e-2)


def test_zero_input_stays_at_rest():
    model = GraphCauer(n_surface=3, n_depth=3)
    u = torch.zeros(50, 3)
    xs = model.simulate(u, dt=1e-3)
    assert torch.allclose(xs, torch.zeros_like(xs))


def test_single_port_drive_reduces_to_a_1d_ladder_when_lateral_G_is_zero():
    # with lateral (horizontal) conductances forced to (near) zero, driving one column
    # should not perturb its neighbors at all -- this is the "independent Cauer ladders"
    # baseline used in the notebook.
    n_surface, n_depth = 3, 3
    model = GraphCauer(n_surface, n_depth, c_init=1.0, g_init=2.0)
    with torch.no_grad():
        horiz_mask = ~model.vertical_edge_mask()
        model.raw_G[horiz_mask] = -30.0  # softplus(-30) ~ 0
    T, dt = 2000, 1e-3
    u = torch.zeros(T, n_surface)
    u[:, 0] = 1.0  # drive only column 0
    xs = model.simulate(u, dt=dt)

    def idx(i, j):
        return i * n_depth + j

    other_columns_state = xs[-1, [idx(1, 0), idx(1, 1), idx(2, 0), idx(2, 1)]]
    assert torch.allclose(other_columns_state, torch.zeros_like(other_columns_state), atol=1e-3)
