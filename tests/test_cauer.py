import torch

from atlas_physics.cauer import CauerLadder1D, chain_incidence_matrix, port_vector


def test_topology_is_fixed_buffer_not_parameter():
    model = CauerLadder1D(n_layers=5)
    params = dict(model.named_parameters())
    assert "D" not in params and "d0" not in params
    assert "raw_C" in params and "raw_G" in params


def test_C_and_G_are_strictly_positive_for_any_raw_value():
    model = CauerLadder1D(n_layers=6)
    with torch.no_grad():
        model.raw_C.copy_(torch.randn(6) * 10)
        model.raw_G.copy_(torch.randn(6) * 10)
    assert torch.all(model.C() > 0)
    assert torch.all(model.G() > 0)


def test_A_matrix_is_negative_semidefinite_passivity():
    model = CauerLadder1D(n_layers=8, c_init=0.7, g_init=2.3)
    A, _, _ = model.system_matrices()
    x = torch.randn(200, 8)
    quad_form = (x @ A.T * x).sum(dim=1)
    assert torch.all(quad_form <= 1e-6)


def test_step_response_converges_to_uniform_steady_state():
    # Diffusive settling time along an N-layer chain scales ~ N^2 * (C/G), not
    # linearly in N, so the horizon must be generous relative to the layer count.
    n = 6
    model = CauerLadder1D(n_layers=n, c_init=1.0, g_init=3.0)
    T = 60000
    dt = 1e-3
    u = torch.ones(T)
    xs = model.simulate(u, dt=dt)
    x_final = xs[-1]
    assert torch.allclose(x_final, torch.ones(n), atol=2e-2)


def test_zero_input_stays_at_rest():
    model = CauerLadder1D(n_layers=5)
    u = torch.zeros(100)
    xs = model.simulate(u, dt=1e-3)
    assert torch.allclose(xs, torch.zeros_like(xs))


def test_step_interface_matches_simulate_one_step():
    model = CauerLadder1D(n_layers=4, c_init=1.2, g_init=0.8)
    dt = 1e-3
    x0 = model.initial_state()
    u0 = torch.tensor([0.5])
    _, x1_step = model.step(u0, x0, dt)
    xs = model.simulate(torch.tensor([0.5]), dt=dt, x0=x0.unsqueeze(0))
    assert torch.allclose(x1_step, xs[1], atol=1e-6)


def test_incidence_matrix_shape_and_port_vector():
    D = chain_incidence_matrix(7)
    d0 = port_vector(7)
    assert D.shape == (7, 7)
    assert d0.shape == (7,)
    assert d0.sum() == 1.0
