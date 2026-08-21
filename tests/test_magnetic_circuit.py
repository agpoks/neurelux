import torch

from atlas_physics.magnetic_circuit import (
    MU0,
    FixedMuR,
    NeuralMuR,
    PhysicsConstrainedB,
    PhysicsInformedB,
    SaturatingMuR,
    VelocityMMFLoss,
    VelocityMMFLossConstrained,
    solve_B,
)


def test_fixed_mu_r_positive_and_constant_in_B():
    model = FixedMuR(mu_r_init=2000.0)
    B = torch.tensor([0.0, 0.5, 1.0, 1.5])
    mu_r = model(B, torch.full_like(B, 25.0))
    assert torch.all(mu_r > 0)
    assert torch.allclose(mu_r, mu_r[0] * torch.ones_like(mu_r))


def test_saturating_mu_r_decreases_with_B():
    model = SaturatingMuR(mu_r0_ref=2000.0, Bsat_ref=1.6, n_exp=7.0)
    B = torch.linspace(0.01, 1.5, 10)
    mu_r = model(B, torch.full_like(B, 25.0))
    assert torch.all(torch.isfinite(mu_r))
    assert torch.all(torch.diff(mu_r) <= 0)  # monotonically non-increasing as B grows


def test_neural_mu_r_bounded_within_30_percent_of_base():
    model = NeuralMuR(mu_r0_init=2000.0, B_scale=1.5, T_scale=100.0)
    with torch.no_grad():
        for p in model.net.parameters():
            p.copy_(torch.randn_like(p) * 20)  # push the network to its extremes
    B = torch.linspace(-3.0, 3.0, 50)  # includes far-out-of-distribution inputs
    T = torch.linspace(-200.0, 200.0, 50)
    mu_r = model(B, T)
    base = torch.nn.functional.softplus(model.raw_mu_r0)
    assert torch.all(mu_r >= 0.7 * base - 1e-3)
    assert torch.all(mu_r <= 1.3 * base + 1e-3)


def test_solve_B_recovers_B_for_the_true_saturating_system():
    true_system = SaturatingMuR(mu_r0_ref=2000.0, Bsat_ref=1.6, n_exp=7.0)
    l_core = 0.2
    B_true = torch.linspace(0.05, 1.3, 30)
    g = torch.full_like(B_true, 1.75e-3)
    T = torch.full_like(B_true, 40.0)
    Theta = true_system.H_core(B_true, T) * l_core + (B_true / MU0) * g

    B_solved = solve_B(Theta, g, T, true_system, l_core, n_iter=15)
    assert torch.allclose(B_solved, B_true, atol=1e-4)


def test_physics_constrained_b_never_exceeds_cap():
    model = PhysicsConstrainedB(I_scale=1.0, g_scale=1e-3, T_scale=100.0, B_cap=1.8)
    with torch.no_grad():
        for p in model.net.parameters():
            p.copy_(torch.randn_like(p) * 50)  # push the network to its extremes
    I = torch.linspace(-1000.0, 1000.0, 100)  # wildly out-of-distribution
    g = torch.full_like(I, 0.15e-3)
    T = torch.full_like(I, 40.0)
    B = model(I, g, T)
    assert torch.all(torch.isfinite(B))
    assert torch.all(B.abs() <= 1.8 + 1e-6)


def test_physics_informed_b_residual_matches_ampere_law_manually():
    model = PhysicsInformedB(I_scale=1.0, g_scale=1e-3, T_scale=100.0, B_scale=1.5)
    N, l_core = 500, 0.2
    I = torch.tensor([0.3, 0.6])
    g = torch.tensor([0.15e-3, 0.15e-3])
    T = torch.tensor([40.0, 40.0])
    with torch.no_grad():
        B = model(I, g, T)
        mu_r = model.mu_r(B, T)
        H_core = B / (MU0 * mu_r)
        expected = N * I - H_core * l_core - (B / MU0) * g
        resid = model.residual(I, g, T, N, l_core)
    assert torch.allclose(resid, expected, atol=1e-6)


def test_physics_informed_mu_r_stays_positive_and_unbounded():
    model = PhysicsInformedB(I_scale=1.0, g_scale=1e-3, T_scale=100.0, B_scale=1.5)
    B = torch.linspace(-5.0, 5.0, 50)
    T = torch.linspace(-500.0, 500.0, 50)
    mu_r = model.mu_r(B, T)
    assert torch.all(mu_r > 0)
    assert torch.all(torch.isfinite(mu_r))


def test_velocity_mmf_loss_zero_at_zero_and_saturates_monotonically():
    model = VelocityMMFLoss(MMF_max_init=50.0, v_c_init=5.0)
    v = torch.linspace(0.0, 200.0, 200)
    loss = model(v)
    assert torch.isclose(model(torch.tensor([0.0])), torch.tensor([0.0]), atol=1e-6)
    assert torch.all(torch.diff(loss) >= 0)  # monotonically non-decreasing in v
    MMF_max = torch.nn.functional.softplus(model.raw_MMF_max)
    assert torch.all(loss <= MMF_max + 1e-4)  # saturates, never exceeds MMF_max


def test_velocity_mmf_loss_constrained_stays_within_cap():
    model = VelocityMMFLossConstrained(v_scale=30.0, MMF_cap=80.0)
    with torch.no_grad():
        for p in model.net.parameters():
            p.copy_(torch.randn_like(p) * 50)  # push the network to its extremes
    v = torch.linspace(-500.0, 500.0, 100)  # wildly out-of-distribution
    loss = model(v)
    assert torch.all(torch.isfinite(loss))
    assert torch.all(loss >= -1e-6)
    assert torch.all(loss <= 80.0 + 1e-6)


def test_solve_B_matches_closed_form_for_constant_permeability():
    model = FixedMuR(mu_r_init=2000.0)
    l_core = 0.2
    Theta = torch.tensor([500.0, 1000.0, 1500.0])
    g = torch.tensor([1.5e-3, 2.0e-3, 2.5e-3])
    T = torch.zeros(3)
    B_solved = solve_B(Theta, g, T, model, l_core, n_iter=5)
    mu_r = torch.nn.functional.softplus(model.raw_mu_r)
    B_closed_form = MU0 * mu_r * Theta / (l_core + mu_r * g)
    assert torch.allclose(B_solved, B_closed_form, atol=1e-6)
