import torch

from atlas_physics.friction import BlackBoxFriction, GuidedFriction, friction_force, true_mu
from atlas_physics.thermal import LumpedThermal


def test_friction_force_is_exactly_mu_times_FN():
    mu = torch.tensor([0.1, 0.2, 0.35])
    F_N = torch.tensor([100.0, 500.0, 1000.0])
    assert torch.allclose(friction_force(mu, F_N), mu * F_N)


def test_true_mu_decreases_with_velocity():
    v = torch.linspace(0.1, 50, 20)
    T = torch.full_like(v, 25.0)
    mu = true_mu(v, T)
    assert torch.all(torch.diff(mu) <= 0)


def test_guided_friction_mu_positive_for_any_raw_weights():
    model = GuidedFriction()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 10)
    v = torch.linspace(0, 50, 10)
    F_N = torch.full_like(v, 500.0)
    T = torch.full_like(v, 25.0)
    assert torch.all(model.mu(v, F_N, T) > 0)


def test_guided_friction_vanishes_at_zero_normal_force():
    # F_R = mu * F_N is hard-coded, so F_N=0 must give F_R=0 regardless of mu --
    # mu itself is allowed to depend on F_N (per PLAN.md's mu(v,F_N,T) spec), so
    # F_R need not be linear in F_N in general, but this zero must hold exactly.
    model = GuidedFriction()
    v = torch.full((5,), 10.0)
    T = torch.full((5,), 25.0)
    F_N = torch.zeros(5)
    F_R = model(v, F_N, T)
    assert torch.allclose(F_R, torch.zeros_like(F_R))


def test_blackbox_friction_shape():
    model = BlackBoxFriction()
    v = torch.linspace(0, 50, 10)
    F_N = torch.full_like(v, 500.0)
    T = torch.full_like(v, 25.0)
    out = model(v, F_N, T)
    assert out.shape == v.shape


def test_thermal_params_positive():
    model = LumpedThermal()
    with torch.no_grad():
        model.raw_C_th.copy_(torch.tensor(-50.0))
        model.raw_hA.copy_(torch.tensor(-50.0))
    C_th, hA = model.params()
    assert C_th > 0 and hA > 0


def test_thermal_reaches_steady_state_consistent_with_balance():
    model = LumpedThermal(C_th_init=500.0, hA_init=5.0)
    T_ambient = torch.tensor(25.0)
    P_loss = torch.full((20000,), 100.0)
    T_seq = model.simulate(P_loss, T0=T_ambient, T_ambient=T_ambient, dt=1.0)
    # steady state: P_loss = hA*(T-T_ambient) => T = T_ambient + P_loss/hA = 25 + 100/5 = 45
    assert abs(T_seq[-1].item() - 45.0) < 0.5


def test_thermal_zero_loss_stays_at_ambient():
    model = LumpedThermal()
    T_ambient = torch.tensor(25.0)
    P_loss = torch.zeros(100)
    T_seq = model.simulate(P_loss, T0=T_ambient, T_ambient=T_ambient, dt=1.0)
    assert torch.allclose(T_seq, torch.full_like(T_seq, 25.0), atol=1e-4)
