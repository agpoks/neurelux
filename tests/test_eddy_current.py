import numpy as np
import torch

from atlas_physics.eddy_current import (
    MU0,
    EddyLevitationAveraged,
    analytical_semi_infinite_H,
    skin_depth,
)


def test_skin_depth_decreases_with_frequency():
    sigma = 3.526e7
    d50 = skin_depth(50, sigma)
    d200 = skin_depth(200, sigma)
    assert d200 < d50
    # skin depth ~ 1/sqrt(f): delta(200)/delta(50) should be 1/sqrt(4) = 0.5
    assert abs(d200 / d50 - 0.5) < 1e-6


def test_analytical_H_matches_surface_boundary_condition():
    sigma = 3.526e7
    t = np.linspace(0, 0.02, 20)
    H = analytical_semi_infinite_H(0.0, t, f=50, sigma=sigma, H0=2.0)
    assert np.allclose(H, 2.0 * np.cos(2 * np.pi * 50 * t), atol=1e-9)


def test_analytical_H_decays_with_depth():
    sigma = 3.526e7
    z = np.linspace(0, 0.02, 10)
    H = analytical_semi_infinite_H(z, 0.0, f=50, sigma=sigma, H0=1.0)
    assert np.all(np.diff(np.abs(H)) <= 1e-12)  # amplitude envelope non-increasing with depth


def test_eddy_levitation_params_positive_for_any_raw_value():
    model = EddyLevitationAveraged(m_plate=0.107, I_hat=20.0, f0=50.0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 10)
    M0, z0, R2, L2, damp = model.params()
    assert M0 > 0 and z0 > 0 and R2 > 0 and L2 > 0 and damp >= 0


def test_eddy_levitation_force_decreases_with_height():
    model = EddyLevitationAveraged(m_plate=0.107, I_hat=20.0, f0=50.0)
    z = torch.linspace(0.001, 0.05, 20)
    F = model.force(z)
    assert torch.all(F > 0)  # always repulsive
    assert torch.all(torch.diff(F) <= 0)  # weaker further away


def test_eddy_levitation_simulate_shape_and_finite():
    model = EddyLevitationAveraged(m_plate=0.107, I_hat=20.0, f0=50.0)
    t_grid = np.linspace(0, 0.1, 15)
    z = model.simulate(t_grid, z0_init=0.0038, dt=1e-3)
    assert z.shape == (15,)
    assert torch.all(torch.isfinite(z))
    assert abs(z[0].item() - 0.0038) < 1e-6
