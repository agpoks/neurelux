import torch

from atlas_physics.coenergy import CoenergyNet, IndependentNets, flux_and_force_from_potential, true_coenergy


def test_true_coenergy_finite_and_positive():
    I = torch.linspace(0.1, 5.0, 10)
    g = torch.full_like(I, 2e-3)
    T = torch.full_like(I, 25.0)
    W = true_coenergy(I, g, T)
    assert torch.all(torch.isfinite(W))
    assert torch.all(W >= 0)


def test_true_flux_saturates_with_current():
    I = torch.linspace(0.01, 20.0, 50)
    g = torch.full_like(I, 2e-3)
    T = torch.full_like(I, 25.0)
    flux, force = flux_and_force_from_potential(true_coenergy, I, g, T)
    # flux linkage should be increasing but concave (saturating), not linear, at high I
    d_flux = torch.diff(flux)
    assert torch.all(d_flux > 0)
    assert d_flux[-1] < d_flux[1]  # slope decreases -- saturation


def test_true_force_positive_and_decreasing_with_gap():
    I = torch.full((10,), 2.0)
    g = torch.linspace(1e-3, 5e-3, 10)
    T = torch.full_like(I, 25.0)
    flux, force = flux_and_force_from_potential(true_coenergy, I, g, T)
    assert torch.all(force > 0)  # attractive force magnitude, always positive here
    assert torch.all(torch.diff(force) < 0)  # weaker at larger gap


def test_coenergy_net_is_self_consistent_by_construction():
    torch.manual_seed(0)
    model = CoenergyNet()
    I = torch.rand(20) * 3 + 0.1
    g = torch.rand(20) * 3e-3 + 1e-3
    T = torch.rand(20) * 50 + 25
    flux, force = model.flux_and_force(I, g, T)

    # Maxwell reciprocity: d(flux)/dg == -d(force)/dI, since both come from the same W'
    g2 = g.clone().requires_grad_(True)
    I2 = I.clone().requires_grad_(True)
    flux2, force2 = model.flux_and_force(I2, g2, T)
    dflux_dg = torch.autograd.grad(flux2.sum(), g2, retain_graph=True)[0]
    dforce_dI = torch.autograd.grad(force2.sum(), I2, retain_graph=True)[0]
    assert torch.allclose(dflux_dg, -dforce_dI, atol=1e-4)


def test_independent_nets_shapes():
    torch.manual_seed(0)
    model = IndependentNets()
    I = torch.rand(15) * 3 + 0.1
    g = torch.rand(15) * 3e-3 + 1e-3
    T = torch.rand(15) * 50 + 25
    flux, force = model.flux_and_force(I, g, T)
    assert flux.shape == (15,) and force.shape == (15,)
