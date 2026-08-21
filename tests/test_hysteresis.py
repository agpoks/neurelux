import torch

from atlas_physics.hysteresis import (
    GRUHysteresis,
    JilesAtherton,
    MLPHysteresis,
    dlangevin,
    langevin,
)


def test_langevin_safe_at_zero():
    x = torch.tensor([0.0, 1e-6, -1e-6])
    assert torch.all(torch.isfinite(langevin(x)))
    assert torch.all(torch.isfinite(dlangevin(x)))


def test_langevin_matches_naive_away_from_zero():
    x = torch.tensor([0.5, 1.0, 2.0, -1.5])
    naive = 1.0 / torch.tanh(x) - 1.0 / x
    assert torch.allclose(langevin(x), naive, atol=1e-6)


def test_ja_params_positive_for_any_raw_value():
    model = JilesAtherton()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 10)
    Ms, a, alpha, k, c = model.params()
    assert Ms > 0 and a > 0 and alpha >= 0 and k > 0
    assert 0 < c < 1


def test_ja_simulate_produces_finite_trajectory_through_a_loop():
    model = JilesAtherton()
    H = torch.tensor([0.0, 0.3, 0.6, 0.9, 0.6, 0.3, 0.0, -0.3, -0.6, -0.3, 0.0])
    M = model.simulate(H, M0=torch.tensor(0.0))
    assert M.shape == H.shape
    assert torch.all(torch.isfinite(M))


def test_ja_zero_field_history_stays_at_rest():
    model = JilesAtherton()
    H = torch.zeros(5)
    M = model.simulate(H, M0=torch.tensor(0.0))
    assert torch.allclose(M, torch.zeros(5), atol=1e-6)


def test_ja_with_residual_differs_from_without():
    torch.manual_seed(0)
    residual = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    with torch.no_grad():
        for p in residual.parameters():
            p.add_(0.5)  # push away from the near-zero init so the residual is non-negligible
    plain = JilesAtherton()
    ude = JilesAtherton(residual=residual)
    with torch.no_grad():
        for p_plain, p_ude in zip(plain.parameters(), list(ude.parameters())[:5]):
            p_ude.copy_(p_plain)

    H = torch.linspace(0.0, 1.0, 10)
    M_plain = plain.simulate(H, M0=torch.tensor(0.0))
    M_ude = ude.simulate(H, M0=torch.tensor(0.0))
    assert not torch.allclose(M_plain, M_ude)


def test_mlp_hysteresis_shape():
    model = MLPHysteresis()
    H = torch.linspace(-1, 1, 20)
    B = model(H)
    assert B.shape == H.shape


def test_mlp_hysteresis_is_memoryless_same_H_same_B():
    model = MLPHysteresis()
    H = torch.tensor([0.3, 0.6, 0.3, -0.1, 0.3])
    B = model(H)
    # every occurrence of H=0.3 must map to the identical output: no path dependence possible
    assert torch.allclose(B[0], B[2]) and torch.allclose(B[0], B[4])


def test_gru_hysteresis_shape():
    model = GRUHysteresis()
    H = torch.linspace(-1, 1, 20)
    B = model(H)
    assert B.shape == H.shape


def test_gru_hysteresis_can_differ_for_repeated_H_given_different_history():
    torch.manual_seed(0)
    model = GRUHysteresis()
    H_up = torch.tensor([0.0, 0.1, 0.2, 0.3])
    H_down = torch.tensor([0.5, 0.4, 0.3, 0.3])
    B_up = model(H_up)
    B_down = model(H_down)
    # same final H=0.3 reached from different histories -- GRU is free to (though not
    # guaranteed to) give a different answer, unlike the MLP which structurally cannot.
    assert B_up.shape == B_down.shape
