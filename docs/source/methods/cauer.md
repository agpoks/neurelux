(cauer)=
# Cauer ladder — skin effect

The Cauer ladder is the first method built in this project, because it's the most direct match to something ATLAS already does: represent the rail's skin effect with a layered network instead of solving the full field PDE — the same "discretize into layers with frequency-dependent impedance" idea used classically for winding eddy currents ({cite:t}`dowell1966effects`), here applied to a solid conducting rail instead. `src/atlas_physics/cauer.py` implements it; this page walks through the physics, the code, and what it actually learns.

## The equation

1D magnetic diffusion through depth $z$ (quasi-static, no displacement current):

$$
\mu \frac{\partial H}{\partial t} = \frac{\partial}{\partial z}\!\left(\rho \frac{\partial H}{\partial z}\right), \qquad \rho = \frac{1}{\sigma}
$$

Discretize depth into $N$ layers, finite-volume style. Layer $i$ gets a capacitance $C_i = \mu_i\, dz_i$ (how much field it stores per unit potential) and each interface a conductance $G_i \approx 1/(\rho_i\, dz_i)$ (how easily field diffuses across it), with layer 0 additionally coupled to a driven surface boundary through $G_0$. Writing Kirchhoff's current law at every node gives exactly a series-conductance / shunt-capacitance ladder:

$$
C\, \dot{x} = -D^\top G D\, x + B_u\, u
$$

$x \in \mathbb{R}^N$ is the per-layer field state (layer 0 = surface), $D$ is the **fixed** incidence matrix of the ladder topology, $C=\mathrm{diag}(C_i)$, $G=\mathrm{diag}(G_i)$, and $u(t)$ is the surface excitation.

**PHYSICS-ENCODED:** the ladder topology (which layer connects to which) is a constant buffer, never touched by the optimizer. Only $C_i = \mathrm{softplus}(\cdot) > 0$ and $G_i = \mathrm{softplus}(\cdot) > 0$ are trainable — a positivity constraint that keeps the ladder passive/dissipative for *any* value training finds.

## The ladder, drawn

```{eval-rst}
.. plot:: _diagrams/cauer_ladder.py
```

Each node's field state $x_i$ is coupled to its neighbors by a learned conductance and drains to "ground" through a learned capacitance — nothing else. No node is directly connected to any other node except its immediate depth-neighbors: that fixed, sparse structure *is* the discretized diffusion equation, and it's what makes this different from letting a generic recurrent network learn the same input/output behavior.

## How it's built

`CauerLadder1D` in [`src/atlas_physics/cauer.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/cauer.py) builds the topology once, as a fixed buffer, and derives the state-space matrices from it every forward pass rather than hand-simplifying them — so the derivation above is directly checkable against the code:

```python
def system_matrices(self):
    D = self.D                      # fixed topology buffer, never trained
    G = torch.diag(self.G())        # learned, softplus(raw_G) > 0
    C = self.C()                    # learned, softplus(raw_C) > 0
    A = -(D.T @ G @ D)              # exactly -D^T G D from the KCL derivation
    B_u = -(D.T @ G @ self.d0)      # exactly -D^T G d0
    return A, B_u, C

def rhs(self, x, u):
    A, B_u, C = self.system_matrices()
    return (x @ A.T + u * B_u) / C  # dx/dt = C^-1 (A x + B_u u)
```

`simulate()` rolls this out with explicit (forward) Euler — deliberately the simplest possible fixed-step integrator, because `step()` (below) needs to be a pure function of `(input, state, dt)` with no adaptive-step solver hidden inside it, for the eventual Simulink deployment contract (`interfaces/simulink/README.md` in the repository):

```python
def step(self, u_t, x, dt):
    dx = self.rhs(x, u_t)
    x_next = x + dt * dx
    outputs = x_next[..., :1]   # surface-node observable, by convention
    return outputs, x_next
```

`tests/test_cauer.py` checks this construction actually has the properties claimed above: the topology buffer is never a `nn.Parameter` (so gradient descent structurally cannot touch it), $C_i, G_i$ stay strictly positive for any raw value, and the state matrix $A$ is negative semi-definite for any trained $C, G$ — i.e. the ladder cannot become active/unstable no matter what training finds.

## Results

Trained on **synthetic** data only (no real ATLAS measurements exist yet — see {doc}`Getting started <../getting-started>` for exactly what data this project does and doesn't have): a ground-truth 6-layer ladder with geometrically-growing layer thickness is simulated under four excitations (step, ramp, sinusoid, multi-frequency), Gaussian noise is added to two observables (surface field, total flux), and a student ladder — initialized flat and uninformative — is trained on the step and ramp excitations only.

```{image} ../_static/results/depth_time_and_surface_deep.png
:alt: Depth-time field penetration and surface vs. deep field, true system
:width: 100%
```

The true system's field penetration makes the skin effect directly visible: high-frequency content stays confined to the shallow layers, low-frequency content reaches deeper, with a visible amplitude drop and phase lag at depth.

```{image} ../_static/results/bode_frequency_response.png
:alt: True vs. learned frequency response
:width: 85%
```

This is the sharper test. Trained on *only* the step and ramp, the learned ladder's frequency response matches the true system closely in the frequency band that training actually excited (~2.5% error below 0.3 Hz) and diverges markedly outside it (~37% error above 1 Hz). That's not a training failure — it's a genuine **structural identifiability** result: a surface-only measurement under slow excitation constrains the system's low-frequency input-output behavior well, but individual deep-layer parameters (which mostly affect fast/high-frequency response) stay only weakly determined. The practical consequence for any real deployment: the excitation used to calibrate a model like this must span the frequency/velocity range it's meant to be accurate over, not just the slow transients that happened to be easy to collect.

Time-domain generalization to excitation types never seen during training (sinusoid, multi-frequency) is nonetheless good — RMSE of 0.038 and 0.016 respectively, versus a noise floor around 0.01 — because the *topology* being correct is enough to generalize across excitation shape even when individual parameters aren't fully pinned down. See the full discussion, including the actual learned-vs-true parameter comparison, in the notebook below.

## Try it

Full walkthrough with all plots and the training loop: {doc}`../notebooks/01_skin_effect_cauer_synthetic`.

Source: [`src/atlas_physics/cauer.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/cauer.py) · [`tests/test_cauer.py`](https://github.com/agpoks/neurelux/blob/main/tests/test_cauer.py).

## Next

This 1D ladder only resolves depth. The rail also needs lateral (surface-direction) resolution to capture eddy-current redistribution and velocity dependence — that's {doc}`the surface × depth Graph-Cauer network <graph_cauer>`, this project's central candidate for the rail.
