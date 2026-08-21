(graph-cauer)=
# Surface × depth Graph-Cauer network

Extends {doc}`the 1D Cauer ladder <cauer>` into a 2D graph over both surface position and depth — `PLAN.md`'s central candidate method for ATLAS — and tests whether the added lateral coupling earns its cost against both a topology-poorer baseline and a topology-free but much larger one.

## The equation

Same conservation form as the 1D ladder, now over a grid graph with vertical (depth, within a column) and horizontal (surface, across columns at fixed depth) edges:

$$
C\, \dot{x} = -D^\top G D\, x + B_u\, u(t)
$$

Every edge stays symmetric and positive (softplus-constrained), so the same negative-semidefiniteness passivity proof from the 1D case covers the full 2D graph unchanged. Velocity enters through the **excitation** — a Gaussian pulse sweeping across surface positions at a rate set by $v$ — not through directional conductances, which would need their own passivity argument.

## How it's built

`src/atlas_physics/graph_cauer.py::build_grid_graph` constructs the fixed topology: $n_{\text{surface}}$ boundary edges (one port per column), $n_{\text{surface}}(n_{\text{depth}}-1)$ vertical internal edges, $n_{\text{depth}}(n_{\text{surface}}-1)$ horizontal edges. `GraphCauer` generalizes `CauerLadder1D` to a multi-port drive ($B_u$ becomes an $(N, n_{\text{surface}})$ matrix), otherwise identical: `D` is a fixed buffer, only per-node $C_i$ and per-edge $G_e$ are trainable.

## Results

A moving Gaussian excitation sweeps across 6 surface positions × 4 depth layers at two low velocities (training) and one held-out higher velocity (testing) — three models compared:

| model | parameters | test-velocity MSE |
|---|---|---|
| independent ladders (lateral $G$ forced to ~0) | 68 | 0.000052 |
| **Graph-Cauer** (full lateral coupling) | 68 | **0.000001** |
| free-linear (unconstrained $N\times N$ state matrix) | 720 | 0.000285 |

```{image} ../_static/results/nb04_1.png
:alt: Extrapolation to unseen velocity, probe-node comparison
:width: 75%
```

## Discussion

The cleanest result in this project so far in favor of physics-encoding, worth being precise about why. Independent ladders miss the lateral dimension entirely — each column can only respond to its own local excitation, never the field spreading in from a neighbor as the pulse sweeps past, exactly what changes at an unseen sweep velocity. Graph-Cauer wins with the *same 68 parameters* as that baseline — the only difference is lateral conductances being nonzero and learnable. The free-linear model, despite roughly 10x the parameters, extrapolates worse than Graph-Cauer and only modestly better than the topology-starved baseline: with no structural constraint, it has to learn *both* the correct sparse connectivity pattern and the right parameter values from two training velocities, strictly harder than learning parameter values alone given a topology that's already correct by construction. More raw capacity did not buy better extrapolation here — the right structural constraint did.

## Try it

Full walkthrough: {doc}`../notebooks/04_graph_cauer_surface_depth`. Source: [`src/atlas_physics/graph_cauer.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/graph_cauer.py).
