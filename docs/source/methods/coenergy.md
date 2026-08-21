(coenergy)=
# Co-energy network

Tests whether learning one shared co-energy potential for flux and force — instead of two independent models — earns its keep, on both accuracy and a directly measurable physical-consistency violation.

## The equation

$$
\lambda = \frac{\partial W'}{\partial I}, \qquad F_A = -\frac{\partial W'}{\partial g}
$$

Being derivatives of the same smooth $W'$ implies a Maxwell-type reciprocity relation that holds for *any* valid co-energy function, learned or not:

$$
\frac{\partial \lambda}{\partial g} = -\frac{\partial F_A}{\partial I}
$$

This is the notebook's main diagnostic: computing both sides for a trained model and comparing them directly tests physical consistency, rather than assuming it.

## How it's built

`CoenergyNet` learns a single scalar $W'_\theta(I,g,T)$; flux and force are its exact partial derivatives via autograd, computed the same way at training and inference time:

```python
def flux_and_force_from_potential(w_fn, I, g, T):
    I = I.requires_grad_(True)
    g = g.requires_grad_(True)
    W = w_fn(I, g, T)
    dWdI, dWdg = torch.autograd.grad(W.sum(), [I, g], create_graph=True)
    return dWdI, -dWdg
```

This means `CoenergyNet` can **never** be evaluated inside `torch.no_grad()` — flux and force *are* derivatives, so the computation graph has to exist even at "inference." `IndependentNets` is the baseline: two separate networks for $\lambda$ and $F_A$, structurally free to disagree.

**PHYSICS-ENCODED:** Maxwell reciprocity isn't a loss term anywhere in `CoenergyNet`'s training — it's a mathematical consequence of both outputs being derivatives of one function, so it holds for *any* trained weights, not just ones a penalty happened to discourage from violating it. `IndependentNets` is the physics-guided-only counterpoint: same inputs, same physical quantities as targets, no structural link between them.

## Results

Both trained on the same joint flux+force loss; evaluated on the training range and an extrapolated current range never seen during training, plus the reciprocity violation itself:

| model | flux (train) | force (train) | flux (extrap.) | force (extrap.) |
|---|---|---|---|---|
| CoenergyNet | 0.00002 | 0.00000 | 0.00671 | **0.00715** |
| IndependentNets | 0.00001 | 0.03444 | **0.00154** | 0.46159 |

| model | mean \|reciprocity violation\| | max |
|---|---|---|
| CoenergyNet | 0.00005 | 0.00049 |
| IndependentNets | **421.0** | **1471.7** |

```{image} ../_static/results/nb07_0.png
:alt: Flux and force vs current, true vs both models, training range shaded
:width: 100%
```

## Discussion

The reciprocity violation is the sharpest result: `CoenergyNet`'s is at floating-point noise level everywhere, by construction — nothing in training targeted it directly, it falls out of predicting one potential instead of two functions. `IndependentNets`' violation is **orders of magnitude larger** — two networks minimizing their own error individually have no reason to agree on this cross-derivative relationship, and empirically they don't. `IndependentNets` also extrapolates far worse on force than on flux specifically, consistent with force being the harder, more nonlinear quantity here — exactly the situation where borrowing information through a shared potential should help most, and does. Not a uniform win, though: `IndependentNets`' flux extrapolation is slightly *better* than `CoenergyNet`'s — a reminder that the co-energy structure's benefit is concentrated where it's actually earned (the harder, more consistency-dependent quantity), not automatic everywhere.

## Try it

Full walkthrough: {doc}`../notebooks/07_energy_consistent_force`. Source: [`src/atlas_physics/coenergy.py`](https://github.com/agpoks/neurelux/blob/main/src/atlas_physics/coenergy.py).
