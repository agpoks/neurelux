# Results so far

Only one method is built and trained end to end right now: the [Cauer ladder](methods/cauer.md) (Notebook 01). Everything else in [`methods/overview.md`](methods/overview.md) is a candidate, not yet evidence.

## What's actually been shown

- A physics-encoded ODE (fixed ladder topology, only positivity-constrained parameters trained) **recovers the true system's low-frequency behavior almost exactly** from a step and a ramp excitation alone, and generalizes to excitation shapes it never saw during training.
- The same experiment also surfaces a real limitation, not a hidden one: individual deep-layer parameters are only weakly identifiable from surface-only, low-frequency data. See [`methods/cauer.md`](methods/cauer.md#results) for the frequency-response comparison that shows exactly where the learned model stays accurate and where it doesn't.

That second point matters more than it might look: it's a concrete, checkable answer to "how much can we trust this kind of model outside the conditions it was trained on", which is the central risk in replacing any part of ATLAS's existing circuit with a learned component.

## What's next

Everything downstream builds on this ladder: [`methods/overview.md`](methods/overview.md) lists the full method-by-method plan, and `PLAN.md` §7/§9 has the notebook build order and the ablation study (M0-M8) that will eventually compare all of these against each other and against the existing ATLAS circuit — accuracy, extrapolation, physical-residual violation, and cost, not just accuracy alone.
