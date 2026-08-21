# NeuRelux

Physics-guided, physics-informed, and physics-encoded neural networks for the ATLAS magnetic track brake — a research project trying to replace parts of a simplified equivalent-magnetic-circuit model with learned components, without throwing away the physical structure (topology, conservation, positivity) that makes the original model trustworthy and fast.

Start with [**Background**](background.md) if you're new to the problem — it covers what a magnetic track brake is, the governing equations, and the guided/informed/encoded distinction the rest of this documentation relies on. Then [**Method landscape**](methods/overview.md) surveys the candidate approaches, and [**Cauer ladder**](methods/cauer.md) walks through the first one actually built, equation to code to results.

```{toctree}
:maxdepth: 2
:caption: Tutorial

background
methods/overview
methods/cauer
results
```

```{toctree}
:maxdepth: 1
:caption: Notebooks (run it yourself)

notebooks/00_overview_methods
notebooks/01_skin_effect_cauer_synthetic
notebooks/02_hysteresis_material_model
notebooks/03_neural_reluctance_circuit
notebooks/04_graph_cauer_surface_depth
notebooks/05_eddy_current_team7
notebooks/06_moving_conductor_team28
notebooks/07_energy_consistent_force
notebooks/08_friction_temperature_model
notebooks/09_atlas_small_combined_model
```

```{toctree}
:maxdepth: 1
:caption: Project

plan
readme
references
papers
bibliography
```
