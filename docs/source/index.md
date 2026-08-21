# NeuRelux

Physics-guided, physics-informed, and physics-encoded neural networks for the ATLAS magnetic track brake — a research project trying to replace parts of a simplified equivalent-magnetic-circuit model with learned components, without throwing away the physical structure (topology, conservation, positivity) that makes the original model trustworthy and fast.

## How to read this documentation

| | |
|---|---|
| **1. The physics** | {doc}`Background <background>` — what a magnetic track brake is, the governing equations, and precisely what "guided", "informed", and "encoded" mean here. |
| **2. The method landscape** | {doc}`Method landscape <methods/overview>` — every candidate representation surveyed, and why the recommended combination was chosen before anything was built. |
| **3. The methods, one at a time** | Ten pages, one per implemented method, equation → code → real results: {doc}`Cauer ladder <methods/cauer>`, {doc}`hysteresis <methods/hysteresis>`, {doc}`neural reluctance circuit <methods/neural_circuit>`, {doc}`Graph-Cauer <methods/graph_cauer>`, {doc}`TEAM7 <methods/team7>`, {doc}`TEAM28 <methods/team28>`, {doc}`co-energy force <methods/coenergy>`, {doc}`friction & thermal <methods/friction_thermal>`, {doc}`velocity-dependent flux weakening <methods/velocity_eddy_weakening>`, {doc}`combined model <methods/combined>`. |
| **4. The evidence** | {doc}`Results <results>` — what each method actually showed, in one place, including the ones that didn't go as expected. |

New to the project? Start with {doc}`Getting started <getting-started>`, then {doc}`Background <background>`.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Start here

getting-started
background
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Methods

methods/overview
methods/cauer
methods/hysteresis
methods/neural_circuit
methods/graph_cauer
methods/team7
methods/team28
methods/coenergy
methods/friction_thermal
methods/velocity_eddy_weakening
methods/combined
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Evidence

results
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Notebooks

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
notebooks/10_velocity_dependent_eddy_weakening
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

papers
bibliography
```
