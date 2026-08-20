# REFERENCES.md

External repositories consulted or planned to be consulted, and how each relates to this project. "Reused" means code/data is (or will be) pulled in via `scripts/download_*.py` or `git submodule`/`pip install`; "inspiration only" means we reimplement the relevant concept ourselves in clean PyTorch inside `src/atlas_physics/`, per the project's "do not blindly copy" instruction.

| Repository | Purpose | Useful concepts | Related notebook(s) | Reused or inspiration-only | License |
|---|---|---|---|---|---|
| [chandratue/HystRNN](https://github.com/chandratue/HystRNN) | Physics-aware recurrent hysteresis model | Recurrent state update combining a physical hysteresis operator with a learned residual | 02 | Inspiration-only (reimplement the recurrence in PyTorch; cite approach) | check repo (not yet verified) |
| [chandratue/magnetic_hysteresis_neural_operator](https://github.com/chandratue/magnetic_hysteresis_neural_operator) | Neural-operator (FNO/RIFNO-style) hysteresis model | Operator-learning formulation `H(t) -> B(t)` as function-to-function map | 02 (optional) | Inspiration-only | check repo |
| [PrincetonUniversity/magnet](https://github.com/PrincetonUniversity/magnet) | MagNet dataset + baseline models for core loss / B-H prediction | Dataset loaders, waveform representation conventions | 02 | Reused (data only, via `scripts/download_magnet.py`) | check repo |
| [minjiechen/magnetchallenge-2](https://github.com/minjiechen/magnetchallenge-2) | MagNet Challenge 2 dataset/benchmark | Extended waveform/material coverage | 02 | Reused (data only) | check repo |
| [upb-lea/materialdatabase](https://github.com/upb-lea/materialdatabase) | Structured magnetic material parameter database | Material parameter schema, permeability/loss lookups | 02, 03 | Reused (data/schema) | check repo (LEA, Paderborn — commonly MIT, verify) |
| [upb-lea/FEM_Magnetics_Toolbox](https://github.com/upb-lea/FEM_Magnetics_Toolbox) | FEM-based magnetic component simulation | Possible FEM cross-check for material/geometry sanity | reference only | Inspiration-only / optional tool dependency | check repo (verify before install) |
| [pyphs/pyphs](https://github.com/pyphs/pyphs) | Port-Hamiltonian system modeling in Python | Passive/dissipative structure (`J`, `R`, `Q` matrices) for a possible port-Hamiltonian NN variant | future — not yet scheduled to a specific notebook | Inspiration-only | check repo |
| TEAM Workshop Problem 7 (Testing Electromagnetic Analysis Methods) | Public eddy-current benchmark: conducting plate with a hole, asymmetric coil excitation | Reference geometry + FEM/measured `B_z` field for skin-effect validation | 05 | Data only, if/when locatable — manual download, see `scripts/download_team7.py` | benchmark, treat as freely published for research use — verify per-source terms |
| TEAM Workshop Problem 28 | Public moving-conductor eddy-current force benchmark | Reference force-vs-velocity curve; closest public analogue to ATLAS's velocity-dependent eddy force | 06 | Data only, manual download, see `scripts/download_team28.py` | benchmark, verify per-source terms |

## Other concepts investigated (no single canonical repo, reimplemented directly)

- **Cauer ladder networks** — standard circuit-theory construction (RC/RL ladder equivalent of a diffusion PDE); implemented directly in `src/atlas_physics/cauer.py` from the discretized diffusion equation, no external code needed.
- **Physics-informed graph neural networks** — general pattern (message passing constrained to respect a known conservation law); implemented ad hoc in `src/atlas_physics/graph_cauer.py` rather than adopting a general-purpose PIGNN library, to keep the encoded structure (incidence matrix `D`, `C`, `G`) explicit and inspectable.
- **Maxwell / vector-potential PINNs** — standard PINN construction (`∇×(ν∇×A)=J` as a collocation loss); reimplemented minimally as an optional baseline in Notebook 05, not adopted from a specific PINN framework, to keep dependencies light.

## Further reading: Liquid Neural Networks

Liquid Time-constant Networks (Hasani et al., AAAI 2021 — verified citation in `papers/references.bib`) are continuous-time, ODE-based recurrent networks: the hidden state evolves as a learned nonlinear ODE, similar in spirit to how the Cauer ladder in `src/atlas_physics/cauer.py` is itself a fixed, physically-structured ODE (`C dx/dt = -DᵀGD x + B_u u`). That similarity is why they're worth flagging here as a candidate continuous-time architecture for the memory/hysteresis notebook (02) — not yet adopted, just noted. Accessible write-ups, for orientation before the original paper:

- [Liquid Neuronal Networks using Pytorch — Andrea Rosales](https://medium.com/@andrea.rosales08/liquid-neuronal-networks-using-pytorch-0d0bef41d504) — implementation walkthrough in PyTorch, the framework this project uses throughout.
- [Liquid Neural Networks: A Paradigm Shift in Artificial Intelligence — Shinde Vinayak Rao Patil](https://medium.com/@shindevinayakraopatil/liquid-neural-networks-a-paradigm-shift-in-artificial-intelligence-3be1a750869c) — general conceptual overview.
- [Applying Liquid Neural Networks (LNN) in Self-Driving Labs (SDL) — Sissi Feng](https://medium.com/@isissifeng/applying-liquid-neural-networks-lnn-in-self-driving-labs-sdl-837447b7df5e) — real-time adaptive control application, thematically close to ATLAS's real-time embedded operating context.

These are informal/blog sources, not peer-reviewed — cited here as explainers and pointers, not as authoritative technical references.

## Verification note

License strings above are **not yet independently verified** — "check repo" markers must be resolved (open the repository's `LICENSE` file) before any code (not just data) is copied verbatim into `src/atlas_physics/`. Data-only reuse (downloading published measurement/benchmark files) is lower risk but should still respect each source's stated terms of use.
