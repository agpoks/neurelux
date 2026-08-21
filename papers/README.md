# papers/

Literature backing the methods and benchmarks used in this project — distinct from `REFERENCES.md` (which tracks *code repositories* consulted/reused) and from the dataset-provenance notes there (which track *dataset* sources). This directory tracks the *papers*: the governing equations, benchmark definitions, and method descriptions the notebooks implement.

- `references.bib` — BibTeX entries, kept in sync with the table below. Only entries whose bibliographic details (authors, venue, year) are independently confirmed are added — this project does not fabricate citations any more than it invents ATLAS measurement values. Rendered into the docs as a bibliography page (`docs/source/bibliography.md`) via `sphinxcontrib-bibtex`.
- `pdfs/` — local cache of downloaded papers. **Not committed** (see `.gitignore`) — academic PDFs are typically copyrighted and this repository does not redistribute them. Keep your own local copies here for convenience; only `references.bib` (citation metadata) and this table are version-controlled.

## Status

| Topic | Reference | Notebook | Status |
|---|---|---|---|
| Ferromagnetic hysteresis (Jiles–Atherton model) | Jiles, D.C. and Atherton, D.L., "Theory of ferromagnetic hysteresis," *Journal of Magnetism and Magnetic Materials*, vol. 61, pp. 48–60, 1986 | 02 | in `references.bib` |
| TEAM Workshop Problem 7 official results (eddy-current benchmark) | Fujiwara, K. and Nakata, T., "Results for Benchmark Problem 7 (Asymmetrical Conductor with a Hole)," *COMPEL*, 9(3), pp. 137–154, 1990 | 05 | in `references.bib` |
| TEAM Workshop Problem 28 specification (moving-conductor benchmark) | Karl, H., Fetzer, J., Kurz, S., Lehner, G. and Rucker, W.M., "Description of TEAM Workshop Problem 28: An Electrodynamic Levitation Device," Proc. TEAM Workshop, Sixth Round, Graz, 1997 | 06 | in `references.bib` |
| Physics-informed neural networks (PINN, canonical formulation) | Raissi, M., Perdikaris, P. and Karniadakis, G.E., "Physics-Informed Neural Networks," *Journal of Computational Physics*, 378, pp. 686–707, 2019 | 05, 03 | in `references.bib` |
| Skin/proximity effect via layered circuits (windings) | Dowell, P.L., "Effects of Eddy Currents in Transformer Windings," *Proc. IEE*, 113(8), pp. 1387–1394, 1966 | 01 | in `references.bib` |
| Magnetic-equivalent-circuit modeling of electromagnets | Kallenbach, E. et al., *Elektromagnete: Grundlagen, Berechnung, Entwurf und Anwendung*, 5th ed., Springer Vieweg, 2018 | 03, 10 | in `references.bib` |
| Coulomb/Stribeck friction models, survey | Armstrong-Hélouvry, B., Dupont, P. and Canudas de Wit, C., "A Survey of Models, Analysis Tools and Compensation Methods for the Control of Machines with Friction," *Automatica*, 30(7), pp. 1083–1138, 1994 | 08 | in `references.bib` |
| Hamiltonian Neural Networks (potential-derivative pattern) | Greydanus, S., Dzamba, M. and Yosinski, J., "Hamiltonian Neural Networks," NeurIPS 32, pp. 15353–15363, 2019 (arXiv:1906.01563) | 07 | in `references.bib` |
| Magnetic track brake velocity-dependent attraction force (motion-induced eddy currents) | Ebner, B., Plöchl, M. and Edelmann, J., "Stability Behaviour of a Basic Magnetic Track Brake Model: Influences of System Parameters and Motion-Induced Eddy Currents," *Nonlinear Dynamics*, 2025 — real, current TU Wien measurement of exactly this effect | 10 | in `references.bib` |
| Motional eddy currents via magnetic-equivalent-circuit method (rotating) | Wang, J. and Zhu, J., "A Simple Method for Performance Prediction of Permanent Magnet Eddy Current Couplings Using a New Magnetic Equivalent Circuit Model," *IEEE Trans. Industrial Electronics*, 65(3), pp. 2487–2495, 2018 | 10 | in `references.bib` |
| Motional eddy currents via magnetic-equivalent-circuit method (linear) | Gholizad, H., Funieru, B. and Binder, A., "Direct Modeling of Motional Eddy Currents in Highly Saturated Solid Conductors by the Magnetic Equivalent Circuit Method," *IEEE Trans. Magnetics*, 45(3), pp. 1016–1019, 2009 | 10 | in `references.bib` |
| Universal Differential Equations | Rackauckas, C. et al., "Universal Differential Equations for Scientific Machine Learning," arXiv:2001.04385, 2020 | 02 | in `references.bib` |
| Liquid Time-constant Networks (continuous-time, ODE-based RNN) | Hasani, Lechner, Amini, Rus, Grosu, "Liquid Time-constant Networks," AAAI 2021 (arXiv:2006.04439) | candidate for 02 (not yet adopted) | in `references.bib` — see `REFERENCES.md` for accessible write-ups |
| HystRNN (physics-aware recurrent hysteresis) | see `REFERENCES.md` for the code repository; paper citation to be added once verified | 02 | **not yet added** |
| Magnetic hysteresis neural operator | see `REFERENCES.md` for the code repository; paper citation to be added once verified | 02 | **not yet added** |
| Princeton MagNet dataset | dataset paper to be added once verified | 02 | **not yet added** |
| Port-Hamiltonian systems (`pyphs`) | paper citation to be added once verified | future (method L) | **not yet added** |

Add a paper: (1) drop the PDF in `pdfs/` (local only), (2) add a verified BibTeX entry to `references.bib`, (3) update the table above and, if the paper's method gets its own notebook, cite it in that notebook's markdown.
