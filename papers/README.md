# papers/

Literature backing the methods and benchmarks used in this project — distinct from `REFERENCES.md` (which tracks *code repositories* consulted/reused) and from `PLAN.md` §6 (which tracks *dataset* sources). This directory tracks the *papers*: the governing equations, benchmark definitions, and method descriptions the notebooks implement.

- `references.bib` — BibTeX entries, kept in sync with the table below. Only entries whose bibliographic details (authors, venue, year) are independently confirmed are added — this project does not fabricate citations any more than it invents ATLAS measurement values (see `PLAN.md` §0/§6 for the same principle applied to data). Rendered into the docs as a bibliography page (`docs/source/bibliography.md`) via `sphinxcontrib-bibtex`.
- `pdfs/` — local cache of downloaded papers. **Not committed** (see `.gitignore`) — academic PDFs are typically copyrighted and this repository does not redistribute them. Keep your own local copies here for convenience; only `references.bib` (citation metadata) and this table are version-controlled.

## Status

| Topic | Reference | Notebook | Status |
|---|---|---|---|
| Ferromagnetic hysteresis (Jiles–Atherton model) | Jiles, D.C. and Atherton, D.L., "Theory of ferromagnetic hysteresis," *Journal of Magnetism and Magnetic Materials*, vol. 61, pp. 48–60, 1986 | 02 | in `references.bib` |
| TEAM Workshop Problem 7 (eddy-current benchmark) | official TEAM Workshop problem specification | 05 | **not yet added** — exact citation/specification document needs to be located and verified, see `PLAN.md` §6/§11 open items |
| TEAM Workshop Problem 28 (moving-conductor benchmark) | official TEAM Workshop problem specification | 06 | **not yet added** — same as above |
| HystRNN (physics-aware recurrent hysteresis) | see `REFERENCES.md` for the code repository; paper citation to be added once verified | 02 | **not yet added** |
| Magnetic hysteresis neural operator | see `REFERENCES.md` for the code repository; paper citation to be added once verified | 02 | **not yet added** |
| Princeton MagNet dataset | dataset paper to be added once verified | 02 | **not yet added** |
| Port-Hamiltonian systems (`pyphs`) | paper citation to be added once verified | future (method L) | **not yet added** |
| Magnetic track brake / eddy-current rail brake modeling | none identified yet — literature search needed once real ATLAS documentation is available | 01, 06, 09 | **not yet added** |
| Liquid Time-constant Networks (continuous-time, ODE-based RNN) | Hasani, Lechner, Amini, Rus, Grosu, "Liquid Time-constant Networks," AAAI 2021 (arXiv:2006.04439) | candidate for 02 (not yet adopted) | in `references.bib` — see `REFERENCES.md` for accessible write-ups |

Add a paper: (1) drop the PDF in `pdfs/` (local only), (2) add a verified BibTeX entry to `references.bib`, (3) update the table above and, if the paper's method gets its own notebook, cite it in that notebook's markdown.
