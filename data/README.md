# data/

- `raw/` — untouched downloads (public datasets, and real ATLAS material if/when provided under `raw/atlas/`). Not committed (see `.gitignore`) except a `.gitkeep`.
- `processed/` — small, notebook-generated artifacts (arrays, fitted parameters). Only commit files that are genuinely small; anything multi-MB stays local/regeneratable from the notebook that made it.
- `public/` — extracted/prepared public datasets, produced by `scripts/download_*.py` + `scripts/prepare_public_data.py`. Not committed.

No dataset is committed to this repository. Every notebook checks for its required local data first and prints the exact command to fetch it if missing (see `scripts/README.md` and `PLAN.md` §15). After the first successful download, notebooks run fully offline.

See `PLAN.md` §6 for the full dataset table (URL, variables, license, auto-download status, which notebook uses it, ATLAS relevance) and `REFERENCES.md` for how each external repository is used.

If real ATLAS documentation, measurements, or FEM exports become available, place them under `raw/atlas/` — this will let `PLAN.md` §1–2 be rewritten from the actual source instead of the current assumed placeholder, and lets Notebook 09 load real data instead of synthetic ATLAS-like data.
