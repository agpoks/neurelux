# scripts/

Download/prepare scripts for the public datasets in `PLAN.md` §6. None of these are implemented yet beyond a documented stub — they are filled in when the notebook that needs them is reached (`PLAN.md` §20), so that "create the project structure" doesn't quietly turn into "silently attempt several-GB downloads no one asked for" (see `PLAN.md` §15).

Every script, once implemented, must:

1. Download only a **small** subset by default (never several GB without being explicitly asked).
2. If credentials are required (e.g. Kaggle) and unavailable, **not fail the whole project** — instead print the dataset URL, expected directory, expected filenames, and the preprocessing command, exactly as specified in `PLAN.md` §15.
3. Be idempotent / safe to re-run (skip if data already present).

| Script | Dataset | Used by | Status |
|---|---|---|---|
| `download_magnet.py` | Princeton MagNet | Notebook 02 | stub |
| `download_magnet_challenge2.py` | MagNet Challenge 2 | Notebook 02 | stub |
| `download_materialdatabase.py` | UPB Material Database | Notebook 02/03 | implemented — extracts real N87 B-H major loops to `data/public/materialdatabase/bh_loops.json` |
| `download_team7.py` | TEAM Workshop Problem 7 | Notebook 05 | stub (likely manual — no confirmed automatic source yet) |
| `download_team28.py` | TEAM Workshop Problem 28 | Notebook 06 | stub (likely manual — no confirmed automatic source yet) |
| `prepare_public_data.py` | (all of the above) | all | stub — shared post-download normalization/caching into `data/public/` |
