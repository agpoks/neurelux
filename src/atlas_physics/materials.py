"""Loader for the real B-H major-loop data used by Notebook 02.

Data itself is not committed as a large blob -- `scripts/download_materialdatabase.py`
extracts a small subset (one ferrite material's manufacturer-datasheet major
loops) from the UPB `materialdatabase` package into
`data/public/materialdatabase/bh_loops.json`, and this module just reads it.

Ferrite (N87), not steel: ATLAS uses steel, but no open steel B-H dataset has
been identified yet -- this is the documented fallback, used for method
validation, not as a stand-in for real ATLAS material behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DEFAULT_PATH = Path("data/public/materialdatabase/bh_loops.json")


def load_bh_loop(temperature: int, path: Path | str = _DEFAULT_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, B) arrays (A/m, T) for the real N87 major loop at `temperature` (25 or 100 °C).

    Raises FileNotFoundError with a pointer to the download script if the
    data hasn't been fetched yet -- notebooks should catch this and print
    the same instruction rather than crash uninformatively.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_materialdatabase.py"
        )
    with open(path) as f:
        data = json.load(f)
    loop = next((l for l in data["loops"] if l["temperature"] == temperature), None)
    if loop is None:
        available = [l["temperature"] for l in data["loops"]]
        raise ValueError(f"No loop at temperature={temperature} in {path}; available: {available}")
    H = np.array(loop["magnetic_field_strength"], dtype=np.float64)
    B = np.array(loop["flux_density"], dtype=np.float64)
    return H, B


def bh_loop_exists(path: Path | str = _DEFAULT_PATH) -> bool:
    return Path(path).exists()
