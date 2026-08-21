#!/usr/bin/env python3
"""Fetch the official TEAM Workshop Problem 7 specification PDF and write out
the real input parameters used by notebooks/05_eddy_current_team7.ipynb.

Source (official, freely published): "Problem 7 -- Asymmetrical Conductor
with a Hole," https://www.compumag.org/wp/wp-content/uploads/2018/06/problem7.pdf

Note: that specification PDF gives the real geometry, material, and
excitation parameters (transcribed into `PARAMETERS` below) but is a
scanned document with no embedded reference/measurement table. The
benchmark's actual reference field values (at probe points A1-A4, B1-B4)
are published in a companion results paper (COMPEL, 1990, Emerald
Publishing) not freely accessible from this environment -- notebook 05
does not use or invent any values from that paper; see REFERENCES.md.

Usage:
    python scripts/download_team7.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TARGET_DIR = Path("data/public/team7")
PDF_PATH = TARGET_DIR / "problem7.pdf"
PARAMS_PATH = TARGET_DIR / "parameters.json"
SOURCE_URL = "https://www.compumag.org/wp/wp-content/uploads/2018/06/problem7.pdf"

# Transcribed directly from Table 1 and Fig. 1 of the official problem7.pdf.
PARAMETERS = {
    "plate_conductivity_S_per_m": 3.526e7,
    "plate_material": "aluminum",
    "plate_relative_permeability": 1.0,
    "plate_dimensions_mm": {"x": 294, "y": 294, "thickness": 19},
    "hole_dimensions_mm": {"x": 108, "y": 108, "offset_x": 18, "offset_y": 18},
    "coil_ampere_turns": 2742,
    "coil_inner_opening_mm": 150,
    "coil_height_mm": 100,
    "coil_gap_above_plate_mm": 30,
    "frequencies_Hz": [50, 200],
}


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        try:
            import urllib.request

            req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(PDF_PATH, "wb") as f:
                f.write(resp.read())
            print(f"Downloaded {PDF_PATH}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not download {SOURCE_URL}: {exc}")
            print(f"Manual fallback: download it yourself and place it at {PDF_PATH}")
            print("(not required for the notebook -- parameters.json below is written regardless)")

    if not PARAMS_PATH.exists():
        with open(PARAMS_PATH, "w") as f:
            json.dump(PARAMETERS, f, indent=2)
        print(f"Wrote {PARAMS_PATH}")
    else:
        print(f"{PARAMS_PATH} already exists -- nothing to do.")

    print(
        "\nNote: the benchmark's reference field measurements (probe points A1-A4, B1-B4) "
        "are in a companion results paper not freely accessible here -- notebooks/05 does not "
        "use them; see REFERENCES.md for what this notebook does and doesn't reproduce."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
