#!/usr/bin/env python3
"""Extract the real B-H major-loop data used by notebooks/02_hysteresis_material_model.ipynb
from the UPB `materialdatabase` package: https://github.com/upb-lea/materialdatabase

Only a tiny, specific subset is pulled -- the manufacturer-datasheet major
loop(s) for one ferrite material (N87, Epcos) -- and saved locally so the
notebook runs fully offline afterward. See scripts/README.md and PLAN.md §15.

Usage:
    pip install materialdatabase   # if not already installed
    python scripts/download_materialdatabase.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TARGET_DIR = Path("data/public/materialdatabase")
TARGET_FILE = TARGET_DIR / "bh_loops.json"
MATERIAL = "N87"


def main() -> int:
    if TARGET_FILE.exists():
        print(f"{TARGET_FILE} already exists -- nothing to do.")
        return 0

    try:
        import materialdatabase  # noqa: F401
    except ImportError:
        print("materialdatabase is not installed.")
        print("Run: pip install materialdatabase --no-build-isolation")
        print(f"Then re-run this script. Expected output: {TARGET_FILE}")
        return 1

    import materialdatabase as mdb

    db_path = Path(mdb.__file__).parent / "data" / "material_data_base.json"
    with open(db_path) as f:
        db = json.load(f)

    if MATERIAL not in db:
        print(f"Material {MATERIAL!r} not found in the installed materialdatabase package.")
        return 1

    entry = db[MATERIAL]
    loops = entry["manufacturer_datasheet"]["b_h_curve"]

    out = {
        "material": MATERIAL,
        "manufacturer": entry.get("Manufacturer"),
        "source": "pip package 'materialdatabase' (LEA, Paderborn University), material_data_base.json",
        "source_url": "https://github.com/upb-lea/materialdatabase",
        "license": "GPL-3.0 (as declared by the materialdatabase package)",
        "note": (
            "Digitized from the manufacturer datasheet, not an original measurement by this "
            "project. Ferrite, not steel -- ATLAS uses steel; used here as a real B-H loop for "
            "method validation, per PLAN.md's documented steel-preferred/ferrite-fallback policy."
        ),
        "loops": loops,
    }

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    with open(TARGET_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {TARGET_FILE} ({len(loops)} loop(s) for {MATERIAL}: "
          f"temperatures {[l['temperature'] for l in loops]}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
