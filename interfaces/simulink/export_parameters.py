#!/usr/bin/env python3
"""export_parameters.py — see interfaces/simulink/README.md for the deployment contract.

Not yet implemented (stub). Will be filled in once the first src/atlas_physics
submodel (cauer.CauerLadder1D) is trained and frozen, so there is a concrete
model to generate reference trajectories / export parameters for.
"""
import sys


def main() -> int:
    print(f"{__file__} is not yet implemented — see interfaces/simulink/README.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
