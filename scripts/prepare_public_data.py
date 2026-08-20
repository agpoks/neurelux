#!/usr/bin/env python3
"""Shared post-download normalization/caching of public datasets into data/public/.

Not yet implemented (stub) — invoked after the relevant download_*.py script(s)
have populated data/raw/. See scripts/README.md and PLAN.md §15.
"""
import sys


def main() -> int:
    print("scripts/prepare_public_data.py is not yet implemented.")
    print("Run the relevant scripts/download_*.py first once implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
