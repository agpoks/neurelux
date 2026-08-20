#!/usr/bin/env python3
"""Download a small subset of: https://github.com/minjiechen/magnetchallenge-2

Not yet implemented (stub) — see scripts/README.md and PLAN.md §15 for the
required behavior once implemented: small subset only, never fail the whole
project if credentials/network are unavailable, print manual instructions
instead, and be idempotent.
"""
import argparse
import sys

DATASET_URL = "https://github.com/minjiechen/magnetchallenge-2"
TARGET_DIR = "data/public/magnet_challenge2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(f"scripts/{__file__.rsplit('/', 1)[-1]} is not yet implemented.")
    print(f"Dataset: {DATASET_URL}")
    print(f"Expected target directory: {TARGET_DIR}")
    print("Manual fallback: download the dataset from the URL above and place it under the target directory.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
