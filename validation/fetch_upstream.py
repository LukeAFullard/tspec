"""Fetch pinned upstream Ghaderpour/LSWAVE-SignalProcessing into gitignored validation/upstream_ref/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

UPSTREAM_URL = "https://github.com/Ghaderpour/LSWAVE-SignalProcessing.git"
PINNED_SHA = "bd15923b1d4880b799b14fe016de4c1252259013"
TARGET_DIR = Path(__file__).resolve().parent / "upstream_ref"


def main() -> int:
    if TARGET_DIR.exists():
        sha = subprocess.check_output(
            ["git", "-C", str(TARGET_DIR), "rev-parse", "HEAD"], text=True
        ).strip()
        if sha == PINNED_SHA:
            print(f"Upstream oracle already present at pinned commit {sha}.")
            return 0
        print(f"Existing upstream_ref at {sha} != {PINNED_SHA}; updating...")
        subprocess.check_call(["git", "-C", str(TARGET_DIR), "fetch", "--depth=1", "origin", PINNED_SHA])
        subprocess.check_call(["git", "-C", str(TARGET_DIR), "checkout", PINNED_SHA])
        return 0

    print(f"Cloning {UPSTREAM_URL} into gitignored {TARGET_DIR}...")
    subprocess.check_call(["git", "clone", "--depth=1", UPSTREAM_URL, str(TARGET_DIR)])
    sha = subprocess.check_output(
        ["git", "-C", str(TARGET_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != PINNED_SHA:
        subprocess.check_call(["git", "-C", str(TARGET_DIR), "fetch", "--depth=1", "origin", PINNED_SHA])
        subprocess.check_call(["git", "-C", str(TARGET_DIR), "checkout", PINNED_SHA])
    print(f"Upstream oracle ready at {PINNED_SHA}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
