"""Equivalent to `neurolens calibrate ...`; works from a checkout without `pip install`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurolens.cli import main

if __name__ == "__main__":
    sys.exit(main(["calibrate", *sys.argv[1:]]))
