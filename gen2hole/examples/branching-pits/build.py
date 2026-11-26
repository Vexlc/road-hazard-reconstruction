"""Build three fractured pavement captures from their original survey coordinates."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wheel-track-pits"))
from build_scans import build, restrict_to_capture

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "materials", "out"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()), survey=HERE / "survey.json", refine=restrict_to_capture)
