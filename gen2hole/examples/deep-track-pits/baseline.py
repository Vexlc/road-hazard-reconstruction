"""Rebuild the bounded height-field comparison with exactly the same record IDs and folds."""

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bounded_scan", HERE.parent / "contact-pits/build.py")
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "materials", "out"):
        parser.add_argument(name, type=Path)
    bounded.build(
        **vars(parser.parse_args()), survey=HERE / "survey.json", refine=bounded.reconstruct_capture
    )
