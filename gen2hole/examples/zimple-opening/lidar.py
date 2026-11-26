"""Replay recorded vehicle beams and retain unresolved depths through the open intersection shaft."""

import argparse
import importlib.util
from pathlib import Path
from camera import HERE, S

spec = importlib.util.spec_from_file_location(
    "opening_replay", HERE.parent / "broadmoor-opening/lidar.py"
)
opening = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opening)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("models", "drive", "raw", "out"):
        parser.add_argument(key, type=Path)
    opening.check(
        **vars(parser.parse_args()),
        asset_id="M582",
        radius=S["assumed_outer_diameter_m"] / 2,
        extra_parts={"marker_hits": 176, "line_hits": 177},
        replacement_parts=(5, 6, 176, 177),
        replacement_state="replaced_road_with_observed_geometry"
    )
