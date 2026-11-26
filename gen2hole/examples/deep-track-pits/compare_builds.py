"""Verify all native products from two independent builds and record their digests."""

import argparse
import hashlib
import json
from pathlib import Path


def compare(first, second, output):
    result = {}
    for state in json.loads(Path(__file__).with_name("survey.json").read_text())["states"]:
        key = state["id"]
        names = ["asset.glb", "asset.ply", "surface.npz", "splat_material.npz", "metadata.json"]
        if state["section_parameters"][0] != "bowl":
            names.append("raised-edge.glb")
        result[key] = {}
        for name in names:
            rel = Path("series/assets/pothole") / key / name
            a, b = [
                hashlib.sha256((root / rel).read_bytes()).hexdigest() for root in (first, second)
            ]
            if a != b:
                raise ValueError(f"{rel} differs between builds")
            result[key][name] = a
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "output"):
        parser.add_argument(name, type=Path)
    compare(**vars(parser.parse_args()))
