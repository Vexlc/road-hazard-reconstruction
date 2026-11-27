"""Compare every native model product, including its material indices and sampling grid."""

import argparse
import hashlib
import json
from pathlib import Path


def compare(first, second, output):
    files = [Path("grate.glb"), Path("empty.glb")]
    for state in json.loads(Path(__file__).with_name("survey.json").read_text())[
        "relative_grate_poses"
    ]:
        for name in (
            "asset.glb",
            "asset.ply",
            "metadata.json",
            "surface.npz",
            "splat_material.npz",
        ):
            files.append(Path(state) / "model/series/assets/missing/M589" / name)
    report = {}
    for rel in files:
        a, b = [hashlib.sha256((p / rel).read_bytes()).hexdigest() for p in [first, second]]
        if a != b:
            raise ValueError(f"Independent reconstruction differs at {rel}")
        report[rel.as_posix()] = a
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("first", "second", "output"):
        p.add_argument(n, type=Path)
    compare(**vars(p.parse_args()))
