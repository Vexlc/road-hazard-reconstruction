"""Rebuild both surveyed anchors and compare all model and camera outputs to the fixed release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from cargo_support import check_members
from mesh_transfer import fit_footprint, footprint_size
from road_geometry import glb_triangles


def check(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}: {actual}")


def cli(*args):
    subprocess.run([sys.executable, str(ROOT / "scripts/transfer-mesh-view.py"), *map(str, args)], check=True)


def check_contact_controls(model):
    axis = np.arange(-1.6, 1.61, .025)
    x, y = np.meshgrid(axis, axis)
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    plane = dict(center_world=[0., 0., 0.], basis_columns=np.eye(3).tolist())
    check_members(model, plane, points)
    x0, y0 = json.loads((model / "metadata.json").read_text())["instances"][1]["xy_m"]
    below = (abs(points[:, 0] - x0) < .35) & (abs(points[:, 1] - y0) < .35)
    raised = points.copy()
    raised[below, 2] += .055
    vertices = np.concatenate([v for _, v, _ in glb_triangles(model / "asset.glb")])
    foot = vertices[np.argmin(vertices[:, 2]), :2]
    nearby = np.linalg.norm(points[:, :2] - foot, axis=1) < .16
    for name, surveyed in [("missing returns", points[~below]), ("raised road", raised),
                           ("unobserved foot", points[~nearby])]:
        # The combined rectangle still passes. Only the member check can reject these cases.
        anchor = fit_footprint(plane, surveyed, footprint_size(model))
        try:
            check_members(model, anchor, surveyed)
        except ValueError:
            continue
        raise ValueError("Unsupported placement accepted: " + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "packets", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    manifest = json.loads((ROOT / "validation/lost-cargo.json").read_text())
    cams = json.loads((args.packets / "cameras.json").read_text())
    for key, files in manifest["models"].items():
        for name, expected in files.items():
            check(args.models / key / name, expected)
    count = 0
    for pair, cases in cams.items():
        model = args.models / pair
        check_contact_controls(model)
        dest = args.out / pair
        dest.mkdir(parents=True, exist_ok=True)
        src = args.packets / pair / "cycles"
        frame = json.loads((src / "frame.json").read_text())
        cli("road-points", dest / "road.npy", src / "front.npz", src / "edge.npz", src / "rear.npz", "--stride", 2)
        cli("fit", dest / "road.npy", dest / "anchor.json", "--seed", *frame["seed"], "--heading",
            *frame["heading"], "--up", *frame["up"], "--radius", 1.5, "--asset", model, "--lighting", src / "light.json")
        check(dest / "anchor.json", manifest["refit_anchors"][pair])
        for case in cases:
            group, view = case.split("/")
            anchor = dest / "anchor.json" if group == "cycles" else args.packets / pair / group / "anchor.json"
            cli("render", model, anchor, args.packets / pair / (case + ".npz"), dest / case)
            for name, expected in manifest["replay"][pair][case].items():
                check(dest / case / name, expected)
                count += 1
    print(f"Verified {count} camera files, two refitted anchors, six rejected placements and all model files")


if __name__ == "__main__":
    main()
