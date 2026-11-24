"""Measure instance, metric range and grade errors against independently cut native Cycles meshes."""

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np
from PIL import Image

reference = import_module("check-cone-cycles")
ROOT = reference.ROOT
from host_depth import geometry
from road_geometry import save_json
from pavement_layout import PavementLayout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=ROOT / "outputs/measured-pavement/cycles")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/measured-pavement/cycles-check")
    p.add_argument("--assets", type=Path, default=ROOT / "runs/measured-pavement")
    p.add_argument("--layout",type=Path,default=ROOT / "gen2hole/examples/measured-pavement/crown-layout.json")
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    recipe = a.layout
    views = ["center", "edge", "rear"]
    cams = {}
    for name in views:
        cams[name], world, _ = reference.packet(a.views / name / "host", name == "edge", flat_reference=False)
        np.savez_compressed(a.out / (name + ".npz"), **cams[name])
    cli = reference.cli
    cli("road-points", a.out / "road.npy", a.out / "center.npz", a.out / "edge.npz", "--stride", 2)
    cli(
        "fit",
        a.out / "road.npy",
        a.out / "anchor.json",
        "--seed",
        *(world[:3, 3] + world[:3, 2] * 0.024),
        "--heading",
        *world[:3, 0],
        "--up",
        *world[:3, 2],
        "--radius",
        0.85,
        "--asset",
        recipe,
        "--asset-root",
        a.assets,
    )
    anchor = json.loads((a.out / "anchor.json").read_text())
    true = json.loads((a.views / "reference.json").read_text())
    model = PavementLayout(recipe, a.assets)
    model.bind(anchor)
    stats = dict(poses=[], views={})
    normals = []
    for seat, gt in zip(anchor["instances"], true):
        origin = np.array(seat["anchor"]["center_world"])
        n = np.array(seat["anchor"]["basis_columns"])[:, 2]
        normals.append(n)
        native = np.array(gt)
        delta = float(np.linalg.norm(origin - native[:3, 3]))
        angle = float(np.degrees(np.arccos(np.clip(n @ native[:3, 2], -1, 1))))
        stats["poses"].append(dict(id=seat["id"], center_error_m=delta, normal_error_deg=angle))
        if delta > 0.005 or angle > 0.1:
            raise ValueError(f"Independent survey: center {delta} m, normal {angle} degrees")
    stats["between_normals_deg"] = float(np.rad2deg(np.arccos(np.clip(normals[0] @ normals[1], -1, 1))))
    if stats["between_normals_deg"] < 7:
        raise ValueError("Expected separate grade on the two sides")
    for name in views:
        cam = cams[name]
        dest = a.out / name
        cli("render", recipe, a.out / "anchor.json", a.out / (name + ".npz"), dest, "--asset-root", a.assets)
        parts = np.asarray(Image.open(dest / "instance.png"))
        depth = np.load(dest / "range.npy")
        ids = np.load(a.views / name / "native/id.npy")
        dist = np.load(a.views / name / "native/range.npy")
        truth = np.where(ids >= 3, ids - 2, 0)
        before = np.load(a.views / name / "host/id.npy")
        # Render the same view with the measured occluder removed to count tested foreground pixels.
        rays, _, mask, pose = geometry(cam)
        clear = model.sample(cam["rgb"], pose, rays[0], anchor, np.full(parts.shape, np.inf, "f4"), mask[0])[
            3
        ]
        covered = int(((before == 2) & (clear > 0)).sum())
        assert not np.any(parts[before == 2])
        per = {}
        for i, seat in enumerate(anchor["instances"], 1):
            hit = parts == i
            expected = truth == i
            if not expected.any():
                if hit.any():
                    raise ValueError("Projected a hidden instance")
                continue
            overlap = expected & hit
            iou = float(overlap.sum() / (expected | hit).sum())
            err = float(np.percentile(abs(depth[overlap] - dist[overlap]), 95))
            behind = int((overlap & (rays[0, ..., 2] < 0)).sum())
            per[seat["id"]] = dict(
                native=int(expected.sum()),
                portable=int(hit.sum()),
                iou=iou,
                p95_range_error_m=err,
                rear_hemisphere_pixels=behind,
            )
            print(name, seat["id"], per[seat["id"]], flush=True)
            if iou < 0.977 or err > 0.005:
                raise ValueError("Recessed mesh failed independent ray or metric test")
        stats["views"][name] = dict(instances=per, host_occluded_opening_pixels=covered)
    if sum(r["host_occluded_opening_pixels"] for r in stats["views"].values()) < 25:
        raise ValueError("No measured occlusion")
    if sum(v["rear_hemisphere_pixels"] for v in stats["views"]["edge"]["instances"].values()) < 60:
        raise ValueError("No behind-lens check")
    save_json(a.out / "check.json", stats)


if __name__ == "__main__":
    main()
