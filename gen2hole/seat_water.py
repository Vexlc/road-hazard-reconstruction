"""Rebuild ponded water from the original measured beds at separately fitted road seats."""

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np

from gen2hole import ROOT, load, save
from fill_pits import build

sys.path.insert(0, str(ROOT / "downstream"))
from pavement_layout import PavementLayout
from road_geometry import sha256


def seat(args, dest):
    dry = PavementLayout(args.dry_layout, args.beds)
    anchor = load(args.dry_anchor)
    dry.bind(anchor)
    up = np.asarray(anchor.get("gravity_up_world"), dtype=float)
    if up.shape != (3,) or not np.isclose(np.linalg.norm(up), 1.0, atol=1e-7):
        raise ValueError("The dry survey needs gravity_up_world in the host coordinate frame")
    wet = load(args.layout)
    targets = wet["instances"]
    if len(targets) != len(dry.members):
        raise ValueError("The layouts must describe the same seats")
    survey = dict(beds={}, anchors={}, states=[])
    replacement = deepcopy(anchor)
    if "pavement_appearance_by_camera" in anchor:
        raise ValueError("Use the geometric dry survey, before registering appearance")
    for (base, folder), row, item in zip(dry.members, targets, replacement["instances"]):
        if any(row[key] != base[key] for key in ("id", "xy_m", "heading_deg")):
            raise ValueError("Target seat, heading or offset differs from the measured dry layout")
        bed, key = base["asset"], row["asset"]
        if "water" not in row:
            if key != bed:
                raise ValueError("Dry source identity changed")
            if not (dest / bed).exists():
                shutil.copytree(folder, dest / bed)
            continue
        state = row["water"]
        if (
            state["bed"] != bed
            or key == bed
            or not key.isalnum()
            or key in [s["id"] for s in survey["states"]]
        ):
            raise ValueError("Each wet ID must identify one seated state of the registered bed")
        bundle = folder.parents[2]
        if not (bundle / "calibration" / (bed + ".npz")).is_file():
            raise ValueError(
                "Rebuild the original dry capture to recover its calibrated bed and photographs"
            )
        site = item["id"]
        survey["beds"][bed] = dict(run=str(bundle.parent.resolve().relative_to(args.beds.resolve())))
        survey["anchors"][site] = (up @ np.asarray(item["anchor"]["basis_columns"])).tolist()
        survey["states"].append(
            dict(
                id=key,
                bed=bed,
                site=site,
                label=state["label"],
                input_liters=state["input_liters"],
                seed_uv=state["seed_uv"],
            )
        )
        item["asset"] = key
    if not survey["states"]:
        raise ValueError("Supply a pavement layout with at least one ponded bed")
    recipes = dest / "water-survey.json"
    save(recipes, survey)
    stage = dest / "water-build"
    stage.mkdir()
    build(
        argparse.Namespace(
            survey=recipes, runs=args.beds, config=args.config, mesh_only=False, preview=False
        ),
        stage,
    )
    final = PavementLayout(args.layout, dest)
    for (_, folder), item in zip(final.members, replacement["instances"]):
        item.update(mesh_sha256=sha256(folder / "asset.glb"), surface_sha256=sha256(folder / "surface.npz"))
    replacement["layout_sha256"] = final.digest
    final.bind(replacement)
    save(dest / "anchor.json", replacement)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dry_layout", "dry_anchor", "layout", "beds", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "config/local.json")
    a = p.parse_args()
    if a.out.exists():
        p.error("Choose a new output directory for the seated assets and anchor")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out.parent) as tmp:
        dest = Path(tmp) / "models"
        dest.mkdir()
        seat(a, dest)
        dest.rename(a.out)


if __name__ == "__main__":
    main()
