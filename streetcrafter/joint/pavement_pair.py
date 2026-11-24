"""Cut and seat independently surveyed pavement openings in the native KITTI joint checkpoint."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from pit_pairs import BASE, ROOT, RIG, SCENE, digest, load_rig, road_plate, terrain_asset, write_pair

sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from pavement_layout import PavementLayout


def trim_road_support(members, ring_width):
    """Use the nearest opening's pavement repair ring once, preserving both measured cavities."""
    for index, ((own, origin, basis), layers, _, record) in enumerate(members):
        ring = layers[1]
        world = ring["_xyz"].numpy()
        own_d = own.signed_distance(((world - origin) @ basis)[:, :2])
        keep = own_d < ring_width
        for other, ((cut, center, axes), _, _, _) in enumerate(members):
            if other == index:
                continue
            dist = cut.signed_distance(((world - center) @ axes)[:, :2])
            keep &= (dist > 0.025) & ((own_d < dist) | ((own_d == dist) & (index < other)))
        layers[1] = {k: v[keep] for k, v in ring.items()}
        record["road_support_points"] = int(keep.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("anchor", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--ground-method",
        choices=["split", "whole"],
        default="split",
        help="Use whole-particle removal to reproduce the comparison baseline",
    )
    args = parser.parse_args()
    from road_geometry import read_json

    layout = PavementLayout(args.recipe, args.assets)
    anchor = read_json(args.anchor)
    if anchor.get("host_scene") != SCENE or len(layout.members) != 2:
        raise ValueError("Expected two openings surveyed in this KITTI host")
    layout.bind(anchor)
    baseline = torch.load(BASE / "joint.pt", weights_only=False, map_location="cpu")
    if (baseline["scene"], baseline["steps"]) != (SCENE, 30000):
        raise ValueError("Incorrect original host checkpoint")
    split, cameras = load_rig(RIG)
    members = []
    for seat, (_, folder) in zip(anchor["instances"], layout.members):
        folder = folder.resolve()
        registered = seat["anchor"]
        center = np.asarray(registered["center_world"])
        basis = np.asarray(registered["basis_columns"])
        stats = dict(
            instance=seat["id"],
            support=registered["footprint_support"],
            surface_sha256=seat["surface_sha256"],
            mesh_sha256=seat["mesh_sha256"],
        )
        with np.load(folder / "surface.npz") as f:
            pitch = (float(f["width"]) * float(f["length"])) / np.prod(np.array(f["defect_mask"].shape) - 1)
            area = float((f["defect_mask"] > 0.5).sum() * pitch)
        case = dict(
            model=str(folder / "asset.glb"),
            outline_area_m2=area,
            source_ply=seat["asset"] + "/asset.ply",
            label=read_json(folder / "metadata.json")["label"],
            native_hashes={"asset.ply": digest(folder / "asset.ply")},
        )
        plate = road_plate(center, basis, cameras, split)
        members.append(terrain_asset(seat["asset"], case, (center, basis, stats), plate, None, voxel=0.009))
    split_ground = args.ground_method == "split"
    trim_road_support(members, 0.18 if split_ground else 1.45)
    dest = args.out / layout.key
    dest.mkdir(parents=True, exist_ok=True)
    write_pair(
        layout.key,
        dest,
        baseline,
        digest(BASE / "joint.pt"),
        members,
        cameras,
        split,
        split_ground=split_ground,
        minimum_separation=0.0,
    )


if __name__ == "__main__":
    main()
