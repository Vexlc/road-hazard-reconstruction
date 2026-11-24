"""Replay the localized native change in the same registered front and rear KITTI exposures."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
sys.path.insert(0, str(ROOT / "gen2hole/downstream/sensors"))
from pavement_layout import PavementLayout
from pavement_support import PavementSupport
from prop_lighting import linear, srgb
from road_geometry import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("layout", "models", "anchor", "renders", "out"):
        parser.add_argument(key, type=Path)
    parser.add_argument(
        "--support-m", type=float, default=0.18, help="Width of the road repair band in metres"
    )
    args = parser.parse_args()
    layout = PavementLayout(args.layout, args.models)
    anchor = json.loads(args.anchor.read_text())
    layout.bind(anchor)
    if anchor.get("host_scene") != "kitti360-0000" or args.out.resolve() == args.renders.resolve():
        raise ValueError("Expected a fitted KITTI placement and an output different from the raw renders")
    baseline = json.loads((args.renders / "original/native-source.json").read_text())
    source = json.loads((args.renders / layout.key / "native-source.json").read_text())
    placement = source["placement"]
    if placement["source_sha256"] != baseline["checkpoint_sha256"] or placement["pair"] != layout.key:
        raise ValueError("Native renders come from different checkpoints or placements")
    if len(placement["placements"]) != len(anchor["instances"]):
        raise ValueError("Unexpected native instance count")
    for installed, seat in zip(placement["placements"], anchor["instances"]):
        if (
            installed["id"] != seat["asset"]
            or installed["instance"] != seat["id"]
            or any(installed[k] != seat[k] for k in ("mesh_sha256", "surface_sha256"))
            or any(installed[k] != seat["anchor"][k] for k in ("center_world", "basis_columns"))
        ):
            raise ValueError("Native geometry or pose differs from the requested support")
    members = [
        (PavementSupport(layout.meshes[s["asset"]], args.support_m), s["anchor"])
        for s in anchor["instances"]
    ]
    dest = args.out / layout.key
    args.out.mkdir(parents=True, exist_ok=True)
    original = args.out / "original"
    if not original.exists():
        original.symlink_to((args.renders / "original").resolve(strict=True), target_is_directory=True)
    if original.resolve() != (args.renders / "original").resolve():
        raise ValueError("Source baseline changed for this output")
    records = {}
    for view in ["front", "rear"]:
        folder = dest / view
        folder.mkdir(parents=True, exist_ok=True)
        counts = dict(
            frames=0,
            raw_outside_changes=0,
            localized_outside_changes=0,
            preserved_interior_changes=0,
            max_interior_error=0,
        )
        frames = source["views"][view]
        if not frames or set(frames) != set(baseline["views"][view]):
            raise ValueError("Missing registered native exposures")
        for name, meta in sorted(frames.items()):
            if Path(name).name != name:
                raise ValueError("Invalid native exposure name")
            before = np.asarray(Image.open(original / view / name))
            native = np.asarray(Image.open(args.renders / layout.key / view / name))
            ref = baseline["views"][view][name]
            for key in ("world_to_camera", "calibration"):
                if ref[key] != meta[key]:
                    raise ValueError("The source and edited renders used different cameras")
            if (
                hashlib.sha256(before.tobytes()).hexdigest() != ref["rgb_sha256"]
                or hashlib.sha256(native.tobytes()).hexdigest() != meta["rgb_sha256"]
            ):
                raise ValueError("Native image changed after rendering")
            pose = np.linalg.inv(meta["world_to_camera"])
            k = meta["calibration"]
            _, width, height, fx, fy, cx, cy = k
            if before.shape != (height, width, 3) or native.shape != before.shape:
                raise ValueError("Unregistered native images")
            row, col = np.where(np.any(before != native, axis=-1))
            final = before.copy()
            if len(row):
                rays = np.stack([(col + 0.5 - cx) / fx, (row + 0.5 - cy) / fy, np.ones(len(row))], axis=1)
                rays /= np.linalg.norm(rays, axis=1)[:, None]
                rays = rays @ pose[:3, :3].T
                weight = np.maximum.reduce(
                    [support.weights(pose[:3, 3], rays, seat) for support, seat in members]
                )
                a, b = before[row, col], native[row, col]
                keep, blend = weight == 1, (weight > 0) & (weight < 1)
                final[row[keep], col[keep]] = b[keep]
                w = weight[blend, None]
                final[row[blend], col[blend]] = srgb((1 - w) * linear(a[blend]) + w * linear(b[blend]))
                delta = np.abs(a.astype(int) - b).max(1) > 8
                counts["raw_outside_changes"] += int(((weight == 0) & delta).sum())
                counts["localized_outside_changes"] += int(
                    (final[row[weight == 0], col[weight == 0]] != a[weight == 0]).any(1).sum()
                )
                counts["preserved_interior_changes"] += int((keep & delta).sum())
                if keep.any():
                    error = int(abs(final[row[keep], col[keep]].astype(int) - b[keep]).max())
                    counts["max_interior_error"] = max(counts["max_interior_error"], error)
            Image.fromarray(final).save(folder / name)
            counts["frames"] += 1
        if (
            not counts["preserved_interior_changes"]
            or counts["localized_outside_changes"]
            or counts["max_interior_error"]
        ):
            raise ValueError(f"Native composite failed: {counts}")
        records[view] = counts
    save_json(dest / "support.json", dict(support_m=args.support_m, views=records))


if __name__ == "__main__":
    main()
