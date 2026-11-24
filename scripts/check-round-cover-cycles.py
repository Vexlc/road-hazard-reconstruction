"""Compare portable placements with the native Cycles part, occluder and metric range passes."""

import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image

reference = import_module("check-cone-cycles")
ROOT = reference.ROOT
from road_geometry import save_json


def ortho_packet(folder):
    meta = json.loads((folder / "camera.json").read_text())
    pose = np.asarray(meta["camera_to_world"])
    ground = np.asarray(meta["reference_ground_to_world"])
    size, width = meta["size"], meta["ortho_width_m"]
    y, x = (np.mgrid[:size, :size] + 0.5) / size - 0.5
    # Re-reference the same lines to a slightly curved origin surface. Radial ranges must
    # change with this offset while the signed optical Z of the host hit stays invariant.
    shift = 0.06 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    origins = np.float32(np.stack([width * x, width * y, shift], -1))
    rays = np.zeros_like(origins)
    rays[..., 2] = 1
    ids, distance = (np.load(folder / (name + ".npy")) for name in ("id", "range"))
    n = ground[:3, 2]
    plane = (ground[:3, 3] - pose[:3, 3] - origins @ pose[:3, :3].T) @ n / (pose[:3, 2] @ n)
    measured = distance - shift
    if np.percentile(abs(measured[ids == 1] - plane[ids == 1]), 99) > 0.0001:
        raise ValueError("Orthographic road depth disagrees with the native camera")
    return {
        "rgb": np.array(Image.open(folder / "rgb.png").convert("RGB")),
        "rays_camera": rays,
        "ray_origins_camera": origins,
        "camera_to_world": pose,
        "protected": ids == 0,
        "host_z": distance,
        "receiver_mask": ids == 1,
        "occluder_range": np.where(ids == 2, measured, np.inf),
    }, ground


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "runs/scanned-round-covers")
    parser.add_argument("--views", type=Path, default=ROOT / "outputs/scanned-round-covers/cycles")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/scanned-round-covers/oracle")
    args = parser.parse_args()
    report = {}
    for pair in ("M240-A241", "A240-M241"):
        dest = args.out / pair
        dest.mkdir(parents=True, exist_ok=True)
        source = args.views / pair
        recipe = ROOT / "gen2hole/examples/scanned-round-covers" / (pair + ".json")
        packets = {}
        packets["ortho"], world = ortho_packet(source / "ortho/host")
        packets["rear"], _, _ = reference.packet(source / "rear/host", False)
        for name, packet in packets.items():
            np.savez_compressed(dest / (name + ".npz"), **packet)
        cli = reference.cli
        cli("road-points", dest / "road.npy", dest / "ortho.npz", "--stride", 3)
        cli(
            "fit",
            dest / "road.npy",
            dest / "anchor.json",
            "--seed",
            *world[:3, 3],
            "--heading",
            *world[:3, 0],
            "--up",
            *world[:3, 2],
            "--radius",
            1.05,
            "--asset",
            recipe,
            "--asset-root",
            args.assets,
        )
        report[pair] = {}
        for name, packet in packets.items():
            view = dest / name
            cli(
                "render",
                recipe,
                dest / "anchor.json",
                dest / (name + ".npz"),
                view,
                "--asset-root",
                args.assets,
            )
            instances = np.asarray(Image.open(view / "instance.png"))
            parts = np.asarray(Image.open(view / "part.png"))
            native = np.load(source / name / "native/id.npy")
            ranges = np.load(source / name / "native/range.npy")
            if name == "ortho":
                ranges = ranges - packet["ray_origins_camera"][..., 2]
            known = native // 32
            labels = np.where(known > 0, native % 32, 0)
            protected = packet["protected"] | np.isfinite(packet["occluder_range"])
            assert not np.any(instances[protected])
            assert np.array_equal(
                np.asarray(Image.open(view / "after.png"))[protected], packet["rgb"][protected]
            )
            stats = {}
            for i, key in enumerate(pair.split("-"), 1):
                expected, hit = known == i, instances == i
                same = expected & hit & (parts == labels)
                iou = (expected & hit).sum() / (expected | hit).sum()
                agreement = same.sum() / expected.sum()
                err = np.percentile(abs(np.load(view / "range.npy")[same] - ranges[same]), 95)
                counts = {str(k): int((same & (parts == k)).sum()) for k in np.unique(labels[expected])}
                print(pair, name, key, iou, agreement, err, counts, flush=True)
                if iou < 0.968 or agreement < 0.956 or err > 0.003:
                    raise ValueError("Independent Cycles part or range comparison failed")
                if name == "ortho" and key.startswith("M") and counts.get("7", 0) < 1000:
                    raise ValueError("No independently verified well floor")
                stats[key] = {
                    "iou": float(iou),
                    "part_agreement": float(agreement),
                    "p95_range_error_m": float(err),
                    "matched_part_pixels": counts,
                }
            report[pair][name] = stats
    save_json(args.out / "check.json", report)


if __name__ == "__main__":
    main()
