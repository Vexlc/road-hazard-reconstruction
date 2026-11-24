"""Check full-frame, tightly cropped and painted exposures against independent fisheye renders."""

import argparse
import json
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

ref = import_module("check-cone-cycles")
from road_geometry import save_json

ROOT = ref.ROOT
PAIR = "P246-P247"
RECIPE = ROOT / "gen2hole/examples/asphalt-breakouts"


def render(recipe, assets, anchor, packet, dest):
    ref.cli("render", recipe, anchor, packet, dest, "--asset-root", assets)
    return (
        np.array(Image.open(dest / "after.png")).astype(int),
        np.array(Image.open(dest / "instance.png")),
        np.load(dest / "range.npy"),
    )


def tight_view(packet, box):
    y, x, r = box
    crop = (slice(y - r, y + r + 1), slice(x - r, x + r + 1))
    h, w = packet["rgb"].shape[:2]
    result = {}
    for k, a in packet.items():
        if k in {"camera_to_world", "sensor_name"}:
            result[k] = a
        elif a.shape[:2] == (h, w):
            result[k] = a[crop]
        elif a.ndim >= 3 and a.shape[1:3] == (h, w):
            result[k] = a[(slice(None), *crop)]
        else:
            raise ValueError("Unexpected field in camera packet: " + k)
    return result, crop


def closeups(args, temp):
    summary = {}
    for site in ["kitti", "audi"]:
        for cam in (
            ["image_00", "image_03"]
            if site == "kitti"
            else ["front_left", "rear_center"]
        ):
            parent = args.views / site
            folder = parent / PAIR / cam
            with np.load(folder / "view.npz") as p:
                packet = dict(p)
            ids = np.array(Image.open(folder / "instance.png"))
            image = np.asarray(Image.open(folder / "after.png")).astype(int)
            ranges = np.load(folder / "range.npy")
            anchor = parent / "anchor.json"
            legacy = json.loads(anchor.read_text())
            old = temp / "legacy.json"
            legacy.pop("pavement_appearance_by_camera")
            save_json(old, legacy)
            summary[cam] = {}
            for i, key in enumerate(["P246", "P247"], 1):
                clearance = cv2.distanceTransform(
                    np.pad((ids == i).astype("u1"), 1), cv2.DIST_L2, 5
                )[1:-1, 1:-1]
                y, x = np.unravel_index(clearance.argmax(), clearance.shape)
                radius = min(28, int(clearance[y, x] / 1.65))
                if radius < 4:
                    raise ValueError(f"Too few closeup pixels for {cam} {key}")
                v, c = tight_view(packet, (y, x, radius))
                assert (ids[c] == i).all()
                path = temp / "close.npz"
                np.savez_compressed(path, **v)
                new, parts, depth = render(
                    RECIPE / "layout.json", args.assets, anchor, path, temp / "new"
                )
                _, old_ids, _ = render(
                    RECIPE / "layout.json", args.assets, old, path, temp / "old"
                )
                core = (slice(1, -1), slice(1, -1))
                error = int(abs(new[core] - image[c][core]).max())
                depth_error = float(np.max(abs(depth[core] - ranges[c][core])))
                row = {
                    "pixels": int(parts.size),
                    "cropped_legacy_visible": int((old_ids == i).sum()),
                    "cropped_registered_visible": int((parts == i).sum()),
                    "max_core_rgb_error": error,
                    "max_core_range_error_m": depth_error,
                }
                print(cam, key, row, flush=True)
                if (parts != i).any() or error > 1 or depth_error > 1e-4:
                    raise ValueError(
                        "Tight crop does not reproduce the intact full frame"
                    )
                summary[cam][key] = row
    return summary


def paint_control(args, temp):
    dest = args.views / "controls"
    recipe = RECIPE / "crown.json"
    fitted = dest / "appearance.json"
    ref.cli(
        "calibrate-pavement",
        recipe,
        dest / "anchor.json",
        fitted,
        dest / "center.npz",
        dest / "edge.npz",
        "--asset-root",
        args.assets,
    )
    variants = [("image_context", dest / "anchor.json"), ("registered", fitted)]
    with np.load(dest / "center.npz") as view:
        has_albedo = "road_albedo_srgb" in view
    if has_albedo:
        comparison = []
        for name in ["center", "edge"]:
            frame = temp / (name + ".npz")
            with np.load(dest / (name + ".npz")) as view:
                np.savez_compressed(
                    frame, **{k: view[k] for k in view.files if k != "road_albedo_srgb"}
                )
            comparison.append(frame)
        without = temp / "rgb-only.json"
        ref.cli(
            "calibrate-pavement",
            recipe,
            dest / "anchor.json",
            without,
            *comparison,
            "--asset-root",
            args.assets,
        )
        variants.append(("without_albedo", without))
    evidence = {}
    for name in ["center", "edge", "rear"]:
        normal = dest / (name + ".npz")
        painted, _, _ = ref.packet(
            args.views / "cycles/painted" / name / "host",
            name == "edge",
            flat_reference=False,
        )
        path = temp / "painted.npz"
        np.savez_compressed(path, **painted)
        native = np.array(
            Image.open(args.views / "cycles" / name / "native/rgb.png").convert("RGB")
        ).astype(int)
        truth = np.load(args.views / "cycles" / name / "native/id.npy")
        with np.load(normal) as v:
            diff = abs(v["rgb"].astype(int) - painted["rgb"].astype(int)).max(2)
        inner = (truth >= 3) & (diff > 16)
        if inner.sum() < 10:
            raise ValueError("Insufficient actual paint under the rendered cut")
        row = {"native_marked_pixels": int(inner.sum())}
        for kind, anchor in variants:
            reference, ids, depth = render(
                recipe,
                args.assets,
                anchor,
                normal,
                dest / name if kind == "image_context" else dest / (name + "-" + kind),
            )
            after, labels, ranges = render(
                recipe, args.assets, anchor, path, temp / kind
            )
            if not np.array_equal(ids, labels) or not np.allclose(
                ranges, depth, equal_nan=True
            ):
                raise ValueError("Removable paint changed the aperture geometry")
            row[kind] = {
                "paint_delta_mae": float(abs(after - reference)[inner].mean()),
                "rgb_mae_to_cycles": float(abs(after - native)[inner].mean()),
            }
        if (
            row["registered"]["paint_delta_mae"] > 0.05
            or row["image_context"]["paint_delta_mae"] <= 1
        ):
            raise ValueError(
                "Marked host surface remains visible in replacement material"
            )
        if (
            has_albedo
            and row["registered"]["rgb_mae_to_cycles"]
            >= row["without_albedo"]["rgb_mae_to_cycles"]
        ):
            raise ValueError(
                "Known host albedo did not improve material calibration in the control scene"
            )
        print(name, row, flush=True)
        evidence[name] = row
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=ROOT / "outputs/asphalt-breakouts")
    p.add_argument("--assets", type=Path, default=ROOT / "runs/asphalt-breakouts")
    p.add_argument("--cycles-only", action="store_true")
    args = p.parse_args()
    with TemporaryDirectory() as tmp:
        stats = {"paint": paint_control(args, Path(tmp))}
        if not args.cycles_only:
            stats["closeups"] = closeups(args, Path(tmp))
    save_json(
        args.views
        / ("appearance-cycles.json" if args.cycles_only else "appearance.json"),
        stats,
    )


if __name__ == "__main__":
    main()
