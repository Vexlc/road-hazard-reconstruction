"""Check the placed meshes against the source apertures and compare the registered drive exposures."""

import argparse
from importlib import import_module
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont, ImageOps

rig = import_module("check-cone-cycles")
from pavement_layout import asset_folder
from mesh_transfer import RoadMesh
from road_geometry import save_json


def check_cover(models, src):
    survey = json.loads((src / "004/survey.json").read_text())
    mask = np.array(Image.open(src / "004/openings.png")) > 0
    n, ids = cv2.connectedComponents(mask.astype("u1"))
    probes = []
    for i in range(1, n):
        dist = cv2.distanceTransform((ids == i).astype("u1"), cv2.DIST_L2, 5)
        row, col = np.unravel_index(dist.argmax(), dist.shape)
        probes.append([col, row])
    # This opaque strip bisects the obstructed upper-left vent in the photograph.
    probes.extend([[427, 306], [527, 495]])
    quad = np.asarray(survey["perforations"]["corners_xy"])
    xy = (np.asarray(probes) - quad[0]) / (quad[2] - quad[0]) - 0.5
    xy *= np.asarray(survey["perforations"]["size_m"]) * [1, -1]
    results = {}
    for key in ("A259", "M259"):
        folder = asset_folder(models, key)
        meta = json.loads((folder / "metadata.json").read_text())
        pose = np.array(meta["frame_transform"])
        starts = np.c_[xy, np.full(len(xy), 0.25)] @ pose[:3, :3].T + pose[:3, 3]
        dirs = np.broadcast_to(-pose[:3, 2], starts.shape)
        mesh = RoadMesh(folder)
        hits = mesh.scene.cast_rays(o3d.core.Tensor(np.c_[starts, dirs].astype("f4")))
        parts = [mesh.surfaces[int(i)][0] for i in hits["geometry_ids"].numpy()]
        if n != 18 or parts[:-2] != [7] * 17 or parts[-2:] != ([9, 9] if key == "A259" else [7, 7]):
            raise ValueError(f"Cover mesh disagrees with scan probes: {key}, {parts}")
        results[key] = dict(
            floor_rays_at_vent_mouths=parts[:-2].count(7), part_at_bridge=parts[-2], part_at_center=parts[-1]
        )
    return results


def review(root, models, src, pairs=None):
    viewsets = (
        ("kitti", "image_00", "KITTI 前视"),
        ("audi", "front_left", "Audi 左前"),
        ("audi", "rear_center", "Audi 驶过后车尾"),
    )
    canvas = Image.new("RGB", (1140, 480), "#252a2c")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20, index=2)
    stats = {}
    for i, pair in enumerate(pairs or ("P260-A259", "P261-M259")):
        for j, (site, sensor, title) in enumerate((*viewsets, ("kitti", "image_03", "KITTI 右侧"))):
            site_root = root / pair / site
            row = json.loads((site_root / "index.json").read_text())[pair]["views"][sensor]
            folder = site_root / pair / sensor
            before = np.asarray(Image.open(folder / "before.png"))
            after = np.asarray(Image.open(folder / "after.png"))
            label = np.asarray(Image.open(folder / "instance.png"))
            with np.load(folder / "view.npz") as packet:
                protected = packet["protected"].all(0)
                if (
                    label[protected].any()
                    or (after[protected] != before[protected]).any()
                    or not np.all(np.bincount(label.ravel(), minlength=3)[1:] > 12)
                ):
                    raise ValueError("Changed protected camera area or failed to resolve both openings")
            stats[f"{pair}/{sensor}"] = dict(
                frame=row["frame"],
                both_visible_frames=len(row["frames_showing_all_instances"]),
                unclipped_preview=row["unclipped_preview"],
                instance_pixels=row["best_frame_instance_pixels"],
                protected_edits=0,
            )
            if j == 3:
                continue
            yy, xx = np.nonzero(label)
            pic = Image.fromarray(after)
            crop = pic.crop(
                (
                    max(0, xx.min() - 12),
                    max(0, yy.min() - 12),
                    min(pic.width, xx.max() + 13),
                    min(pic.height, yy.max() + 13),
                )
            )
            crop = ImageOps.contain(crop, (368, 196), Image.Resampling.LANCZOS)
            x, y = j * 380, i * 240
            draw.text((x + 8, y + 7), f"{pair}   {title}", font=font, fill="white")
            canvas.paste(crop, (x + (380 - crop.width) // 2, y + 34 + (196 - crop.height) // 2))
    canvas.save(root / "road-pairs.jpg", quality=93)
    report = dict(views=stats) if pairs else dict(geometry=check_cover(models, src), views=stats)
    save_json(root / "real-check.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("views", type=Path)
    p.add_argument("assets", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("--pairs", nargs=2, help="Review these two layouts without the vented-cover scan checks")
    a = p.parse_args()
    review(a.views, a.assets, a.source, a.pairs)
