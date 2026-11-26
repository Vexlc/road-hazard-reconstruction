"""Place each metric object independently in the same surveyed PandaSet road patch."""

import argparse
from contextlib import closing
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from traffic_safe_drive import pandaset, read, write

renderer = import_module("render-mixed-road")
from cargo_support import check_members
from mesh_transfer import RoadMesh, fit_footprint, fit_road, footprint_size
from prop_lighting import fit_receivers
from traffic_clearance import world_bounds


def pose_matrix(record):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([record["heading"][axis] for axis in "xyzw"]).as_matrix()
    pose[:3, 3] = [record["position"][axis] for axis in "xyz"]
    return pose


@lru_cache(maxsize=20)
def cloud(raw, frame):
    data = pd.read_pickle(raw / "lidar" / f"{frame:02}.pkl.gz")
    return data[["x", "y", "z"]].to_numpy()


@lru_cache(maxsize=4)
def occluding_cloud(raw,frame):
    sem=raw/"annotations/semseg"
    (reflection,)=[int(k) for k,v in read(sem/"classes.json").items() if v=="Reflection"]
    labels=pd.read_pickle(sem/f"{frame:02}.pkl.gz")["class"].to_numpy()
    return cloud(raw,frame)[labels!=reflection]


def road_points(raw, frame):
    ids = pd.read_pickle(raw / "annotations/semseg" / f"{frame:02}.pkl.gz")["class"].to_numpy()
    return cloud(raw, frame)[ids == 7]


def host_patch(raw, shot, project, poses, near_pixel):
    pose = pose_matrix(poses[shot])
    road = road_points(raw, shot)
    optical = (road - pose[:3, 3]) @ pose[:3, :3]
    valid = (optical[:, 2] > 1) & (optical[:, 2] < 30)
    uv = project(optical[valid])
    distance = np.linalg.norm(uv - near_pixel, axis=1)
    if distance.min() > 22:
        raise ValueError("No labeled road return near the selected camera pixel")
    seed = road[valid][distance.argmin()]
    local = []
    for frame in range(max(0, shot - 16), min(80, shot + 17), 2):
        pts = road_points(raw, frame)
        local.append(pts[np.linalg.norm(pts - seed, axis=1) < 4])
    return seed, np.concatenate(local), pose[:3, 2]


def nearby_color(image, points, pose, project):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    uv = np.rint(project(camera[camera[:, 2] > 0])).astype(int)
    w, h = image.size
    uv = uv[(uv >= 0).all(1) & (uv < [w, h]).all(1)]
    if len(uv) < 20:
        raise ValueError("Too few visible road returns for exposure matching")
    return np.median(np.asarray(image)[uv[:, 1], uv[:, 0]], axis=0).tolist()


def frame_detail(image, result, size):
    x, y = result["xy"]
    rows, cols = np.nonzero(result["part"])
    cx = x + (cols.min() + cols.max()) // 2
    cy = y + (rows.min() + rows.max()) // 2
    scale = max(1.0, 1.2 * (np.ptp(cols) + 1) / size[0], 1.2 * (np.ptp(rows) + 1) / size[1])
    scale = min(scale, image.width / size[0], image.height / size[1])
    w, h = [round(v * scale) for v in size]
    # Slide the crop along the edge instead of adding black padding to the photo.
    left = int(np.clip(cx - w // 2, 0, image.width - w))
    top = int(np.clip(cy - h // 2, 0, image.height - h))
    return image.crop((left, top, left + w, top + h)).resize(size, Image.Resampling.LANCZOS)


def comparison(image, result, key, label, frame, font):
    full = image.copy()
    full.paste(Image.fromarray(result["after"]), tuple(result["xy"]))
    tile = Image.new("RGB", (960, 752), "#202226")
    tile.paste(full.resize((960, 540), Image.Resampling.LANCZOS))
    for offset, source in [(0, image), (480, full)]:
        tile.paste(frame_detail(source, result, (480, 172)), (offset, 540))
    text = ImageDraw.Draw(tile)
    text.text((10, 716), f"原始画面  ·  {frame:02d}", font=font)
    text.text((490, 716), f"{key}   {label}", font=font)
    return tile


def render(args, out, writer, font):
    raw, models = args.raw, args.models
    survey, shots, poses, lens = pandaset(raw)
    front = raw / "camera/front_camera"
    camera = raw / "camera" / args.camera
    with Image.open(camera / f"{args.shot:02}.jpg") as image:
        width, height = image.size

    def pinhole(lens):
        k = [[lens["fx"], 0, lens["cx"]], [0, lens["fy"], lens["cy"]], [0, 0, 1]]
        return renderer.view_model(
            "camera", {"source_calibration": {"camera": dict(size=[width, height], model="PINHOLE", K=k)}}
        )

    seed, points, heading = host_patch(raw, args.shot, pinhole(lens)[1], poses, args.pixel)
    if camera != front:
        shots, poses, lens = [
            read(camera / file) for file in ("timestamps.json", "poses.json", "intrinsics.json")
        ]
    model = pinhole(lens)
    window = renderer.RayWindow(model)
    # The rear window includes the roof. Reserve the entire lower band for the recording car.
    cutoff = height if camera == front else round(0.73 * height)
    ego = dict(grid=[width, height], edge=[[0, cutoff], [width, cutoff]])
    scan_times = np.array(read(raw / "lidar/timestamps.json"))
    daylight = read(args.daylight) if args.daylight else {}
    if daylight and daylight["scene"] != raw.name:
        raise ValueError("Sunlight must be measured in the selected sequence")
    texture_points = points[np.linalg.norm(points - seed, axis=1) < 0.9]
    plate_rows, reports = [], {}
    headings = read(args.headings) if args.headings else {}
    for folder in sorted(models.iterdir()):
        if not (folder / "asset.glb").is_file():
            continue
        key = folder.name
        meta = read(folder / "metadata.json")
        mesh = RoadMesh(folder, filter_albedo=True)
        direction = Rotation.from_euler("z", headings.get(key, args.yaw), degrees=True).apply(heading)
        if meta.get("gravity_support"):
            from prop_balance import fit_balanced
            anchor = fit_balanced(folder, seed, points, direction, meta)
        else:
            anchor = fit_road(points, seed, direction, up_world=[0, 0, 1], radius_m=1.05)
            anchor = fit_footprint(anchor, points, np.maximum(footprint_size(folder), 0.5))
        grouped = bool(meta.get("member_ground_check"))
        support = check_members(folder, anchor, points, single=not grouped)
        if grouped:
            anchor["member_support"] = support
            anchor["registered_geometry"] = [renderer.registered_geometry(folder)]
        anchor["ground_receivers"] = fit_receivers(points, anchor)
        boxes = {key: world_bounds(anchor, mesh.bounds)}
        checks = survey.evaluate(boxes)
        dest = out / key
        dest.mkdir()
        best, frames = None, []
        frame_range = (args.shot - 8, args.shot + 7) if camera == front else (args.shot, args.shot + 45)
        for frame in range(max(0, frame_range[0]), min(80, frame_range[1])):
            clear = survey.exposure(checks, shots[frame])
            row = dict(frame=frame, time_s=shots[frame], traffic=clear)
            frames.append(row)
            if not clear["clear"]:
                continue
            pose = pose_matrix(poses[frame])
            path = camera / f"{frame:02}.jpg"
            optical = (np.asarray(anchor["center_world"]) - pose[:3, 3]) @ pose[:3, :3]
            if (
                optical[2] <= 0
                or not ((model[1](optical) >= 0) & (model[1](optical) < [width, cutoff])).all()
            ):
                continue
            with Image.open(path) as image:
                anchor["lighting"] = dict(road_srgb=nearby_color(image, texture_points, pose, model[1]))
            if daylight:
                anchor["lighting"].update(
                    sun_world=daylight["sun_world"], direct_fraction=daylight["direct_fraction"]
                )
            scan = int(np.argmin(abs(scan_times - shots[frame])))
            row["lidar_frame"] = scan
            result = renderer.frame_crop(mesh, anchor, model, pose, path, cloud(raw, scan), window, ego)
            if not result or result["clipped_sides"] or result["touches_mask"]:
                continue
            if grouped:
                counts = np.bincount(result["instance"].ravel(), minlength=len(meta["instances"]) + 1)[1:]
                if min(counts) < 12:
                    continue
                row["member_pixels"] = counts.tolist()
            x, y = result["xy"].tolist()
            h, w = result["part"].shape
            camera_index = 0 if camera == front else 5
            with Image.open(args.masks / f"{frame:03}_{camera_index}.png") as mask:
                if mask.size != (width, height):
                    raise ValueError("Dynamic mask resolution differs from the camera")
                traffic = np.asarray(mask.convert("L").crop((x, y, x + w, y + h))) != 0
            if np.any(traffic & ((result["part"] > 0) | (result["shadow"] > 0))):
                row["masked_overlap"] = True
                continue
            row.update(
                visible_px=result["visible"], crop_xy=[x, y], road_srgb=anchor["lighting"]["road_srgb"]
            )
            with Image.open(path) as image:
                tile = comparison(image, result, key, meta["label"], frame, font)
                writer.send(tile.tobytes())
            if best is None or result["visible"] > best[0]["visible_px"]:
                best = row, path, result
        if best is None:
            raise ValueError(f"No unoccluded, traffic-clear exposure for {key}")
        row, path, result = best
        x, y = row["crop_xy"]
        before = Image.fromarray(result["before"])
        after = Image.fromarray(result["after"])
        with Image.open(path) as image:
            full = image.convert("RGB")
        full.paste(after, (x, y))
        full.save(dest / "drive.jpg", quality=95)
        before.save(dest / "before.png")
        after.save(dest / "after.png")
        Image.fromarray(result["part"]).save(dest / "parts.png")
        if grouped:
            Image.fromarray(result["instance"]).save(dest / "instances.png")
        np.save(dest / "range_m.npy", result["range"])
        anchor["lighting"]["road_srgb"] = row["road_srgb"]
        report = dict(
            scene=raw.name,
            camera=args.camera,
            asset=key,
            glb_sha256=mesh.digest,
            anchor=anchor,
            world_xy_bounds_m=boxes[key].tolist(),
            contact=max(support["instances"], key=lambda member: member["max_measured_gap_m"]),
            frames=frames,
            best_frame=row["frame"],
            best_visible_px=row["visible_px"],
            best_crop_xy=[x, y],
        )
        write(dest / "placement.json", report)
        reports[key] = dict(
            frame=row["frame"],
            visible_px=row["visible_px"],
            clearance_m=row["traffic"]["lower_bound_m"],
            contact_gap_m=report["contact"]["max_measured_gap_m"],
        )
        detail = frame_detail(full, result, (360, 178))
        plate_rows.append((key, meta["label"], detail))
        print(key, reports[key], flush=True)
    if not plate_rows:
        raise ValueError("No standalone models found")
    make_plate(plate_rows, out / "roadside.jpg", font)
    write(out / "summary.json", reports)


def make_plate(rows, dest, font):
    plate = Image.new("RGB", (360 * 2, 210 * ((len(rows) + 1) // 2)), "#202226")
    draw = ImageDraw.Draw(plate)
    for i, (key, label, crop) in enumerate(rows):
        x, y = (i % 2) * 360, (i // 2) * 210
        plate.paste(crop, (x, y))
        draw.text((x + 10, y + 181), f"{key}   {label}", font=font, fill="white")
    plate.save(dest, quality=95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("raw", type=Path, help="PandaSet sequence including raw annotations")
    parser.add_argument("masks", type=Path, help="Matching processed dynamic_masks/all")
    parser.add_argument("out", type=Path)
    parser.add_argument("--shot", type=int, default=35, choices=range(80), metavar="0..79")
    parser.add_argument("--pixel", nargs=2, type=int, default=[830, 835])
    parser.add_argument("--camera", choices=["front_camera", "back_camera"], default="front_camera")
    parser.add_argument("--daylight", type=Path, help="Measured scene-specific direct light")
    parser.add_argument("--headings", type=Path, help="Optional per-asset heading overrides in degrees")
    parser.add_argument("--yaw", type=float, default=25)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh output directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as scratch:
        stage = Path(scratch) / "views"
        stage.mkdir()
        font_path = check_output(["fc-match", "-f", "%{file}", "sans:lang=zh"], text=True)
        font = ImageFont.truetype(font_path, 19)
        with closing(
            imageio_ffmpeg.write_frames(
                str(stage / "drive.mp4"),
                (960, 752),
                fps=10,
                codec="libx264",
                output_params=["-crf", "17", "-threads", "2"],
            )
        ) as writer:
            writer.send(None)
            render(args, stage, writer, font)
        stage.rename(args.out)
