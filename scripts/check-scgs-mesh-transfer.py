"""Replay exported metric ray packets on rendered SC-GS fisheye RGB and compare them to the photo host."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from mesh_transfer import RoadMesh, pixel_rays, resolve_samples  # noqa: E402
from road_geometry import read_json, save_json, sha256  # noqa: E402


def project_directions(rays, cal):
    model, width, height, fx, fy, cx, cy, *k = cal
    if model != "OPENCV_FISHEYE" or len(k) != 4:
        raise ValueError("Use an exported SC-GS calibrated fisheye sensor frame")
    r = np.linalg.norm(rays[..., :2], axis=-1)
    theta = np.arctan2(r, rays[..., 2])
    rad = theta * np.polynomial.polynomial.polyval(theta**2, [1, *k])
    s = np.divide(rad, r, out=np.ones_like(rad), where=r > 1e-8)
    uv = rays[..., :2] * s[..., None] * [fx, fy] + [cx - 0.5, cy - 0.5]
    valid = np.all((uv >= 0) & (uv < [width - 1, height - 1]), axis=-1)
    return np.float32(uv), valid


def native_samples(cal, project_raw, crop_xy, from_range, from_mask):
    _, width, height, fx, fy, cx, cy, *k = cal
    source_h, source_w = from_range.shape
    samples = []
    for dx, dy in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]:
        rays = pixel_rays([cal[0], width, height, fx, fy, cx - dx, cy - dy, *k])
        coord = np.float32(project_raw(rays) - crop_xy)
        available = (rays[..., 2] > 0) & (coord >= 0).all(-1) & (coord < [source_w - 1, source_h - 1]).all(-1)
        near = cv2.remap(
            from_range, coord, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=np.inf
        )
        hard = cv2.remap(
            np.uint8(from_mask), coord, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1
        )
        samples.append((rays, near, ~available | hard.astype(bool)))
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/mixed-road/sc-gs")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs/mixed-road")
    ap.add_argument("--views", type=Path, default=ROOT / "outputs/mixed-road")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ids = ("P223", "A225", "M224")
    meshes = {key: RoadMesh(next(args.runs.glob("*/series/assets/*/" + key))) for key in ids}
    result = {}
    gallery = Image.new("RGB", (1216, 632), "#f2f1eb")
    draw = ImageDraw.Draw(gallery)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 21, index=2)
    title = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 30, index=2)
    draw.text((18, 8), "同一份米制网格迁入 SC-GS 鱼眼画面", font=title, fill="#172d29")
    for j, label in enumerate(("SC-GS 原路面", "P223   相邻浅坑", "A225   方盖横移", "M224   方井缺盖")):
        draw.text((18 + j * 301, 75), label, font=font, fill="#233330")
    for i, (scene, sensor) in enumerate([("kitti360-0000", "image_03"), ("a2d2-frontleft", "rear_center")]):
        base = args.views / scene
        anchor = read_json(base / "anchor.json")
        chosen = read_json(base / "index.json")["M224"]["views"][sensor]
        split = read_json(ROOT / "work/streetcrafter-rig" / scene / "all/split.json")
        (name,) = [
            n
            for n, v in split["views"].items()
            if v["sensor"] == sensor and v["source_frame"] == chosen["frame"] and not v["tile"]
        ]
        cal = split["projections"]["gut_train"][split["views"][name]["projection"]]
        checkpoint = ROOT / "runs/streetcrafter-rig" / scene / f"sc-gs-{args.steps}"
        frame = checkpoint / "sensor_frames" / name
        weights = checkpoint / "fit_node/point_cloud" / f"iteration_{args.steps}" / "point_cloud.ply"
        warp = checkpoint / "fit_node/deform" / f"iteration_{args.steps}" / "deform.pth"
        host = np.asarray(Image.open(frame).convert("RGB"))
        if list(host.shape[1::-1]) != cal[1:3]:
            raise ValueError("SC-GS output size disagrees with the sensor calibration")
        with np.load(base / "M224" / sensor / "view.npz") as packet:
            rays, pose, host_range, protected, photo = (
                packet[k] for k in ("rays_camera", "camera_to_world", "host_range", "protected", "rgb")
            )
        samples, valid = [], []
        for ray in rays:
            uv, inside = project_directions(ray, cal)
            samples.append(cv2.remap(host, uv, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
            valid.append(inside)
        fully_valid = np.all(valid, axis=0)
        target = np.rint(np.mean(samples, axis=0)).astype("u1")
        guarded = protected | ~fully_valid
        raw_project = import_module("render-mixed-road").view_model(sensor, split)[1]
        native = native_samples(
            cal, raw_project, np.asarray(chosen["crop_xy"]), host_range, np.any(protected, axis=0)
        )
        panels = [host]
        union = np.zeros(host.shape[:2], bool)
        row = dict(
            sensor=sensor,
            source_frame=chosen["frame"],
            scgs_frame=name,
            host_rgb_sha256=sha256(frame),
            point_cloud_sha256=sha256(weights),
            deformation_sha256=sha256(warp),
            models={},
        )
        for key, mesh in meshes.items():

            def render(background):
                return resolve_samples(
                    [mesh.render(background, pose, r, anchor, host_range, m) for r, m in zip(rays, guarded)]
                )

            out, part, dist = render(target)
            _, reference, ref_dist = render(photo)
            hits = part > 0
            if (
                hits.sum() < 25
                or not np.array_equal(part, reference)
                or not np.allclose(dist, ref_dist, rtol=0, atol=0, equal_nan=True)
                or not np.array_equal(out[~hits], target[~hits])
                or np.any(hits & ~fully_valid)
            ):
                raise ValueError(f"SC-GS replay, depth or protected pixels differ for {scene}/{key}")
            folder = args.out / scene / key
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out).save(folder / "after.png")
            Image.fromarray(part).save(folder / "part.png")
            np.save(folder / "asset-range.npy", dist)
            full, native_part, _ = resolve_samples(
                [mesh.render(host, pose, r, anchor, near, mask) for r, near, mask in native]
            )
            support = native_part > 0
            if not support.any() or not np.array_equal(full[~support], host[~support]):
                raise ValueError("Invalid native SC-GS sensor support")
            Image.fromarray(full).save(folder / "sensor-after.png")
            Image.fromarray(native_part).save(folder / "sensor-part.png")
            union |= support
            panels.append(full)
            row["models"][key] = dict(
                glb_sha256=mesh.digest,
                visible_pixels=int(hits.sum()),
                native_sensor_pixels=int(support.sum()),
                changed_pixels=int(np.any(target != out, axis=-1).sum()),
                depth_matches_photo_exactly=True,
            )
        top = 115 + i * 251
        yy, xx = np.nonzero(union)
        margin = max(7, int((xx.max() - xx.min()) * 0.32))
        rect = (
            max(0, xx.min() - margin),
            max(0, yy.min() - margin),
            min(host.shape[1], xx.max() + margin + 1),
            min(host.shape[0], yy.max() + margin + 1),
        )
        for j, image in enumerate(panels):
            panel = ImageOps.contain(Image.fromarray(image).crop(rect), (285, 176), Image.Resampling.LANCZOS)
            gallery.paste(panel, (17 + 301 * j + (285 - panel.width) // 2, top + (176 - panel.height) // 2))
        draw.text(
            (18, top + 181),
            f"{scene}   /   {sensor}   ·   实际镜头曝光 {chosen['frame']}",
            font=font,
            fill="#293c39",
        )
        result[scene] = row
    gallery.save(args.out / "scgs-fisheye.jpg", quality=94)
    save_json(args.out / "index.json", result)
    print("Six placements on two independent SC-GS fisheye hosts matched the physical-camera hit ranges.")


if __name__ == "__main__":
    main()
