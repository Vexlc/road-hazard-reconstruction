"""Match visible pavement samples before and after the car passes their fixed location."""

import argparse
from importlib import import_module
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

motion = import_module("check-rolling-props")
from road_geometry import Surface, read_json, save_json, sha256


def exposure(folder, raw, surface, digest):
    record = read_json(folder / "placement.json")
    key = folder.name
    if (
        record["asset"] != key
        or record["scene"] != raw.name
        or record["glb_sha256"] != digest
    ):
        raise ValueError(f"Wrong camera packet for {key}")
    cam = raw / "camera" / record["camera"]
    lens = read_json(cam / "intrinsics.json")
    frame = record["best_frame"]
    shots = record["exposures"]
    best = next(v for v in shots if v["frame"] == frame)
    time = read_json(cam / "timestamps.json")[frame]
    pose = motion.drive.pose_matrix(read_json(cam / "poses.json")[frame])
    if time != best["time_s"] or imageio_ffmpeg.count_frames_and_secs(
        str(folder / "drive.mp4")
    )[0] != len(shots):
        raise ValueError("Camera clock or encoded frame count changed")
    xy = np.array(best["crop_xy"])
    with np.load(folder / "rays.npz") as packet:
        parts, ranges = packet["part"], packet["range_m"]
    before = np.array(Image.open(folder / "before.png").convert("RGB"))
    after = np.array(Image.open(folder / "after.png").convert("RGB"))
    h, w = parts.shape
    with Image.open(cam / f"{frame:02}.jpg") as photo:
        original = np.asarray(photo.convert("RGB").crop((*xy, *(xy + [w, h]))))
    if not np.array_equal(original, before) or np.any(
        np.any(before != after, axis=2) & (parts == 0)
    ):
        raise ValueError("Photographic changes outside the rendered pavement")
    transform = np.eye(4)
    transform[:3, :3] = record["anchor"]["basis_columns"]
    transform[:3, 3] = record["anchor"]["center_world"]
    hits = motion.tracked_surface(
        surface, transform, pose, lens, xy, parts, ranges, camera_ray_dtype=np.float32
    )
    return dict(
        lens=lens,
        parts=parts,
        offset=xy,
        matrix=transform,
        pose=pose,
        time=time,
        frame=frame,
        **hits,
    )


def check(models, raw, views, out):
    results = {}
    for folder in sorted((views / "front").glob("*/placement.json")):
        key = folder.parent.name
        (glb,) = models.glob(f"*/series/assets/*/{key}/asset.glb")
        digest = sha256(glb)
        mesh = Surface(glb)
        front, back = [
            exposure(views / c / key, raw, mesh, digest) for c in ["front", "back"]
        ]
        if (
            not np.array_equal(front["matrix"], back["matrix"])
            or front["time"] >= back["time"]
        ):
            raise ValueError(
                "World anchor or capture order differs between the two cameras"
            )
        dest = out / key
        dest.mkdir(parents=True, exist_ok=True)
        counts = {}
        for direction, src, dst in [
            ("front_to_back", front, back),
            ("back_to_front", back, front),
        ]:
            mapping = motion.flow(src, dst, mesh, dst["lens"])
            visible = mapping["target_visible"]
            if not visible.any():
                raise ValueError(f"No shared visible surface for {key}")
            filename = dest / (direction + ".npz")
            np.savez_compressed(
                filename,
                source_uv=mapping["source_uv"],
                target_uv=mapping["target_uv"],
                asset_xyz_m=mapping["asset_xyz_m"],
                source_crop_rc=mapping["source_crop_rc"],
                target_visible=visible,
            )
            counts[direction] = dict(
                samples=len(visible),
                visible=int(visible.sum()),
                packet_sha256=sha256(filename),
            )
        results[key] = dict(
            scene=raw.name,
            glb_sha256=digest,
            frame_front=front["frame"],
            frame_back=back["frame"],
            elapsed_s=back["time"] - front["time"],
            front_range_error_mm=1000 * front["error_m"],
            back_range_error_mm=1000 * back["error_m"],
            **counts,
        )
        print(key, counts["front_to_back"]["visible"], flush=True)
    if not results:
        raise ValueError("No assets with paired exposures")
    save_json(out / "summary.json", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "raw", "views", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    check(args.models, args.raw, args.views, args.out)
