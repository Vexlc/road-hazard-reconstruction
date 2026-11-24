"""Recover sunlight direction from parallax of a real roadside post and its shadow."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image

drive = import_module("render-mixed-road")
from prop_lighting import linear
from road_geometry import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]


def intersection(origins, directions):
    d = np.asarray(directions)
    project = np.eye(3) - d[:, :, None] * d[:, None, :]
    return np.linalg.solve(project.sum(0), np.einsum("nij,nj->i", project, origins))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--observations",
        type=Path,
        default=ROOT / "gen2hole/examples/roadside-props/kitti-sun-observations.json",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    spec = read_json(args.observations)
    rig = ROOT / "work/streetcrafter-rig" / spec["scene"]
    data, cameras = drive.load_rig(rig)
    sensor = spec["sensor"]
    model = drive.view_model(sensor, data)
    raw = (
        ROOT
        / "work/data"
        / ("kitti360-0000-0200" if spec["scene"] == "kitti360-0000" else spec["scene"])
        / sensor
    )
    origins = []
    rays = {k: [] for k in ["top", "base", "tip"]}
    contrasts = []
    poses = []
    for row in spec["frames"]:
        g = data["frames"].index(row["frame"])
        cam = data["sensors"].index(sensor)
        pose = np.linalg.inv(cameras[f"{g:03}{cam}0.png"])
        poses.append(pose)
        origins.append(pose[:3, 3])
        for name, seq in rays.items():
            r, valid = model[2](*row[name], 1, 1)
            if not valid.item():
                raise ValueError("Reference landmark outside the calibrated lens")
            seq.append(pose[:3, :3] @ r[0, 0])
        with Image.open(raw / f"{row['frame']:010}.png") as img:

            def luminance(xy):
                x, y = xy
                patch = np.array(img.convert("RGB").crop((x - 5, y - 5, x + 6, y + 6)))
                return float(np.median(linear(patch) @ [0.2126, 0.7152, 0.0722]))

            contrasts.append(1 - luminance(row["shade"]) / max(luminance(row["lit"]), 0.001))
    solved = {k: intersection(origins, v) for k, v in rays.items()}
    sun = solved["top"] - solved["tip"]
    sun /= np.linalg.norm(sun)
    errors = {}
    for k, point in solved.items():
        errors[k] = [
            float(np.linalg.norm(model[1]((point - pose[:3, 3]) @ pose[:3, :3]) - row[k]))
            for row, pose in zip(spec["frames"], poses)
        ]
    height = np.linalg.norm(solved["top"] - solved["base"])
    fraction = float(np.median(contrasts))
    if (
        max(max(v) for v in errors.values()) > 8
        or not 0.25 < height < 2
        or not 0.22 < sun[2]
        or not 0 < fraction < 1
        or np.ptp(contrasts) > 0.25
    ):
        raise ValueError(
            f"Inconsistent landmark fit: {errors}, height {height}, direct {contrasts}, sun {sun}"
        )
    result = dict(
        sun_world=sun.tolist(),
        direct_fraction=min(0.9, fraction),
        source_frames=[r["frame"] for r in spec["frames"]],
        post_height_m=float(height),
        max_reprojection_px=max(max(v) for v in errors.values()),
        reference_contrasts=contrasts,
        points_world={k: v.tolist() for k, v in solved.items()},
    )
    save_json(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
