"""Check six-face depth, occlusion and transferred geometry against independent OpenGL cube sampling."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from cube_camera import convert
from road_geometry import save_json


def cli(*args):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/transfer-mesh-view.py"), *map(str, args)],
        check=True,
    )


def check_frame(packet, src, rendered):
    with (
        np.load(src / "reference.npz") as truth,
        np.load(src / "lens.npz") as lens,
        np.load(src / "faces.npz") as cube,
    ):
        valid = ~packet["protected"][0]
        observed = valid & (truth["host_id"] > 0)
        host = packet["host_range"][0]
        foreground = packet["occluder_range"][0]
        part = np.asarray(Image.open(rendered / "part.png"))
        instance = np.asarray(Image.open(rendered / "instance.png"))
        distance = np.load(rendered / "range.npy")
        target = np.where(truth["native_id"] >= 32, truth["native_id"] // 32, 0)
        if (
            np.any(np.isfinite(host[valid]) != (truth["host_id"][valid] > 0))
            or np.any(
                packet["receiver_mask"][0][valid] != (truth["host_id"][valid] == 1)
            )
            or np.any(np.isfinite(foreground[valid]) != (truth["host_id"][valid] == 2))
        ):
            raise ValueError(
                "Host labels or coverage differ from hardware cube sampling"
            )
        err = np.abs(host[observed] - truth["host_range"][observed])
        overlap = valid & (target == instance) & (target > 0)
        native_range = np.abs(distance[overlap] - truth["native_range"][overlap])
        hidden = valid & (truth["clear_id"] >= 32) & (truth["native_id"] == 2)
        cube_bearing = lens["rays_camera"] @ (
            lens["camera_to_world"][:3, :3].T @ cube["camera_to_world"][:3, :3]
        )
        forward = cube_bearing @ cube["face_to_cube"][:, :, 2].T
        nearest = np.sort(forward, axis=-1)[..., -2:]
        seam = (nearest[..., 1] - nearest[..., 0]) < 4 / cube["rgb"].shape[1]
        behind = lens["rays_camera"][..., 2] < 0
        scores = [
            (valid & (target == i) & (instance == i)).sum()
            / (valid & ((target == i) | (instance == i))).sum()
            for i in (1, 2)
        ]
        row = dict(
            host_samples=int(observed.sum()),
            host_p99_m=float(np.percentile(err, 99)),
            host_max_m=float(err.max()),
            asset_p95_m=float(np.percentile(native_range, 95)),
            instance_iou=list(map(float, scores)),
            sampled_asset_pixels=int(overlap.sum()),
            at_face_seam=int((overlap & seam).sum()),
            behind_lens_plane=int((overlap & behind).sum()),
            blocked_asset_pixels=int(hidden.sum()),
            edits_on_blocked_pixels=int((hidden & (part > 0)).sum()),
        )
        automatic = convert(cube, {k: lens[k] for k in lens.files if k != "cube_texel"})
        delta = np.linalg.norm(
            automatic["rays_camera"][0] - packet["rays_camera"][0], axis=-1
        )
        changed = (delta > 1e-6) & valid
        if (
            np.any(delta[valid] > 2.01 / cube["rgb"].shape[1])
            or changed.sum() > 0.001 * valid.sum()
        ):
            raise ValueError("Nearest-texel fallback deviates from native sampling")
        stable = valid & ~changed
        if not np.array_equal(automatic["host_range"][0][stable], host[stable]):
            raise ValueError("Depth changed at identical sample addresses")
        row["automatic_neighbouring_texels"] = int(changed.sum())
        row["automatic_checked_pixels"] = int(valid.sum())
        print(src, row, flush=True)
        if (
            row["host_max_m"] > 0.003
            or row["asset_p95_m"] > 0.004
            or min(scores) < 0.96
            or not np.isfinite(scores).all()
            or row["blocked_asset_pixels"] < 8
            or row["edits_on_blocked_pixels"]
            or (instance[~valid] > 0).any()
            or np.any(
                np.asarray(Image.open(rendered / "after.png"))[hidden]
                != packet["rgb"][hidden]
            )
        ):
            raise ValueError("Native cubemap comparison failed")
        return row


def check_contract(src):
    with np.load(src / "faces.npz") as data, np.load(src / "lens.npz") as camera:
        faces = dict(data)
        # A sparse synthetic lens exercises all six signed axes and both sides of seams.
        axes = np.vstack([np.eye(3), -np.eye(3), [[1, 1, 0.99], [1, 1.01, 0.99]]])
        directions = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        lens = dict(
            camera_to_world=faces["camera_to_world"],
            rays_camera=directions.reshape(1, -1, 3),
            protected=np.zeros((1, len(axes)), bool),
        )
        selected = convert(faces, lens)
        delta = np.linalg.norm(
            selected["rays_camera"][0] - lens["rays_camera"], axis=-1
        )
        if delta.max() > 2 / faces["rgb"].shape[1]:
            raise ValueError("Nearest cube texel is outside its angular footprint")
        second = dict(
            lens,
            rays_camera=np.stack([lens["rays_camera"], -lens["rays_camera"]]),
            sample_weights=np.array([1.0, 3.0]),
        )
        sampled = convert(faces, second)
        if sampled["sample_rgb"].shape != (2, 1, len(axes), 3):
            raise ValueError("Lost multisample color")
        legacy = dict(faces)
        legacy["face_to_cube"] = legacy["face_to_cube"].copy()
        legacy["face_to_cube"][0] = legacy["face_to_cube"][1]
        shifted = dict(lens, camera_to_world=lens["camera_to_world"].copy())
        shifted["camera_to_world"][0, 3] += 0.12
        for f, l in [(legacy, lens), (faces, shifted), (dict(faces, near_m=0.0), lens)]:
            try:
                convert(f, l)
            except ValueError:
                continue
            raise ValueError("Invalid cubemap registered")


def check_depth_conventions(src):
    with np.load(src / "faces.npz") as recorded:
        basis = recorded["face_to_cube"].copy()
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for i in range(6):
        basis[i] = basis[i] @ np.linalg.matrix_power(turn, i % 4)
    yy, xx = np.indices((3, 3))
    axial = 1 + np.arange(6)[:, None, None] + (xx + 3 * yy) / 10.0
    colors = np.round(axial[..., None] * [23, 11, 7]).astype("u1")
    frames = np.array([5, 3, 1, 4, 2, 0])
    cols = np.array([0, 1, 2, 0, 2, 1])
    rows = np.array([2, 0, 1, 1, 2, 0])
    local = np.c_[(cols + 0.5) * 2 / 3 - 1, (rows + 0.5) * 2 / 3 - 1, np.ones(6)]
    scale = np.linalg.norm(local, axis=1)
    rays = np.einsum("nij,nj->ni", basis[frames], local / scale[:, None])
    lens = dict(
        camera_to_world=np.eye(4),
        rays_camera=rays[None],
        protected=np.zeros((1, 6), bool),
    )
    for far in (8.0, np.inf):
        for reverse in (False, True):
            near = 0.25
            d = (1 / near - 1 / axial) / (1 / near - 1 / far)
            faces = dict(
                rgb=colors,
                depth_valid=np.ones_like(d, bool),
                device_depth=1 - d if reverse else d,
                reversed_z=reverse,
                near_m=near,
                far_m=far,
                face_to_cube=basis,
                camera_to_world=np.eye(4),
            )
            packet = convert(faces, lens)
            np.testing.assert_allclose(
                packet["host_range"][0, 0],
                (axial[frames, rows, cols] * scale),
                atol=1e-9,
            )
            np.testing.assert_array_equal(packet["rgb"][0], colors[frames, rows, cols])
    wrong = np.stack([frames, 2 - cols, rows], axis=-1)[None]
    try:
        convert(faces, dict(lens, cube_texel=wrong))
    except ValueError:
        return
    raise ValueError("Unrelated texel address was accepted")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("captures", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("layouts", type=Path, nargs="+")
    args = p.parse_args()
    frame = json.loads((args.captures / "frame.json").read_text())
    summary = {}
    for recipe in args.layouts:
        pair = json.loads(recipe.read_text())["id"]
        base = args.out / pair
        base.mkdir(parents=True, exist_ok=True)
        rows = {}
        for shot in ("side", "rear"):
            src = args.captures / pair / shot
            cli(
                "from-cube", src / "faces.npz", src / "lens.npz", base / (shot + ".npz")
            )
        check_contract(args.captures / pair / "side")
        check_depth_conventions(args.captures / pair / "side")
        cli(
            "road-points",
            base / "road.npy",
            base / "side.npz",
            base / "rear.npz",
            "--stride",
            3,
        )
        cli(
            "fit",
            base / "road.npy",
            base / "anchor.json",
            "--seed",
            *frame["seed"],
            "--heading",
            *frame["heading"],
            "--up",
            *frame["up"],
            "--radius",
            2,
            "--asset",
            recipe,
            "--asset-root",
            args.models,
        )
        for shot in ("side", "rear"):
            src = args.captures / pair / shot
            cli(
                "render",
                recipe,
                base / "anchor.json",
                base / (shot + ".npz"),
                base / shot,
                "--asset-root",
                args.models,
            )
            with np.load(base / (shot + ".npz")) as packet:
                rows[shot] = check_frame(packet, src, base / shot)
        if (
            sum(v["at_face_seam"] for v in rows.values()) < 35
            or rows["side"]["behind_lens_plane"] < 150
        ):
            raise ValueError(
                "The native comparison must cover cube seams and the rear hemisphere"
            )
        summary[pair] = rows
    save_json(args.out / "check.json", summary)


if __name__ == "__main__":
    main()
