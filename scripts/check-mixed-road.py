"""Audit the nine built meshes, merged native hosts, physical lens replays and renderer-neutral packets."""

import argparse
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from mesh_transfer import fit_road, pixel_rays, resolve_samples, RoadMesh  # noqa: E402
from road_geometry import read_json, save_json, sha256, Surface  # noqa: E402


def check_frames(base, cases):
    index = read_json(base / "index.json")
    if set(index) != set(cases):
        raise ValueError(f"Incomplete replay in {base}")
    counts = {}
    for key, asset in index.items():
        views = asset["views"]
        if len(views) != 2 or asset["source_sha256"] != cases[key]["asset.glb"]:
            raise ValueError(f"Incompatible geometry or cameras: {key}")
        counts[key] = {}
        for cam, row in views.items():
            folder = base / key / cam
            before, after, part = [
                np.asarray(Image.open(folder / (name + ".png"))) for name in ("before", "after", "part")
            ]
            distance = np.load(folder / "range.npy", allow_pickle=False)
            hit = part > 0
            if (
                hit.sum() != row["visible"]
                or not np.array_equal(hit, np.isfinite(distance))
                or not np.array_equal(before[~hit], after[~hit])
                or np.any(distance[hit] <= 0)
                or not np.any(before[hit] != after[hit])
                or row["protected_edits"] != 0
            ):
                raise ValueError(f"Invalid RGB, part or depth for {key}/{cam}")
            counts[key][cam] = {
                "frame": row["frame"],
                "visible_pixels": int(hit.sum()),
                "accepted_frames": len(row["captured_frames"]),
                "ego_edits": row["protected_edits"],
            }
    with imageio.get_reader(base / "driving.mp4") as reader:
        video_frames = sum(1 for _ in reader.iter_data())
        fps = reader.get_meta_data()["fps"]
    return dict(
        models=counts,
        road_fit=read_json(base / "anchor.json"),
        encoded_frames=video_frames,
        seconds=video_frames / fps,
    )


def packet_controls(base, mesh, asset_id="M224"):
    anchor = read_json(base / "anchor.json")
    controls = {}
    for packet in sorted((base / asset_id).glob("*/view.npz")):
        with np.load(packet, allow_pickle=False) as view:
            rgb, pose, host, protection, rays = (
                view[k] for k in ("rgb", "camera_to_world", "host_range", "protected", "rays_camera")
            )
        out, part, dist = resolve_samples(
            [mesh.render(rgb, pose, r, anchor, host, p) for r, p in zip(rays, protection)]
        )
        ref = packet.parent
        if (
            not np.array_equal(out, np.asarray(Image.open(ref / "after.png")))
            or not np.array_equal(part, np.asarray(Image.open(ref / "part.png")))
            or not np.allclose(dist, np.load(ref / "range.npy"), rtol=0, atol=0, equal_nan=True)
        ):
            raise ValueError(f"Per-pixel ray packet changed in {ref}")
        for nearer, mask in [(host, np.ones_like(protection[0])), (np.full_like(host, 0.05), protection[0])]:
            hidden, part, dist = mesh.render(rgb, pose, rays[0], anchor, nearer, mask)
            if not np.array_equal(hidden, rgb) or part.any() or np.isfinite(dist).any():
                raise ValueError("Foreground or nearer host geometry was overwritten")
        controls[ref.name] = (
            "Exact RGB, part and metric range. No edits with foreground or nearer-depth controls."
        )
    if len(controls) != 2:
        raise ValueError(f"Render with --export-packet {asset_id} to check both physical cameras")
    return controls


def numeric_controls():
    cal = ["OPENCV_FISHEYE", 400, 300, 183.0, 181.0, 194.0, 148.0, -0.04, 0.005, 0.0, 0.0]
    rays = pixel_rays(cal)[::11, ::11].astype("f8")
    k = np.array([[cal[3], 0, cal[5]], [0, cal[4], cal[6]], [0, 0, 1.0]])
    uv = cv2.fisheye.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, np.array(cal[7:]))[0]
    y, x = np.mgrid[:300:11, :400:11]
    reproj = float(np.linalg.norm(uv.reshape(*x.shape, 2) - np.dstack([x + 0.5, y + 0.5]), axis=2).max())
    if reproj > 0.0002:
        raise ValueError("Fisheye radial inversion failed")
    rotation = Rotation.from_rotvec([0.52, -0.83, 0.24]).as_matrix()
    x, y = np.mgrid[-2:2:40j, -2:2:40j]
    samples = np.c_[x.ravel(), y.ravel(), (0.018 * x - 0.031 * y + 0.04).ravel()]
    shifted = np.array([40.0, -121.0, 8.0])
    a = fit_road(samples, np.zeros(3), np.array([1, 0, 0]), up_world=[0, 0, 1], radius_m=1.9)
    b = fit_road(samples @ rotation.T + shifted, shifted, rotation[:, 0], up_world=rotation[:, 2], radius_m=1.9)
    if not np.allclose(rotation @ a["center_world"] + shifted, b["center_world"], atol=1e-6):
        raise ValueError("Changing the world axes moved the contact point")
    try:
        fit_road(samples[samples[:, 0] > 0], np.zeros(3), np.array([1, 0, 0]), up_world=[0, 0, 1], radius_m=1.9)
    except ValueError:
        return {
            "fisheye_max_pixel_roundtrip": reproj,
            "rotated_gravity_origin": "matched",
            "one_sided_support": "rejected",
        }
    raise ValueError("Accepted an unsupported anchor")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "runs/mixed-road")
    ap.add_argument("--views", type=Path, default=ROOT / "outputs/mixed-road")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    keys = ["P220", "P221", "P222", "P223", "A224", "A225", "A226", "A227", "M224"]
    assets = {}
    for key in keys:
        (folder,) = args.run.glob(f"*/series/assets/*/{key}")
        hashes = {name: sha256(folder / name) for name in ["asset.glb", "asset.ply", "surface.npz"]}
        assets[key] = hashes
        bundle = folder.parents[2]
        (scene,) = [
            s for s in bundle.glob("scenes/*/scene.json") if read_json(s)["assets"][0].split("_")[0] == key
        ]
        info = read_json(scene)
        (pl,) = info["placements"]
        if (
            not info["actors_sky_unchanged"]
            or not info["checkpoint_reloaded"]
            or pl["reload_max_pixel_error"] != 0
            or pl["size_multiplier"] != 1
            or pl["asset_ply_sha256"] != hashes["asset.ply"]
        ):
            raise ValueError("Native checkpoint did not preserve asset or host: " + key)
        hashes.update(
            pandaset=info["id"], reload_pixel_error=pl["reload_max_pixel_error"], local_road_rms_m=pl["rms_m"]
        )
        if key.startswith("P"):
            metrics = read_json(bundle.parent / "evaluation/measurements.json")[key]
            if (
                metrics["meshed_fraction"] < 0.95
                or metrics["withheld_mesh"]["p95_mm"] >= metrics["withheld_road_plane"]["p95_mm"]
            ):
                raise ValueError("Depth-block prediction failed: " + key)
            hashes["withheld_depth"] = metrics["withheld_mesh"]
        if key in ("M224", "A224"):
            surface = Surface(folder / "asset.glb")
            distance, part, _ = surface.cast(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]]))
            if (
                key == "M224"
                and (part[0] != 7 or abs(distance[0] - 1.82) > 0.005)
                or key == "A224"
                and (part[0] != 9 or abs(distance[0] - 1) > 0.01)
            ):
                raise ValueError("Central ray did not hit the intended component: " + key)
            hashes["centre_part"] = int(part[0])
    result = {"assets": assets, "controls": numeric_controls(), "hosts": {}}
    mesh = RoadMesh(next(args.run.glob("*/series/assets/missing/M224")))
    for host in ["a2d2-frontleft", "kitti360-0000"]:
        base = args.views / host
        result["hosts"][host] = dict(**check_frames(base, assets), packets=packet_controls(base, mesh))
    save_json(args.out, result)
    print("Verified 9 native placements, 36 selected photographs, four camera packets and two decoded videos.")


if __name__ == "__main__":
    main()
