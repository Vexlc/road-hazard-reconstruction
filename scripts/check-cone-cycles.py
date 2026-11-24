"""Compare the portable camera adapter to native Cycles object-index and distance passes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from mesh_transfer import pixel_rays, RoadMesh
from road_geometry import save_json
from host_depth import geometry


def cli(*args):
    subprocess.run([sys.executable, str(ROOT / "scripts/transfer-mesh-view.py"), *map(str, args)], check=True)


def packet(folder, axial, *, flat_reference=True):
    meta = json.loads((folder / "camera.json").read_text())
    w = meta["size"]
    f = w / np.deg2rad(meta["fov_deg"])
    rays = pixel_rays(["OPENCV_FISHEYE", w, w, f, f, w / 2, w / 2, 0, 0, 0, 0])
    y, x = np.mgrid[:w, :w] + 0.5
    masked = (x - w / 2) ** 2 + (y - w / 2) ** 2 > (w / 2) ** 2
    ids = np.load(folder / "id.npy")
    distance = np.load(folder / "range.npy")
    known = ids > 0
    raw = np.full(distance.shape, np.inf, "f4")
    raw[known] = distance[known] * rays[..., 2][known] if axial else distance[known]
    pose = np.array(meta["camera_to_world"])
    world = np.array(meta["reference_ground_to_world"])
    p99 = None
    if flat_reference:
        dz = rays @ (pose[:3, :3].T @ world[:3, 2])
        plane = (world[:3, 3] - pose[:3, 3]) @ world[:3, 2] / dz
        close = (ids == 1) & (distance < 5)
        p99 = float(np.percentile(abs(distance[close] - plane[close]), 99))
        if p99 > 0.0002:
            raise ValueError(f"Cycles metric depth registration drifted: {p99}")
    data = dict(
        rgb=np.array(Image.open(folder / "rgb.png").convert("RGB")),
        rays_camera=rays,
        camera_to_world=pose,
        protected=masked,
        receiver_mask=ids == 1,
        occluder_range=np.where(ids == 2, distance, np.inf),
    )
    if (folder / "albedo-srgb.npy").is_file():
        data["road_albedo_srgb"] = np.load(folder / "albedo-srgb.npy", allow_pickle=False)
    data["host_z" if axial else "host_range"] = raw
    return data, world, p99


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/traffic-cones")
    p.add_argument("--views", type=Path, default=ROOT / "outputs/traffic-cones/cycles")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/traffic-cones/cycles-check")
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    asset = a.assets / "C236-C237"
    paths = []
    stats = {}
    source = {}
    for name, axial in [("center", False), ("left-edge", True), ("return-view", False)]:
        cam, world, p99 = packet(a.views / name / "host", axial)
        path = a.out / (name + ".npz")
        np.savez_compressed(path, **cam)
        paths.append(path)
        source[name] = cam
        stats[name] = dict(depth_convention="signed axial Z" if axial else "radial", road_range_p99_m=p99)
    cli("road-points", a.out / "road.npy", *paths[:2], "--stride", 2)
    light = dict(
        road_srgb=np.median(source["center"]["rgb"][source["center"]["receiver_mask"]], axis=0).tolist(),
        sun_world=Rotation.from_euler("xyz", [0.44, -0.63, -0.3]).apply([0, 0, 1]).tolist(),
        direct_fraction=0.48,
    )
    save_json(a.out / "light.json", light)
    anchorfile = a.out / "anchor.json"
    cli(
        "fit",
        a.out / "road.npy",
        anchorfile,
        "--seed",
        *(world[:3, 3] + [0, 0, 0.025]),
        "--up",
        0,
        0,
        1,
        "--heading",
        *world[:3, 0],
        "--radius",
        1.65,
        "--asset",
        asset,
        "--lighting",
        a.out / "light.json",
    )
    anchor = json.loads(anchorfile.read_text())
    mesh = RoadMesh(asset)
    for path, (name, cam) in zip(paths, source.items()):
        dest = a.out / name
        cli("render", asset, anchorfile, path, dest)
        ref = a.views / name / "native"
        ids = np.load(ref / "id.npy")
        native = ids >= 3
        distance = np.load(ref / "range.npy")
        measured = np.load(dest / "range.npy")
        hit = np.isfinite(measured)
        native_rgb = np.array(Image.open(ref / "rgb.png").convert("RGB"))
        after = np.array(Image.open(dest / "after.png"))
        overlap = native & hit
        iou = float(overlap.sum() / (native | hit).sum())
        res = abs(measured[overlap] - distance[overlap])
        core = cv2.erode(overlap.astype("u1"), np.ones((3, 3), "u1")).astype(bool)
        rgb_mae = float(abs(after[core].astype(float) - native_rgb[core]).mean())
        saved = mesh.dielectrics
        mesh.dielectrics = {}
        rays, near, protected, pose = geometry(cam)
        try:
            diffuse, part, _, _ = mesh.render(
                cam["rgb"],
                pose,
                rays[0],
                anchor,
                near[0],
                protected[0],
                receiver_mask=cam["receiver_mask"],
                occluder_range=cam["occluder_range"],
                with_shadow=True,
            )
        finally:
            mesh.dielectrics = saved
        if not np.array_equal(part > 0, hit):
            raise ValueError("Material response changed the silhouette")
        diffuse_mae = float(abs(diffuse[core].astype(float) - native_rgb[core]).mean())
        if not rgb_mae < diffuse_mae:
            raise ValueError("Dielectric material failed the Cycles appearance comparison")
        bare = (
            mesh.render(
                cam["rgb"],
                cam["camera_to_world"],
                cam["rays_camera"],
                anchor,
                np.full(hit.shape, np.inf, "f4"),
                cam["protected"],
            )[1]
            > 0
        )
        occluded = (np.load(a.views / name / "host/id.npy") == 2) & bare & ~native
        original_ids = np.load(a.views / name / "host/id.npy")
        original_dist = np.load(a.views / name / "host/range.npy")
        ahead = overlap & (original_ids == 2) & (distance + 0.01 < original_dist)
        if name == "return-view" and ahead.sum() < 30:
            raise ValueError("Held-out reverse view did not test reversed occlusion")
        behind = overlap & (cam["rays_camera"][..., 2] < 0)
        if not (
            iou > 0.98 and np.percentile(res, 95) < 0.005 and (name == "return-view" or occluded.sum() > 30)
        ):
            raise ValueError(
                f"{name}: IoU={iou:.5f}, 95% radial error={np.percentile(res, 95):.6f}, physically occluded={occluded.sum()}"
            )
        if np.any(hit & occluded) or np.any(np.load(dest / "shadow.npy")[occluded]):
            raise ValueError("The renderer changed a foreground occluder")
        if name == "left-edge" and behind.sum() < 200:
            raise ValueError("No rear-hemisphere cone support")
        stats[name].update(
            visible_native=int(native.sum()),
            visible_portable=int(hit.sum()),
            iou=iou,
            range_p95_m=float(np.percentile(res, 95)),
            behind_optical_plane=int(behind.sum()),
            foreground_occludes=int(occluded.sum()),
            asset_in_front_of_host=int(ahead.sum()),
            core_rgb_mae=rgb_mae,
            diffuse_only_mae=diffuse_mae,
        )
        print(name, stats[name], flush=True)
    save_json(
        a.out / "check.json",
        dict(
            views=stats,
            ground_center_error_m=float(np.linalg.norm(world[:3, 3] - anchor["center_world"])),
            ground_normal_error_deg=float(
                np.degrees(np.arccos(np.clip(world[:3, 2] @ np.array(anchor["basis_columns"])[:, 2], -1, 1)))
            ),
        ),
    )


if __name__ == "__main__":
    main()
