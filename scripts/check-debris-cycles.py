"""Check pair silhouettes, through-openings, foreground depth and tangent normals against native Cycles."""

import argparse
import json
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

reference = import_module("check-cone-cycles")
ROOT = reference.ROOT
from mesh_transfer import RoadMesh
from road_geometry import save_json


def angular(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return np.rad2deg(np.arccos(np.clip((a * b).sum(1), -1, 1)))


def predicted_normals(mesh, packet, anchor):
    pose = packet["camera_to_world"]
    basis = np.array(anchor["basis_columns"])
    directions = packet["rays_camera"] @ pose[:3, :3].T @ basis
    origin = (pose[:3, 3] - anchor["center_world"]) @ basis
    rays = np.float32(np.concatenate([np.broadcast_to(origin, directions.shape), directions], -1))
    hits = mesh.scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
    labels = hits["geometry_ids"].numpy()
    indices = hits["primitive_ids"].numpy()
    weights = hits["primitive_uvs"].numpy()
    distances = hits["t_hit"].numpy()
    from pavement_texture import ray_derivatives

    derivatives = ray_derivatives(directions)
    base = np.zeros_like(rays[..., :3])
    shaded = np.zeros_like(base)
    for gid, (_, tri, uv, _) in mesh.surfaces.items():
        valid = labels == gid
        if not valid.any():
            continue
        idx = tri[indices[valid]]
        w = weights[valid]
        bary = np.c_[1 - w.sum(1), w]
        footprint = (directions[valid], distances[valid], [d[valid] for d in derivatives], 1.0, None)
        n, s = mesh.surface_normal(gid, idx, bary, (uv[idx] * bary[:, :, None]).sum(1), footprint)
        base[valid], shaded[valid] = n @ basis.T, s @ basis.T
    return base, shaded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=ROOT / "outputs/road-debris/cycles")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/road-debris/cycles-check")
    p.add_argument("--assets", type=Path, default=ROOT / "runs/road-debris")
    p.add_argument("--detail", action="store_true", help="Also inspect the near side of the second object")
    p.add_argument("--pairs", nargs="+", default=["R242-R243", "R244-R245"])
    args = p.parse_args()
    stats = {}
    from road_geometry import glb_triangles
    for pair in args.pairs:
        out = args.out / pair
        out.mkdir(parents=True, exist_ok=True)
        mesh = RoadMesh(args.assets / pair)
        packets = {}
        for name in (["front", "edge", "rear", "detail"] if args.detail else ["front", "edge", "rear"]):
            packets[name], world, _ = reference.packet(args.views / pair / name / "host", name == "edge")
            np.savez_compressed(out / (name + ".npz"), **packets[name])
        cli = reference.cli
        survey_views = [out / "front.npz", out / "edge.npz"]
        if args.detail:
            survey_views.append(out / "rear.npz")  # Observe road hidden behind the raised marker.
        cli("road-points", out / "road.npy", *survey_views, "--stride", 2)
        color = packets["front"]["rgb"][packets["front"]["receiver_mask"]]
        save_json(
            out / "light.json",
            {
                "road_srgb": np.median(color, axis=0).tolist(),
                "direct_fraction": 0.48,
                "sun_world": Rotation.from_euler("xyz", [0.44, -0.63, -0.3]).apply([0, 0, 1]).tolist(),
            },
        )
        cli(
            "fit",
            out / "road.npy",
            out / "anchor.json",
            "--seed",
            *world[:3, 3],
            "--heading",
            *world[:3, 0],
            "--up",
            *world[:3, 2],
            "--radius",
            1.5,
            "--asset",
            args.assets / pair,
            "--lighting",
            out / "light.json",
        )
        anchor = json.loads((out / "anchor.json").read_text())
        stats[pair] = {}
        for name, packet in packets.items():
            dest = out / name
            cli("render", args.assets / pair, out / "anchor.json", out / (name + ".npz"), dest)
            native = args.views / pair / name / "native"
            ids = np.load(native / "id.npy")
            dist = np.load(native / "range.npy")
            true_n = np.load(native / "shading-world.npy")
            part = np.array(Image.open(dest / "part.png"))
            depth = np.load(dest / "range.npy")
            base, shaded = predicted_normals(mesh, packet, anchor)
            row = {}
            for i, key in enumerate(pair.split("-")):
                codes = {code for code, _, _ in glb_triangles(args.assets / key / "asset.glb")}
                hit = np.isin(part, list(codes))
                expected = ids == 3 + i
                overlap = hit & expected
                if not overlap.any():
                    if hit.any() or expected.any():
                        raise ValueError("Missing a visible asset")
                    continue
                iou = overlap.sum() / (hit | expected).sum()
                err = np.percentile(abs(dist[overlap] - depth[overlap]), 95)
                core = cv2.erode(overlap.astype("u1"), np.ones((3, 3), "u1")).astype(bool)
                core &= np.linalg.norm(true_n, axis=2) > 0.85
                simple = angular(base[core], true_n[core])
                detailed = angular(shaded[core], true_n[core])
                # Road pixels entirely enclosed by the independent object silhouette test through-openings.
                free = (~expected).astype("u1")
                _, components = cv2.connectedComponents(free, connectivity=4)
                border = np.unique(
                    np.r_[components[0], components[-1], components[:, 0], components[:, -1], 0]
                )
                holes = (~np.isin(components, border)) & (ids == 1)
                row[key] = {
                    "enclosed_road_pixels": int(holes.sum()),
                    "enclosed_road_preserved": int((holes & (part == 0)).sum()),
                    "pixels": int(overlap.sum()),
                    "iou": float(iou),
                    "p95_range_error_m": float(err),
                    "core": int(core.sum()),
                    "smooth_normal_mae_deg": float(simple.mean()),
                    "mapped_normal_mae_deg": float(detailed.mean()),
                    "behind_optical_plane": int((overlap & (packet["rays_camera"][..., 2] < 0)).sum()),
                }
                print(pair, name, key, row[key], flush=True)
                if iou < 0.98 or err > 0.003 or core.sum() < 12 or np.any(hit & (ids == 2)):
                    raise ValueError("Independent object labels, visibility or metric depth disagree")
            stats[pair][name] = row
    rows = [item for views in stats.values() for parts in views.values() for item in parts.values()]
    count = sum(r["core"] for r in rows)
    errors = {
        key: sum(r[key] * r["core"] for r in rows) / count
        for key in ["smooth_normal_mae_deg", "mapped_normal_mae_deg"]
    }
    holes = sum(r["enclosed_road_pixels"] for r in rows)
    preserved = sum(r["enclosed_road_preserved"] for r in rows)
    if (
        errors["mapped_normal_mae_deg"] >= errors["smooth_normal_mae_deg"]
        or holes < 40
        or preserved / holes < 0.98
    ):
        raise ValueError("Shading normals or through-openings failed the independent Cycles comparison")
    if min(sum(r["behind_optical_plane"] for r in views["edge"].values()) for views in stats.values()) < 100:
        raise ValueError("Missing the outer fisheye field in the edge-view comparison")
    stats["summary"] = dict(
        **errors, core_pixels=count, enclosed_road_pixels=holes, enclosed_road_preserved=preserved
    )
    save_json(args.out / "check.json", stats)


if __name__ == "__main__":
    main()
