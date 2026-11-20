"""Convert assets to anisotropic Gaussian PLY, and render each file with gsplat."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from gsplat.rendering import rasterization
import torch

from gaussian_assets import make_gaussians, save_ply, tensor
from sample_hazards import ray_scene, sun_visibility, unit


def sample_mesh(mesh_path, surface_file, metadata):
    with np.load(mesh_path) as mesh, np.load(surface_file) as surface:
        vertex, faces = mesh["xyz"], mesh["faces"]
        triangles = vertex[faces]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(normal, axis=1)*.50
        assert areas.sum() > .1
        geometry = ray_scene({"vertices": vertex, "faces": faces})
        area = float(areas.sum(dtype="f8"))
        importance = np.ones(len(faces), dtype="f8")
        detail = metadata.get("gaussian_detail_weights", {})
        if detail:
            if "face_part" not in mesh or "part_names" not in mesh:
                raise ValueError("Detail-aware sampling needs explicit mesh part IDs")
            weights = np.array([detail.get(str(name), 1.) for name in mesh["part_names"]])
            if not (np.isfinite(weights).all() and np.all((weights >= .25) & (weights <= 16))):
                raise ValueError("Invalid part sampling weights")
            importance = weights[mesh["face_part"]]
        accum = np.cumsum(areas*importance, dtype="f8")
        weighted_area = accum[-1]
        density = metadata.get("gaussian_sampling_density_m2", 26000)
        minimum = metadata.get("gaussian_sampling_min", 42000)
        maximum = metadata.get("gaussian_sampling_max", 122000)
        if not (0 < density and 1000 <= minimum <= maximum <= 2000000):
            raise ValueError("Invalid Gaussian sampling budget")
        count = int(np.clip(area * density, minimum, maximum))
        metadata["gaussian_sampling"] = {"surface_area_m2": float(area),
                                         "requested_points": count,
                                         "spacing_before_alpha_m": float(np.sqrt(area/count)),
                                         "density_per_m2": density, "maximum_points": maximum,
                                         "part_weights": detail}
        rng = np.random.default_rng(metadata["seed"] + 91227)
        triangles_seen = np.searchsorted(accum, (np.arange(count)+rng.uniform(0, 1, count))*weighted_area/count)
        spacing = np.sqrt(weighted_area/(count*importance[triangles_seen]))
        bary_u = np.sqrt(rng.random(count))
        bary_v = rng.random(count)
        weights = np.stack([1-bary_u, bary_u*(1-bary_v), bary_u*bary_v], axis=1).astype("f4")
        xyz = np.einsum("nvc,nv->nc", triangles[triangles_seen], weights)
        uv = np.einsum("nvc,nv->nc", mesh["uv"][faces[triangles_seen]], weights)
        normal = unit(normal[triangles_seen])
        tangent = unit(np.cross(np.broadcast_to([0, 0, 1.], normal.shape), normal))
        vertical = np.abs(normal[:, 2]) > .80
        tangent[vertical] = unit(np.cross(np.broadcast_to([0, 1., 0], normal[vertical].shape), normal[vertical]))
        bitangent = np.cross(normal, tangent)
        materials = mesh["material"][triangles_seen]
        gloss = mesh["parameters"][materials]
        road = mesh["road"][triangles_seen]
        color = np.ones((count, 4), dtype="f4")
        for material_id in np.unique(materials):
            ids = materials == material_id
            texture = mesh[f"texture_{material_id}"]
            texture_yx = (1-uv[ids, 1])*(texture.shape[0]-1), uv[ids, 0]*(texture.shape[1]-1)
            pixel = np.stack([map_coordinates(texture[:, :, chan], texture_yx, order=1, mode="nearest")
                              for chan in range(3)], axis=1)
            color[ids, :3] = pixel/255
        extent = float(surface["width"])*.5
        falloff = np.clip((extent-np.linalg.norm(xyz[:, :2], axis=-1))/.17, 0, 1)
        alpha = np.where(road > .5, falloff**2*(3-2*falloff), 1)
        keep = alpha > .01
        xyz, normal, tangent, bitangent, gloss, road, color, alpha = [
            value[keep] for value in (xyz, normal, tangent, bitangent, gloss, road, color, alpha)
        ]
        if metadata["category"] == "pothole":
            depth_strength = min(.04, metadata["depth_design_m"]*.36)
            # Include extra alignment margin when transplanting the road apron.
            road = road*np.clip(1+xyz[:, 2]/max(.005, depth_strength), 0, 1)**2
        ao = ray_ambient(geometry, xyz, normal, tangent, bitangent, rng)
        sun = sun_visibility(geometry, xyz, normal, [-.35, -.25, 1])
        spacing = spacing[keep]
        scales = np.c_[spacing*.83, spacing*.83, np.full(len(xyz), .00045)]
        return dict(name=metadata["id"], local_xyz=xyz, normals=normal,
                    rotation_matrices=np.stack([tangent, bitangent, normal], axis=-1),
                    scales=scales, alpha=alpha, color=color, normal=(normal+1)/2,
                    arm=np.c_[np.ones(len(xyz)), gloss], ao=ao[:, None], sun_visibility=sun,
                    road_mask=road, geometric_ao=True)


def ray_ambient(geometry, xyz, normal, tangent, bitangent, rng):
    import open3d as o3d
    rays_count = 12
    idx = np.arange(rays_count)
    sin_phi = np.sqrt((idx+.5)/rays_count)
    cos_phi = np.sqrt(1-sin_phi**2)
    base_theta = idx * np.pi*(3-np.sqrt(5))
    result = np.empty(len(xyz), "f4")
    for first in range(0, len(xyz), 16000):
        ids = slice(first, first+16000)
        xyzs = xyz[ids]
        theta = base_theta+rng.uniform(0, 2*np.pi, (len(xyzs), 1))
        d = (tangent[ids, None, :]*(np.cos(theta)*sin_phi)[..., None]
             + bitangent[ids, None, :]*(np.sin(theta)*sin_phi)[..., None]
             + normal[ids, None, :]*cos_phi[None, :, None])
        origin = np.broadcast_to((xyzs+normal[ids]*.00065)[:, None, :], d.shape)
        rays = o3d.core.Tensor(np.concatenate([origin, d], axis=-1).astype("f4"))
        result[ids] = 1 - geometry.test_occlusions(rays, tfar=1.35, nthreads=5).numpy().mean(axis=1)
    return result


def camera(eye, target):
    forward = unit(target-eye)
    right = unit(np.cross(forward, [0, 0, 1]))
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], 1)
    pose[:3, 3] = eye
    return np.linalg.inv(pose)


@torch.no_grad()
def preview(state, metadata, destination, width=660, height=480, second=False):
    cat = metadata["category"]
    if cat in {"stereo", "rgbd"}:
        scale = max(metadata["width_m"], metadata["length_m"], .062)
        eye = scale*np.array([.72, -2.50, 2.55])
        target = np.array([0., 0., -.007])
    elif cat == "pothole":
        scale = max(.41, metadata["length_m"], metadata["width_m"])
        eye = scale*np.array([.92, -1.45, 1.66])
        target = np.array([0., 0., -.015])
    elif cat == "raised" and metadata["state"] == "removed_side":
        eye = np.array([1.84, -2.01, 2.50])
        target = np.array([.30, 0., -.05])
    elif second:
        eye, target = np.array([.86, 1.38, 1.55]), np.array([0., 0., .02])
    else:
        eye, target = np.array([.91, -1.45, 1.75]), np.array([0., 0., .055 if cat == "raised" else -.05])
    focal = width*.98
    if cat in {"stereo", "rgbd"}:
        dimensions = metadata["width_m"], metadata["length_m"]
        if max(dimensions)/min(dimensions) > 2.6:
            focal *= 1.55
    intrinsics = tensor([[focal, 0, width*.5], [0, focal, height*.5], [0, 0, 1]])[None]
    sh = torch.cat([state["feature_dc"], state["feature_rest"]], dim=1)
    rgb, visible, _ = rasterization(
        state["xyz"], state["rotation"], state["scaling"].exp(), state["opacity"].sigmoid()[:, 0],
        sh, tensor(camera(eye, target))[None], intrinsics, width, height,
        sh_degree=1, packed=False, backgrounds=tensor([[.23, .27, .30]]),
        rasterize_mode="antialiased"
    )
    if not bool(torch.isfinite(rgb).all()) or (visible > .55).sum().item() < 800:
        raise ValueError(f"{metadata['id']}: Gaussian preview has insufficient surface coverage")
    image = np.uint8(rgb[0].clamp(0, 1).cpu().numpy()*255+.5)
    Image.fromarray(image).save(destination, quality=93, subsampling=0)


def export(args, meta_path):
    metadata = json.loads(meta_path.read_text())
    asset = metadata["id"]
    dest = meta_path.parent
    ply = dest / "asset.ply"
    if not args.force and ply.exists() and (dest / "preview.jpg").exists() and "gaussians" in metadata:
        return
    sample_file = args.samples / (asset+".npz")
    if metadata["category"] in {"stereo", "rgbd"}:
        with np.load(sample_file) as data:
            surface = dict(data)
        surface.update(name=asset, geometric_ao=True)
    else:
        surface = sample_mesh(args.meshes / (asset+".npz"), dest / "surface.npz", metadata)
        if asset in args.keep_samples:
            np.savez_compressed(sample_file, **{k:v for k,v in surface.items() if isinstance(v, np.ndarray)})
    state, error = make_gaussians(surface)
    preview(state, metadata, dest / "preview.jpg")
    if metadata["category"] == "raised" and asset in args.keep_samples:
        preview(state, metadata, dest / "preview_lid.jpg", second=True)
    save_ply(state, ply)
    metadata.update(gaussians=len(state["xyz"]), sh_degree=1, ply_name=ply.name, gaussian_up="Z",
                    appearance_sh_fit_mae=error, standalone_preview="preview.jpg")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(asset, metadata["gaussians"], f"{error:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meshes", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--asset", action="append")
    parser.add_argument("--category", action="append")
    parser.add_argument("--keep-samples", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    files = sorted(args.out.glob("*/*/metadata.json"))
    if args.category:
        files = [f for f in files if f.parent.parent.name in args.category]
    if args.asset:
        files = [f for f in files if f.parent.name in args.asset]
        if len(files) != len(set(args.asset)):
            parser.error("One or more asset IDs have no mesh.")
    for path in files:
        export(args, path)
    print("GAUSSIAN EXPORT COMPLETE", len(files), flush=True)


if __name__ == "__main__":
    main()
