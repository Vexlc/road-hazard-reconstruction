"""Build a textured road asset from an aligned RGB image, mask and stereo XYZ map."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat
from scipy.spatial import cKDTree
import trimesh


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fit_plane(points):
    samples = points[::max(1, len(points) // 30000)]
    active = np.ones(len(samples), bool)
    for _ in range(6):
        center = samples[active].mean(axis=0)
        relative = samples[active] - center
        normal = np.linalg.eigh(relative.T @ relative)[1][:, 0]
        residual = (samples - center) @ normal
        sigma = np.median(np.abs(residual[active])) * 1.4826
        active = np.abs(residual) < max(.002, sigma * 2.8)
    if normal @ center > 0:
        normal = -normal
    return center, normal, float(np.sqrt(np.mean(residual[active] ** 2)))


def image_projection(xyz, rows, cols):
    anchors = np.arange(0, len(xyz), max(1, len(xyz) // 20000))
    q = xyz[anchors]
    normalized = q[:, :2] / q[:, 2, None]
    coefficients = np.linalg.lstsq(
        np.c_[normalized, np.ones(len(q))], np.c_[cols[anchors], rows[anchors]], rcond=None
    )[0]
    error = np.c_[normalized, np.ones(len(q))] @ coefficients - np.c_[cols[anchors], rows[anchors]]
    rms = float(np.sqrt(np.mean(error ** 2)))
    if rms > .35:
        raise ValueError(f"RGB and XYZ pinhole alignment error is {rms:.2f} pixels")
    return coefficients, rms


def grid_faces(rows, cols):
    indices = np.arange(rows * cols).reshape(rows, cols)
    a, b = indices[:-1, :-1], indices[:-1, 1:]
    c, d = indices[1:, :-1], indices[1:, 1:]
    return np.stack([np.stack([a, c, b], -1), np.stack([c, d, b], -1)], -2).reshape(-1, 3)


def project_grid(rgb, pixel_map, xyz):
    uv = np.einsum("...i,ij->...j", np.concatenate([
        xyz[..., :2] / xyz[..., 2, None], np.ones(xyz.shape[:-1] + (1,))
    ], axis=-1), pixel_map)
    coordinates = [uv[..., 1], uv[..., 0]]
    color = np.stack([ndimage.map_coordinates(rgb[:, :, c], coordinates, order=1, mode="nearest")
                      for c in range(3)], axis=-1)
    return coordinates, color


def reconstruct(args):
    output, sample_path = args.output, args.samples
    output.mkdir(parents=True, exist_ok=True)
    rgb = np.array(Image.open(args.rgb).convert("RGB"))
    mask = np.array(Image.open(args.mask).convert("L")) > 0
    labels, count = ndimage.label(mask)
    if not count:
        raise ValueError("The target mask is empty")
    largest = np.bincount(labels.ravel())[1:].argmax() + 1
    mask = ndimage.binary_fill_holes(labels == largest)
    xyz = loadmat(args.xyz)[args.xyz_key].astype(np.float32) * args.metres_per_unit
    if xyz.shape != rgb.shape or mask.shape != rgb.shape[:2]:
        raise ValueError("RGB, mask and XYZ must share the source resolution")
    low, high = np.array(np.where(mask)).min(axis=1), np.array(np.where(mask)).max(axis=1) + 1
    padding = np.maximum((high - low) * .55, 55).astype(int)
    top, left = np.maximum(0, low - padding)
    bottom, right = np.minimum(mask.shape, high + padding)
    region = np.s_[top:bottom, left:right]
    Image.fromarray(rgb[region]).save(output / "source_crop.png")
    Image.fromarray(np.uint8(mask[region]) * 255).save(output / "source_mask.png")

    cropped = xyz[region]
    valid = np.isfinite(cropped).all(axis=-1) & (cropped[:, :, 2] > .10) & (cropped[:, :, 2] < 50)
    near_target = ndimage.binary_dilation(mask[region], iterations=12)
    ground = cropped[valid & ~near_target]
    pit = cropped[valid & mask[region]]
    if min(len(ground), len(pit)) < 100:
        raise ValueError("Too few valid XYZ points on the pit or the surrounding road")
    plane_center, normal, residual = fit_plane(ground)
    right_axis = np.array([1., 0, 0]) - normal[0] * normal
    right_axis /= np.linalg.norm(right_axis)
    basis = np.stack([right_axis, np.cross(normal, right_axis), normal], axis=1)
    local_pit = (pit - plane_center) @ basis
    min_xy, max_xy = np.percentile(local_pit[:, :2], [.4, 99.6], axis=0)
    center = plane_center + ((min_xy + max_xy) * .5) @ basis[:, :2].T
    size = max_xy - min_xy
    dimensions = size + 2 * args.border
    ncols, nrows = np.clip(np.ceil(dimensions / .00082).astype(int) + 1, 120, 700)
    width, length = dimensions
    x = np.linspace(-width / 2, width / 2, ncols)
    y = np.linspace(length / 2, -length / 2, nrows)
    xx, yy = np.meshgrid(x, y)
    sample_xy = np.stack([xx, yy], -1)
    spacing = max(width / (ncols - 1), length / (nrows - 1))
    local = (cropped[valid] - center) @ basis
    limit = np.abs(local[:, 2]) < max(.12, np.percentile(np.abs(local_pit[:, 2]), 99) * 2)
    local = local[limit]
    tree = cKDTree(local[:, :2])
    distance, indices = tree.query(sample_xy, k=8, workers=4)
    nearby_z = local[indices, 2]
    median_z = np.median(nearby_z, axis=-1, keepdims=True)
    weights = 1 / (distance ** 2 + (.5 * spacing) ** 2)
    weights *= np.abs(nearby_z - median_z) <= max(.007, 3 * residual)
    height = np.sum(weights * nearby_z, axis=-1) / np.maximum(np.sum(weights, axis=-1), 1e-10)
    support = distance[..., 0] <= max(.007, 3 * spacing)
    if support.mean() < .55:
        raise ValueError("The selected border has insufficient stereo support. Reduce --border.")
    height = ndimage.gaussian_filter(height, .85)
    edge_distance = np.minimum(width / 2 - np.abs(xx), length / 2 - np.abs(yy))
    height *= smoothstep(edge_distance / (args.border * .75))
    alpha = smoothstep(edge_distance / (args.border * .75))
    xyz_local = np.stack([xx, yy, height], -1)

    r, c = np.where(valid)
    pixel_map, reprojection_error = image_projection(cropped[valid], r + top, c + left)
    coordinates, color = project_grid(rgb, pixel_map, xyz_local @ basis.T + center)
    texture_path = output / "photo_texture.png"
    texture = Image.fromarray(color)
    texture.save(texture_path)
    mask_distance_px = ndimage.distance_transform_edt(~mask)
    # Project the measured contour to the rectified mesh. The target road supplies
    # the surrounding color, with a short blend through the photographed rim.
    mask_distance = ndimage.map_coordinates(mask_distance_px, coordinates, order=1, mode="nearest")
    projected_mask = mask_distance < .5
    road_weight = smoothstep((mask_distance - 4) / 23)
    if projected_mask.sum() < 100:
        raise ValueError("The measured target did not project onto the reconstructed mesh")

    uv = np.stack([xx / width + .5, yy / length + .5], -1).reshape(-1, 2)
    faces = grid_faces(nrows, ncols)
    material = trimesh.visual.material.PBRMaterial(
        name="Photographed asphalt", baseColorTexture=texture, roughnessFactor=.94, metallicFactor=0
    )
    mesh = trimesh.Trimesh(
        vertices=xyz_local.reshape(-1, 3)[:, [0, 2, 1]] * [1, 1, -1], faces=faces,
        visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=material), process=False
    )
    assert np.all(mesh.face_normals[:, 1] > 0)
    (output / "pothole.glb").write_bytes(mesh.export(file_type="glb"))
    np.savez_compressed(output / "surface.npz", width=width, length=length,
                        height=height.astype("f4"), alpha=alpha.astype("f4"))

    keep = alpha > .01
    hy, hx = np.gradient(height, y, x)
    normals = np.stack([-hx[keep], -hy[keep], np.ones(keep.sum())], -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    tangents = np.stack([np.ones(keep.sum()), np.zeros(keep.sum()), hx[keep]], -1)
    tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True)
    bitangents = np.cross(normals, tangents)
    scale_x = .78 * (x[1] - x[0]) * np.sqrt(1 + hx[keep] ** 2)
    scale_y = .78 * (y[0] - y[1]) * np.sqrt(1 + hy[keep] ** 2)
    scales = np.stack([scale_x, scale_y, np.full_like(scale_x, .00028)], -1)
    rgba = np.c_[color[keep] / 255, np.ones(keep.sum())]
    # Camera RGB already encodes irradiance, so keep the relighting close to neutral.
    fields = dict(local_xyz=xyz_local[keep], normals=normals,
                  rotation_matrices=np.stack([tangents, bitangents, normals], -1),
                  scales=scales, alpha=alpha[keep], uv=uv[keep.ravel()], color=rgba,
                  normal=np.tile([.5, .5, 1], (keep.sum(), 1)),
                  arm=np.tile([1, .96, 0], (keep.sum(), 1)), ao=np.ones((keep.sum(), 1)),
                  sun_visibility=np.ones(keep.sum()), road_mask=road_weight[keep],
                  photo_appearance=np.array(1))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_path, **{k: v.astype("f4") for k, v in fields.items()})
    np.savez_compressed(output / "depth_diagnostics.npz", height=height.astype("f4"),
                        projected_mask=projected_mask, support=support, x=x, y=y)
    inputs = {k: {"filename": p.name, "sha256": sha256(p.read_bytes()).hexdigest()}
              for k, p in zip(("rgb", "mask", "xyz"), (args.rgb, args.mask, args.xyz))}
    report = {
        "source": args.source, "inputs": inputs,
        "meters_per_xyz_unit": args.metres_per_unit, "source_crop_xyxy": list(map(int, [left, top, right, bottom])),
        "road_fit_rms_mm": residual * 1000, "reprojection_rms_pixels": reprojection_error,
        "pothole_extent_xy_m_p0_4_to_p99_6": size.tolist(), "mesh_extent_xy_m": dimensions.tolist(),
        "depth_mm_p50_p95_p99_in_mask": (np.percentile(-height[projected_mask], [50, 95, 99]) * 1000).tolist(),
        "stereo_near_sample_coverage_patch": float(support.mean()),
        "stereo_near_sample_coverage_in_mask": float(support[projected_mask].mean()),
        "hole_samples": len(pit), "mesh_triangles": len(faces), "gaussians": int(keep.sum()),
        "construction": "Organized stereo XYZ, robust surrounding plane, local height field and reprojected source RGB",
        "limits": ["Height field cannot represent hidden overhangs", "Sparse areas use local interpolation",
                   "Camera RGB contains captured illumination", "Depth statistics are estimates from stereo"]
    }
    (output / "capture_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--xyz-key", default="xyzPoints")
    parser.add_argument("--metres-per-unit", type=float, required=True)
    parser.add_argument("--border", type=float, default=.10, help="Surrounding road width in metres.")
    parser.add_argument("--source", default="User-supplied capture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True, help="Output Gaussian inputs NPZ.")
    arguments = parser.parse_args()
    if not (0 < arguments.border < 2 and 0 < arguments.metres_per_unit):
        parser.error("Use a positive scale and border smaller than 2 m")
    reconstruct(arguments)
