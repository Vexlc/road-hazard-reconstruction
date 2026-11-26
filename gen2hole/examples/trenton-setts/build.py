"""Deliver the two-view field mesh and the grade-relative stone exposure for vehicle simulation."""

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from cameras import FOCAL
from surface import HERE, S, photograph, reconstruct, native

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    flatten_scene,
    make_one,
    write_json,
    xyz,
    grid,
    grid_faces,
    face_up,
    texture_uv,
)

NAMES = [
    "Road adjoining the fracture",
    "Setts at the isolated asphalt loss",
    "Setts at the connected asphalt loss",
]


def resample(h):
    valid = h["labels"] > 0
    world = h["world"][valid]
    all_xy = np.vstack((world[:, :2], h["xy"][valid]))
    bounds = np.array([all_xy.min(0), all_xy.max(0)])
    center = np.mean(bounds, axis=0)
    field = grid(np.ptp(bounds, axis=0) + 0.04, 0.004)
    q = field + center
    tree = cKDTree(world[:, :2])
    distance, index = tree.query(q, k=4)
    label = h["labels"][valid][index[..., 0]]
    valid_out = distance[..., 0] < 0.0056
    weights = 1 / np.maximum(distance, 0.00008) ** 2
    weights *= h["labels"][valid][index] == label[..., None]
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)

    def smooth(a):
        return np.sum(a[valid][index] * weights, axis=-1)

    height = smooth(h["relief"])
    source = smooth(h["source_dz"]) + q @ h["grade"][:2] + h["grade"][2]
    # A single photograph hides short sides where depth changes. The road is
    # continuous across these samples. Fill them from adjoining geometry and
    # keep them explicitly outside the stereo-supported mask.
    measured = h["stereo_valid"][valid][index[..., 0]] & valid_out
    uv = h["cameras"][0].project(xyz(q, q @ h["grade"][:2] + h["grade"][2]))
    annotated = (uv - S["annotation_origin_px"]) / S["native_px_per_annotation_px"]
    u, l, r, b = S["crop_bounds"]
    valid_out = (
        (annotated[..., 0] >= u)
        & (annotated[..., 0] <= r)
        & (annotated[..., 1] >= l)
        & (annotated[..., 1] <= b)
    )
    label[~valid_out] = 0
    # Blend only where the camera crop ends, never between the intact asphalt
    # tongue and its neighbouring stones. Full field heights are kept separately.
    t = np.clip(distance_transform_edt(valid_out) * 0.004 / 0.075, 0, 1)
    return field, center, label, height * (t * t * (3 - 2 * t)), source, measured


def build(out):
    h = reconstruct()
    c0, c1 = h["cameras"]
    field, center, label, height, source, measured = resample(h)
    valid = label > 0
    ext = np.ptp(field, axis=(0, 1))
    grade_at_center = float(center @ h["grade"][:2] + h["grade"][2])
    world = xyz(field + center, source)
    # Use the actual photographic pixels at each node's recovered elevation.
    uv = c0.project(world)
    lo = np.maximum(np.floor(uv[valid].min(0) - 4).astype(int), 0)
    hi = np.minimum(np.ceil(uv[valid].max(0) + 5).astype(int), S["size"])
    rgb = photograph("detail")[lo[1] : hi[1], lo[0] : hi[0], ::-1].copy()
    portable = Mesh()
    inspection = Mesh()
    stride = (slice(None, None, 2), slice(None, None, 2))
    ff = grid_faces(*valid[stride].shape)
    flat = label[stride].ravel()
    ff = ff[(flat[ff] > 0).all(1)]
    vertex_lab = flat[ff]
    tri_lab = np.where(
        vertex_lab[:, 0] == vertex_lab[:, 1],
        vertex_lab[:, 0],
        np.where(vertex_lab[:, 1] == vertex_lab[:, 2], vertex_lab[:, 1], vertex_lab[:, 0]),
    )
    tex = texture_uv(uv[stride] - lo, hi - lo).reshape(-1, 2)
    for mesh, geometry in [
        (portable, xyz(field, height)),
        (inspection, xyz(field, source - grade_at_center)),
    ]:
        surface = mesh.material("Registered asphalt, exposed setts and joints", rgb, 0.90, 0, 0.40)
        v = geometry[stride].reshape(-1, 3)
        for code, name in enumerate(NAMES, 1):
            f = face_up(v, ff[tri_lab == code])
            mesh.add(name, v, f, tex, surface, road=code == 1)
    series = out / "series"
    folder = series / "assets/pothole/P583"
    cache = series / "mesh_cache"
    meta = dict(
        id="P583",
        family="trenton_setts",
        category="pothole",
        state="overlay_loss_exposing_setts",
        seed=29583,
        label="特伦顿街口沥青剥失与露石路面",
        surface_dimensions_m=ext.tolist(),
        references=S["images"],
        source_author=S["author"],
        source_license=S["license"],
        camera_eyes_m=[c.eye.tolist() for c in (c0, c1)],
        focal_px=float(FOCAL),
        nominal_casting_diameter_m=S["nominal_diameter_m"],
        local_grade_coefficients=h["grade"].tolist(),
        model_offset_xyz_m=[*center, grade_at_center],
        photographic_crop_xyxy=[*lo.astype(int).tolist(), *hi.astype(int).tolist()],
    )
    portable.save(folder, cache, meta, dict(width=ext[0], length=ext[1], height=height))
    flatten_scene(folder / "asset.glb")
    (folder / "field.glb").write_bytes(inspection.glb_bytes())
    flatten_scene(folder / "field.glb")
    origins = xyz(field, 1.0)
    d, part, _ = Surface(folder / "asset.glb").cast(
        origins, np.broadcast_to([0.0, 0.0, -1.0], origins.shape)
    )
    hit = np.isfinite(d)
    np.savez_compressed(
        folder / "surface.npz",
        width=ext[0],
        length=ext[1],
        height=np.where(hit, 1 - d, 0.0).astype("f4"),
        alpha=hit,
        first_part=part,
        installation_mask=hit,
        defect_mask=part > 1,
        stereo_valid=measured & hit,
        absolute_field_z_m=np.where(hit, source, np.nan).astype("f4"),
        registered_relief=True,
        photographic_rgb=True,
        water_mask=np.zeros_like(hit),
        opening_mask=np.zeros_like(hit),
    )
    np.savez_compressed(
        folder / "stereo.npz",
        grade=h["grade"],
        atlas_minmax=np.array([h["xy"][0, 0], h["xy"][-1, -1]]),
        pitch=h["pitch"],
        labels=h["labels"],
        raw_dz_m=h["raw_dz"].astype("f4"),
        road_dz_m=h["road_dz"].astype("f4"),
        source_dz_m=h["source_dz"].astype("f4"),
        accepted=h["stereo_valid"],
        cycle_error_m=h["cycle_error_m"].astype("f4"),
        peak_score=h["peak_score"],
        peak_margin=h["peak_margin"],
        sparse_grade_indices=np.flatnonzero(h["controls"]),
        training=h["fit"]["train"],
        heldout=h["fit"]["heldout"],
    )
    row = make_one(
        folder,
        cache,
        point_limit=480000,
        kernel_factors=(0.72, 0.16),
        part_importance={NAMES[1]: 5.0, NAMES[2]: 5.0},
    )
    for k, f in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("metadata", "metadata.json"),
        ("surface", "surface.npz"),
    ]:
        row[k] = "pothole/P583/" + f
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    build(parser.parse_args().out)
