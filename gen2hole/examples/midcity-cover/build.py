"""Recover the rim, exposed iron sleeve, displaced lid, notch, and raised casting relief."""

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from calibration import Calibration, HERE, RADIUS, SIZE, SPEC, THICKNESS, circle
from casting import (
    CENTER,
    SIDE,
    SPAN,
    UPRIGHT,
    cast_shapes,
    model_xy,
    perimeter,
    texture,
    transform,
)

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    face_polygon,
    face_up,
    flatten_scene,
    grid,
    make_one,
    rect,
    rings_faces,
    write_json,
    xyz,
)


def sampled(photo, px):
    q = (px + 0.5) * np.array(photo.shape[1::-1]) / SIZE - 0.5
    return cv2.remap(
        photo,
        q[..., 0].astype("f4"),
        q[..., 1].astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def contact_textures(photo, cam, span, chip):
    h, w = chip.shape[:2]
    yy, xx = np.indices((h, w))
    uv = (np.stack([(xx + 0.5) / w, (yy + 0.5) / h], axis=-1) - 0.5) * [1, -1]
    xy = uv * span
    px = cam.project(xyz(xy, 0))
    allowed = np.full(tuple(SIZE[::-1]), 255, "u1")
    cv2.fillPoly(allowed, [np.int32([[0, 0], [148, 0], [0, 184]])], 0)
    rim = cam.project(circle([0, 0], RADIUS + 0.021, 0))
    lid = cam.project(xyz(perimeter(cam), THICKNESS))
    for loop in [rim, lid]:
        cv2.fillPoly(allowed, [np.int32(np.rint(loop))], 0)
    allowed = cv2.erode(allowed, np.ones((9, 9), "u1"))

    def weights(mask, q):
        return (
            cv2.remap(mask, q[..., 0].astype("f4"), q[..., 1].astype("f4"), cv2.INTER_LINEAR) / 255
        )

    # Continue the nearest visible concrete a few pixels beneath opaque geometry.
    # This also preserves contact shadow along the front of the displaced lid.
    distance, nearest = distance_transform_edt(allowed == 0, return_indices=True)
    yy0, xx0 = np.rint(px[..., 1]).astype(int), np.rint(px[..., 0]).astype(int)
    h0, w0 = allowed.shape
    inside = (yy0 >= 0) & (xx0 >= 0) & (yy0 < h0) & (xx0 < w0)
    yy0 = np.clip(yy0, 0, h0 - 1)
    xx0 = np.clip(xx0, 0, w0 - 1)
    extend = inside & (distance[yy0, xx0] > 0) & (distance[yy0, xx0] < 18)
    ground_px = px.copy()
    ground_px[extend] = nearest[:, yy0[extend], xx0[extend]][::-1].T
    valid = weights(allowed, ground_px)
    valid *= cv2.GaussianBlur(valid, (0, 0), 2)
    ground = np.uint8(
        np.rint(
            valid[..., None] * sampled(photo, ground_px)
            + (1 - valid[..., None]) * sampled(photo, chip)
        )
    )
    # The registration is well constrained on the inner edge. Preserve photographed
    # radial wear without allowing the dark bore or adjoining slab to spill onto iron.
    yy, xx = np.indices((512, 512))
    q = (np.stack([xx + 0.5, 511.5 - yy], axis=-1) / 512 - 0.5) * 0.68
    radius = np.linalg.norm(q, axis=-1)
    angle = np.arctan2(q[..., 1], q[..., 0])
    block = np.full(tuple(SIZE[::-1]), 255, "u1")
    cv2.fillPoly(block, [np.int32(np.rint(lid))], 0)
    block = cv2.erode(block, np.ones((15, 15), "u1"))

    def ring(z, lo, hi):
        r = np.clip(radius, lo, hi)

        def at(t):
            return cam.project(xyz(np.stack([r * np.cos(t), r * np.sin(t)], axis=-1), z))

        p = at(angle)
        w = weights(block, p)[..., None]
        # Reflect the hidden sector into the opposite visible side at the same radius.
        return np.uint8(
            np.rint(w * sampled(photo, p) + (1 - w) * sampled(photo, at(np.pi - angle)))
        )

    return (
        ground,
        ring(0, RADIUS + 0.004, RADIUS + 0.015),
        ring(cam.ledge_z, cam.ledge_radius + 0.004, RADIUS - 0.004),
    )


def band(mesh, name, a, b, mat, uv, reverse=False):
    n = len(a)
    i = np.arange(n)
    j = (i + 1) % n
    f = np.vstack([np.c_[i, j, n + j], np.c_[i, n + j, n + i]])
    v = np.vstack([a, b])
    mesh.add(name, v, f[:, ::-1] if reverse else f, uv(v), mat)


def relief(mesh, cam, polygons, mat, uv, height):
    vertices, faces = [], []
    start = 0
    for p in polygons:
        loops = [model_xy(np.asarray(line.coords)[:-1], cam) for line in [p.exterior, *p.interiors]]
        top = []
        sides = []
        offset = 0
        for j, loop in enumerate(loops):
            # CCW exterior and CW holes in metric XY; upper edges taper into the casting.
            direction = np.cross(loop, np.roll(loop, -1, axis=0)).sum()
            if (direction > 0) != (j == 0):
                loop = loop[::-1]
            loops[j] = loop
            centroid = loop.mean(axis=0)
            vec = loop - centroid
            top.append(
                loop
                + (-1 if j == 0 else 1)
                * 0.00085
                * vec
                / np.maximum(0.0001, np.linalg.norm(vec, axis=1, keepdims=True))
            )
            i = offset + np.arange(len(loop))
            sides.append((i, np.roll(i, -1)))
            offset += len(loop)
        base, cap = xyz(np.vstack(loops), THICKNESS), xyz(np.vstack(top), THICKNESS + height)
        n = len(base)
        tri = [face_up(cap, rings_faces(top)) + n]
        for i, j in sides:
            tri.extend((np.c_[i, j, n + j], np.c_[i, n + j, n + i]))
        vertices.append(np.vstack([base, cap]))
        faces.append(np.vstack(tri) + start)
        start += 2 * n
    v = np.vstack(vertices)
    mesh.add("Lid face", v, np.vstack(faces), uv(v), mat)


def build(out):
    src = HERE / "references/M570.jpg"
    if hashlib.sha256(src.read_bytes()).hexdigest() != SPEC["source_sha256"]:
        raise ValueError("Source hash mismatch")
    photo = np.asarray(Image.open(src).convert("RGB"))
    cam = Calibration()
    out.mkdir(exist_ok=True, parents=True)
    mesh = Mesh()
    slab_span = np.array([2.08, 1.60])
    # Mirror only this clean patch of the original concrete; it contains no metal, vegetation or shadow.
    rows, cols = np.indices((640, 768))
    xx = (cols + 0.5) / 768 * 1500
    yy = (rows + 0.5) / 640 * 1200
    mirror = lambda x, w: w - abs(x % (2 * w) - w)
    chip = np.stack([590 + mirror(xx, 215), 5 + mirror(yy, 58)], axis=-1)
    ground_rgb, rim_rgb, shelf_rgb = contact_textures(photo, cam, slab_span, chip)
    slab = mesh.material("Sunlit Mid-City concrete", ground_rgb, 0.94, 0.0, 0.72)
    uv_slab = lambda v: v[:, :2] / slab_span + 0.5
    zero = [0, 0]
    edge = circle(zero, RADIUS, 0)
    outer = circle(zero, RADIUS + 0.020, 0)
    face_polygon(mesh, "Road around the aperture", rect(slab_span), [outer], slab, uv_slab)
    foot = cam.ledge_z
    seat = circle(zero, cam.ledge_radius, foot)
    raw = circle(zero, RADIUS, foot)

    # Map visible far-side photographic strips around the rest of the cylinder at constant elevation.
    def strip(name, r, zlo, zhi, yscale=160):
        v, u = np.indices((yscale, 600))
        u = (u + 0.5) / 600
        phi = np.pi / 2 + np.arcsin(0.78 * np.sin(2 * np.pi * u))
        zz = zhi + (zlo - zhi) * (v + 0.5) / yscale
        world = np.stack([r * np.cos(phi), r * np.sin(phi), zz], axis=-1)
        return mesh.material(
            name, sampled(photo, cam.project(world)), 0.83, 0.28, 0.46, lossless_color=True
        )

    def cylindrical(v):
        angle = np.arctan2(v[:, 1], v[:, 0])
        return np.c_[
            0.5 + np.sin(angle - np.pi / 2) * 0.46,
            np.clip((v[:, 2] - foot) / (-foot), 0.005, 0.995),
        ]

    # Use clean, small patches on each bearing surface, avoiding dark pixels belonging to the bore.
    rim = mesh.material("Worn outer casting", rim_rgb, 0.76, 0.31, 0.32)
    bright = mesh.material("Bright bearing shelf", shelf_rgb, 0.70, 0.30, 0.22)
    uv_ring = lambda v: v[:, :2] / 0.68 + 0.5
    riser = strip("Oxidised riser", RADIUS, foot + 0.001, -0.001)
    face_polygon(mesh, "Cast ring and cover seat", outer, [edge], rim, uv_ring)
    band(mesh, "Casting perimeter wall", raw, edge, riser, cylindrical, reverse=True)
    face_polygon(mesh, "Cast seat riser and ledge", raw, [seat], bright, uv_ring)
    angles = np.arctan2(cam.wall_samples[:, 1], cam.wall_samples[:, 0])
    order = np.argsort(angles)
    th = np.arange(len(seat)) * 2 * np.pi / len(seat)
    z = np.interp(th, angles[order], cam.wall_samples[order, 2], period=2 * np.pi)
    end = circle(zero, cam.ledge_radius, z)
    lining = strip("Visible rusted bore sleeve", cam.ledge_radius, z.mean() + 0.008, foot - 0.006)
    uv_wall = lambda v: np.c_[
        0.5 + 0.47 * v[:, 0] / cam.ledge_radius,
        np.clip((v[:, 2] - z.mean()) / (foot - z.mean()), 0.005, 0.995),
    ]
    band(mesh, "Inspection shaft", end, seat, lining, uv_wall, reverse=True)
    bottom_z = -SPEC["visual_closure_depth_assumed_m"]
    bottom = circle(zero, cam.ledge_radius, bottom_z)
    dark = mesh.material(
        "Shadow from the photographed opening", photo[690:780, 1165:1295], 0.98, 0, 0.05
    )
    band(mesh, "Cavity inferred closure", bottom, end, dark, uv_slab, reverse=True)
    face_polygon(mesh, "Shaft bottom", bottom, [], dark, uv_slab)
    # Remove dark infill surfaces from the observation-only GLB for sensor applications.
    visual_ids = {"Cavity inferred closure", "Shaft bottom"}
    atlas = texture(photo, cam)
    topmat = mesh.material(
        "Photographed LDH iron face", atlas, 0.79, 0.26, 0.38, lossless_color=True
    )
    iron = mesh.material("Worn exposed casting edge", atlas[592:605, 201:252], 0.85, 0.26, 0.35)

    def uv_top(v):
        xy = (v[:, :2] - cam.lid_xy) * [1, -1] * SIDE / SPAN + CENTER
        return ((transform(xy, UPRIGHT) + 0.5) / SIDE) * [1, -1] + [0, 1]

    per = perimeter(cam)
    lower, upper = xyz(per, 0.0), xyz(per, THICKNESS)
    face_polygon(mesh, "Lid face", upper, [], topmat, uv_top)
    band(mesh, "Lid rolled edge", lower, upper, iron, uv_top)
    mesh.add(
        "Lid underside", lower, face_up(lower, rings_faces([per]))[:, ::-1], uv_top(lower), iron
    )
    tread, glyphs = cast_shapes(atlas)
    # Single-source image has no elevation gauge. The relief height is an explicit modelling prior.
    relief(mesh, cam, tread + glyphs, topmat, uv_top, 0.0035)
    # Merge equal part names for renderer compatibility.
    indices = [i for i, p in enumerate(mesh.parts) if p[0] == "Lid face"]
    a, b = [mesh.parts[i] for i in indices]
    mesh.parts[indices[0]] = (
        a[0],
        np.vstack([a[1], b[1]]),
        np.vstack([a[2], b[2] + len(a[1])]),
        np.vstack([a[3], b[3]]),
        a[4],
        a[5],
    )
    mesh.parts.pop(indices[1])
    series = out / "series"
    folder = series / "assets/missing/M570"
    cache = series / "mesh_cache"
    xy = grid(slab_span, 0.004)
    ext = np.ptp(xy, axis=(0, 1))
    meta = dict(
        id="M570",
        family="midcity_cover",
        category="missing",
        state="partially_overlapping_displaced_cover",
        seed=29570,
        label="新奥尔良偏移圆盖与部分敞开的井口",
        source=SPEC["photo"],
        source_author=SPEC["credit"],
        source_license=SPEC["licence"],
        source_sha256=SPEC["source_sha256"],
        surface_dimensions_m=ext.tolist(),
        lid_center_xy_m=cam.lid_xy.tolist(),
        lid_radius_m=cam.lid_radius,
        mouth_radius_nominal_m=RADIUS,
        inner_bore_radius_m=cam.ledge_radius,
        ledge_z_m=foot,
        far_wall_lower_edge_xyz_m=cam.wall_samples.tolist(),
        visual_only_closure_depth_m=-bottom_z,
        relief_height_assumed_m=0.0035,
        tread_islands=len(tread),
    )
    mesh.save(folder, cache, meta, dict(width=ext[0], length=ext[1], height=np.zeros(xy.shape[:2])))
    flatten_scene(folder / "asset.glb")
    mesh.parts = [part for part in mesh.parts if part[0] not in visual_ids]
    (folder / "observed.glb").write_bytes(mesh.glb_bytes())
    flatten_scene(folder / "observed.glb")
    origins = xyz(xy, 1.0)
    ray = np.broadcast_to([0.0, 0.0, -1.0], origins.shape)
    dist, part, _ = Surface(folder / "asset.glb").cast(origins, ray)
    measured, _, _ = Surface(folder / "observed.glb").cast(origins, ray)
    active = ~np.isfinite(measured) & np.isfinite(dist)
    np.savez_compressed(
        folder / "surface.npz",
        width=ext[0],
        length=ext[1],
        height=np.where(np.isfinite(dist), 1 - dist, 0).astype("f4"),
        alpha=np.isfinite(dist),
        opening_mask=active,
        observed_depth_valid=np.isfinite(measured),
        defect_mask=part > 1,
        installation_mask=part > 1,
        water_mask=np.zeros_like(active),
        photographic_rgb=True,
    )
    row = make_one(
        folder,
        cache,
        point_limit=350000,
        kernel_factors=(0.66, 0.13),
        part_importance={"Lid face": 5.0, "Inspection shaft": 3.0},
    )
    for key, file in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("surface", "surface.npz"),
        ("metadata", "metadata.json"),
    ]:
        row[key] = "missing/M570/" + file
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
