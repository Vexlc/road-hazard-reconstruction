"""Construct the photographed cover, its exposed edge, frost-broken asphalt and adjoining road."""

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from camera import HERE, load

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    face_up,
    flatten_scene,
    grid,
    grid_faces,
    make_one,
    rect,
    texture_uv,
    write_json,
    xyz,
)


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def perimeter(poly, directions):
    """First intersection of each centre ray with a closed planar trace."""
    a = np.asarray(poly)
    edges = np.roll(a, -1, axis=0) - a
    denom = cross(directions[:, None], edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cross(a, edges) / denom
        t = cross(a, directions[:, None]) / denom
    hit = np.min(np.where((r > 0) & (t >= -1e-8) & (t <= 1 + 1e-8), r, np.inf), axis=1)
    if not np.isfinite(hit).all():
        raise ValueError("Trace does not enclose face centre")
    return directions * hit[:, None]


def angles(xy):
    return np.arctan2(xy[..., 1], xy[..., 0])


def mask_distance(poly, shape):
    m = cv2.fillPoly(np.zeros(shape, "u1"), [np.asarray(poly, "i4")], 1)
    return cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def sample(image, px):
    return cv2.remap(
        image,
        px[..., 0].astype("f4"),
        px[..., 1].astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def bake(cam, s, photo, flat):
    mesh = Mesh()
    iron = mesh.material("Photographed cast iron", photo, 0.82, 0.27, 0.42, lossless_color=True)
    asphalt = mesh.material(
        "Photographed Callander Road asphalt", photo, 0.96, 0, 0.7, lossless_color=True
    )

    def uv(v):
        return texture_uv(cam.project(v), s["size_px"])

    lid, rim, crack = [
        cam.plane(s[name])[:, :2] for name in ("face_corners_px", "iron_outer_px", "break_outer_px")
    ]
    span = np.array([1.34, 1.34])
    outer = rect(span)[:, :2]
    angle = np.unique(
        np.round(
            np.concatenate(
                [np.linspace(-np.pi, np.pi, 321)[:-1]]
                + [angles(p) for p in (lid, rim, crack, outer)]
            ),
            10,
        )
    )
    angle = np.r_[angle, angle[0] + 2 * np.pi]
    rays = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    face, frame, fracture, road = [perimeter(p, rays) for p in (lid, rim, crack, outer)]
    widths = [np.linalg.norm(p, axis=-1) for p in [face, frame, fracture, road]]
    if np.min(np.diff(widths, axis=0)) < -0.0001:
        raise ValueError("Crossed survey outlines")

    def band(name, inner, outer, n, mat, elevation=None, background=False):
        t = np.linspace(0, 1, n)[:, None, None]
        pts = inner * (1 - t) + outer * t
        if elevation is not None:
            pts[..., 2] += elevation(pts, t[..., 0])
        tri = grid_faces(*pts.shape[:2])
        verts = pts.reshape(-1, 3)
        a, b, c = verts[tri].transpose(1, 0, 2)
        tri = face_up(verts, tri[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-12])
        mesh.add(name, verts, tri, uv(verts), mat, background)

    # The top is a single closed face. The cast lettering and diagonal line are photographed.
    band("Lid face", xyz(face * 0, 0), xyz(face, 0), 16, iron)
    band("Cast ring and cover seat", xyz(face, 0), xyz(frame, 0), 4, iron)
    pairs = s["front_edge_top_and_toe_px"]
    # Recorded front-edge stations drive the visible step. Other perimeter stations
    # control the taper where intact or fragmented asphalt reaches the iron.
    controls = [
        ([202.6, 180.3], 0),
        ([340.8, 156], 0),
        ([364, 190], 0.008),
        ([382, 218], 0.023),
        ([391.4, 233], cam.drop(pairs[-1])),
        *[(q[0], cam.drop(q)) for q in pairs],
        ([333, 244.6], 0),
        ([216.7, 267.8], 0),
    ]
    az = angles(cam.plane([u for u, _ in controls])[:, :2])
    drop = np.interp(angle, az, [d for _, d in controls], period=2 * np.pi)
    if flat:
        drop[:] = 0
    if not flat:
        low, high = xyz(frame, -drop), xyz(frame, 0)
        verts = np.vstack((low, high))
        n = len(frame)
        i = np.arange(n - 1)
        side = np.vstack((np.c_[i, i + 1, i + n], np.c_[i + 1, i + n + 1, i + n]))
        side = side[
            np.linalg.norm(
                np.cross(
                    verts[side[:, 1]] - verts[side[:, 0]], verts[side[:, 2]] - verts[side[:, 0]]
                ),
                axis=1,
            )
            > 1.0e-10
        ]
        mesh.add("Casting perimeter wall", verts, side, uv(verts), iron)
    pocket = mask_distance(s["pocket_px"], photo.shape[:2])
    blocks = np.maximum.reduce([mask_distance(p, photo.shape[:2]) for p in s["blocks_px"]])

    def relief(pts, t):
        if flat:
            return np.zeros(pts.shape[:2])
        # Additional bed relief is a shallow hypothesis, with a zero-height outer boundary.
        px = cam.project(pts)
        weight = np.sin(np.pi * t)
        return weight * (
            -0.012 * np.clip(sample(pocket, px) / 5, 0, 1)
            + 0.004 * np.clip(sample(blocks, px) / 2, 0, 1)
        )

    band("Fracture lip and exposed bed", xyz(frame, -drop), xyz(fracture, 0), 42, asphalt, relief)
    band(
        "Road adjoining the fracture", xyz(fracture, 0), xyz(road, 0), 15, asphalt, background=True
    )
    return mesh, span


def build(out):
    s, cam = load()
    source = HERE / "references/1673294.jpg"
    if hashlib.sha256(source.read_bytes()).hexdigest() != s["photo_sha256"]:
        raise ValueError("Source digest")
    photo = np.asarray(Image.open(source).convert("RGB"))
    series = out / "series"
    folder = series / "assets/raised/A580"
    ref = series / "references/A580.jpg"
    ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, ref)
    cache = series / "mesh_cache"
    mesh, span = bake(cam, s, photo, False)
    xy = grid(span, 0.004)
    ext = np.ptp(xy, axis=(0, 1))
    meta = dict(
        id="A580",
        family="callander_cover",
        category="raised",
        state="frost_broken_surround",
        seed=29580,
        label=s["label"],
        source=s["photo"],
        source_author=s["author"],
        source_license=s["licence"],
        source_sha256=s["photo_sha256"],
        source_image="references/A580.jpg",
        surface_dimensions_m=ext.tolist(),
        nominal_face_width_m=s["nominal_face_width_m"],
        fitted_face_length_m=cam.length,
        camera_eye_m=cam.eye.tolist(),
        modeled_front_exposure_m=[cam.drop(p) for p in s["front_edge_top_and_toe_px"]],
    )
    field = dict(width=ext[0], length=ext[1], height=np.zeros(xy.shape[:2]))
    mesh.save(folder, cache, meta, field)
    flatten_scene(folder / "asset.glb")
    with tempfile.TemporaryDirectory() as stage:
        stage = Path(stage)
        flat, _ = bake(cam, s, photo, True)
        flat.save(stage / "flat", stage / "cache", meta, field)
        flatten_scene(stage / "flat/asset.glb")
        shutil.copyfile(stage / "flat/asset.glb", folder / "flush.glb")
    start = xyz(xy, 1)
    surface = Surface(folder / "asset.glb")
    t, part, _ = surface.cast(start, np.broadcast_to([0.0, 0.0, -1.0], start.shape))
    valid = np.isfinite(t)
    h = np.where(valid, 1 - t, 0)
    delta = xyz(xy, h) - cam.eye
    target = np.linalg.norm(delta, axis=-1)
    seen, _, _ = surface.cast(np.broadcast_to(cam.eye, delta.shape), delta / target[..., None])
    view_count = (valid & (abs(seen - target) < 0.002)).astype("u1")
    np.savez_compressed(
        folder / "surface.npz",
        view_count=view_count,
        attached_cover_surface=True,
        width=ext[0],
        length=ext[1],
        height=h.astype("f4"),
        alpha=valid,
        installation_mask=part > 1,
        defect_mask=part > 1,
        first_part=part,
        opening_mask=np.zeros_like(valid),
        water_mask=np.zeros_like(valid),
        photographic_rgb=True,
    )
    row = make_one(folder, cache, point_limit=450000, kernel_factors=(0.72, 0.14))
    for k, f in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("surface", "surface.npz"),
        ("metadata", "metadata.json"),
    ]:
        row[k] = "raised/A580/" + f
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
