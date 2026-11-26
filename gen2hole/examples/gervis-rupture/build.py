"""Recover three lifted edges and the visible road between them from the Gervis Place photo."""

import argparse
import hashlib
import tempfile
import shutil
import sys

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from pathlib import Path
from camera import HERE, load
from material import portable_asphalt, face_bands, roof_bands
import cv2
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    face_up,
    flatten_scene,
    grid,
    grid_faces,
    make_one,
    rings_faces,
    texture_uv,
    write_json,
    xyz,
)


def outline(points):
    return Polygon(np.asarray(points)[..., :2]).buffer(0)


def components(shape):
    if shape.geom_type == "Polygon":
        if shape.area > 1.0e-9:
            yield shape
    else:
        for part in getattr(shape, "geoms", []):
            yield from components(part)


def patchwork(shape, pitch=0.18):
    """Cut sparse planar polygons into metric tiles to limit sheet-to-photo UV distortion."""
    x0, y0, x1, y1 = shape.bounds
    vertices = []
    triangles = []
    offset = 0
    for x in np.arange(x0, x1, pitch):
        for y in np.arange(y0, y1, pitch):
            for p in components(shape.intersection(box(x, y, x + pitch, y + pitch))):
                rings = [np.asarray(r.coords)[:-1] for r in [p.exterior, *p.interiors]]
                v = np.vstack(rings)
                f = rings_faces(rings)
                if not len(f):
                    continue
                vertices.append(v)
                triangles.append(f + offset)
                offset += len(v)
    return xyz(np.vstack(vertices), 0), np.vstack(triangles)


def sheet(cam, rec, photo, heldout_x):
    crest, toe, hinge = [np.array(rec[k], float) for k in ("crest", "toe", "return_to_grade")]
    h = cam.lips(crest, toe)
    if np.min(h) <= 0:
        raise ValueError("Toe projects above crest")
    # Sample along the hand-traced edge, preserving every observed change of direction.
    q = np.linspace(0, len(h) - 1, 6 * (len(h) - 1) + 1)

    def interp(coords):
        return np.column_stack([np.interp(q, np.arange(len(h)), a) for a in coords.T])

    front, back = interp(crest), interp(hinge)
    foot = interp(toe)
    # Resolve small chips photographically at intermediate stations. Keep control
    # vertices pinned and exclude the three validation columns and their immediate neighbours.
    grey = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype("f4")
    gy = cv2.GaussianBlur(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3), (3, 3), 0.6)
    for path, sign in [(front, -1), (foot, 1)]:
        offsets = np.arange(-1.5, 1.51, 0.25)
        u = np.broadcast_to(path[:, 0, None], (len(path), len(offsets))).astype("f4")
        v = (path[:, 1, None] + offsets).astype("f4")
        evidence = sign * cv2.remap(gy, u, v, cv2.INTER_LINEAR) - 27 * offsets**2
        correction = offsets[np.argmax(evidence, axis=1)]
        fixed = abs(q - np.rint(q)) < 1e-8
        if rec["name"] == "foreground":
            fixed |= (np.abs(path[:, 0, None] - heldout_x) < 3).any(axis=1)
        # At this shallow view, a one-pixel toe shift moves the ground intersection
        # several centimetres. Keep the dark, poorly resolved toe on the traced trend.
        strength = 1.0 if sign < 0 else 0.12
        path[:, 1] += strength * np.where(fixed, 0, correction)
    lift = np.maximum(0.001, cam.lips(front, foot))
    t = np.linspace(0, 1, 17)[:, None]
    uv = front * (1 - t[..., None]) + back * t[..., None]
    # Smoothly return the sheet to the working road plane at the far trace.
    z = lift * (1 - t * t * (3 - 2 * t))
    roof = cam.plane(uv, z)
    base = roof[0].copy()
    base[:, 2] = 0
    edge = np.stack([base, roof[0]])
    visible = outline(cam.plane(np.vstack([cam.project(base), back[::-1]])))
    footprint = outline(np.vstack([roof[0], roof[-1, ::-1]]))
    return (
        roof,
        edge,
        visible.union(footprint),
        dict(
            length_m=float(np.linalg.norm(np.diff(roof[0], axis=0), axis=1).sum()),
            max_apparent_rise_m=float(h.max()),
            original_station_rises_m=h.tolist(),
        ),
    )


def build(out):
    s, cam = load()
    source = HERE / "references/2559905.jpg"
    if hashlib.sha256(source.read_bytes()).hexdigest() != s["photo_sha256"]:
        raise ValueError("Source image digest")
    photo = np.asarray(Image.open(source).convert("RGB"))
    mesh = Mesh()
    top = mesh.material("Rain-wet photographed asphalt", photo, 0.47, 0, 0.6, lossless_color=True)
    side = mesh.material("Observed dark rupture faces", photo, 0.88, 0, 0.8, lossless_color=True)
    insert = mesh.material("Striped road insert", photo, 0.61, 0.13, 0.55, lossless_color=True)
    groups = {}

    def add(name, v, f, mat, background=False, vertical=False):
        v = v.reshape(-1, 3)
        a, b, c = v[f].transpose(1, 0, 2)
        cross = np.cross(b - a, c - a)
        keep = np.linalg.norm(cross, axis=1) > 1e-10
        f = f[keep].copy()
        if vertical:
            flip = (cross[keep] * (cam.eye - a[keep])).sum(axis=1) < 0
            f[flip] = f[flip, ::-1]
        else:
            f = face_up(v, f)
        groups.setdefault((name, mat, background), []).append((v, f))

    cut = []
    measures = {}
    edges = []
    roofs = []
    for rec in s["sheets"]:
        roof, edge, shadow, stats = sheet(
            cam, rec, photo, np.array(s["heldout_front_top_toe_px"])[:, 0, 0]
        )
        edges.append(edge)
        roofs.append(roof)
        measures[rec["name"]] = stats
        add("Detached pavement top", roof, grid_faces(*roof.shape[:2]), top)
        add("Detached pavement edges", edge, grid_faces(*edge.shape[:2]), side, vertical=True)
        cut.append(shadow)
    # Unknown space below lifted sheets stays unmeshed. The image does not resolve slots
    # and backing material separately, so the striped insert is one observed surface.
    plate = outline(cam.plane(s["ribbed_surface_px"]))
    plate_visible = plate.difference(unary_union(cut))
    pv, pf = patchwork(plate_visible)
    add("Photographed ribbed road insert", pv, pf, insert)
    boundary = outline(cam.plane(s["observed_footprint_px"]))
    ground = boundary.difference(unary_union([*cut, plate]))
    v, f = patchwork(ground)
    add("Fracture lip and exposed bed", v, f, top)
    # A narrow observed perimeter supplies a road colour reference during installation.
    apron = boundary.buffer(0.085, join_style=2).difference(boundary)
    v, f = patchwork(apron)
    add("Road adjoining the fracture", v, f, top, True)
    series = out / "series"
    folder = series / "assets/pothole/P581"
    cache = series / "mesh_cache"
    ref = series / "references/P581.jpg"
    ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, ref)
    bounds = np.array(apron.bounds).reshape(2, 2)
    span = np.max(abs(bounds), axis=0) * 2 + 0.04
    xy = grid(span, 0.012)
    ext = np.ptp(xy, axis=(0, 1))
    meta = dict(
        id="P581",
        family="gervis_rupture",
        category="pothole",
        state="uplifted_carriageway",
        seed=30581,
        label=s["label"],
        source=s["source"],
        source_author=s["author"],
        source_license=s["licence"],
        source_image="references/P581.jpg",
        source_sha256=s["photo_sha256"],
        surface_dimensions_m=ext.tolist(),
        sheets=measures,
        camera_eye_m=cam.eye.tolist(),
        nominal_cone_height_m=s["working_cone_height_m"],
    )
    field0 = dict(width=ext[0], length=ext[1], height=np.zeros(xy.shape[:2]))
    transfer = Mesh()
    atlas = portable_asphalt(photo, cam, ext)
    pavement = transfer.material(
        "Metric wet pavement swatches", atlas, 0.69, 0, 0.75, lossless_color=True
    )
    roof_image, roof_uv = roof_bands(photo, cam, roofs, atlas, ext)
    roof_mat = transfer.material(
        "Registered asphalt chips over metric swatches",
        roof_image,
        0.68,
        0,
        0.6,
        lossless_color=True,
    )
    road_rgb = np.median(atlas, axis=(0, 1))
    band_image, band_uv = face_bands(photo, cam, edges, road_rgb)
    binder = transfer.material(
        "Registered fracture face bands", band_image, 0.91, 0, 0.50, lossless_color=True
    )
    striped = transfer.material(
        "Observed striped insert",
        np.uint8(
            np.clip(
                photo * (road_rgb.mean() / np.median(photo[307:320, 280:313], axis=(0, 1)).mean()),
                0,
                255,
            )
        ),
        0.61,
        0.13,
        0.55,
        lossless_color=True,
    )
    portable_groups = {}
    for (name, mat, road), pieces in groups.items():
        for index, (v, f) in enumerate(pieces):
            photo_uv = texture_uv(cam.project(v), s["size_px"])
            if photo_uv.min() < 0 or photo_uv.max() > 1:
                raise ValueError("Texture outside source")
            xy_uv = v[:, :2] / ext + 0.5
            port_name = name
            port_mat = pavement
            if name == "Detached pavement top":
                xy_uv = roof_uv[index]
                port_mat = roof_mat
            elif name == "Detached pavement edges":
                xy_uv = band_uv[index]
                port_mat = binder
            elif name == "Photographed ribbed road insert":
                xy_uv = photo_uv
                port_mat = striped
            elif name == "Fracture lip and exposed bed":
                port_name = "Road adjoining the fracture"
            portable_groups.setdefault(
                (port_name, port_mat, port_name == "Road adjoining the fracture"), []
            ).append((v, f, xy_uv))
        vs = np.vstack([v for v, _ in pieces])
        offsets = np.cumsum([0] + [len(v) for v, _ in pieces[:-1]])
        fs = np.vstack([f + d for (_, f), d in zip(pieces, offsets)])
        mesh.add(name, vs, fs, texture_uv(cam.project(vs), s["size_px"]), mat, road)
    for (name, mat, road), pieces in portable_groups.items():
        offsets = np.cumsum([0] + [len(v) for v, _, _ in pieces[:-1]])
        transfer.add(
            name,
            np.vstack([v for v, _, _ in pieces]),
            np.vstack([f + d for (_, f, _), d in zip(pieces, offsets)]),
            np.vstack([uv for _, _, uv in pieces]),
            mat,
            road,
        )
    transfer.save(folder, cache, meta, field0)
    flatten_scene(folder / "asset.glb")
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp)
        mesh.save(stage / "field", stage / "cache", meta, field0)
        flatten_scene(stage / "field/asset.glb")
        shutil.copyfile(stage / "field/asset.glb", folder / "inspection.glb")
    surf = Surface(folder / "asset.glb")
    origin = xyz(xy, 1.0)
    down = np.broadcast_to([0, 0, -1.0], origin.shape)
    dist, part, _ = surf.cast(origin, down)
    hit = np.isfinite(dist)
    field = np.where(hit, 1 - dist, 0).astype("f4")
    # Cosine-weighted sky occlusion on visible road cells adjoining a lifted edge.
    receiver = (part == 1) & (distance_transform_edt(part != 13) * 0.012 < 0.75)
    o = xyz(xy[receiver], 0.002)
    blocked = np.zeros(len(o))
    weights = 0.0
    for i in range(24):
        z = (i + 0.5) / 24
        az = i * np.pi * (3 - np.sqrt(5))
        ray = np.array([np.cos(az) * np.sqrt(1 - z * z), np.sin(az) * np.sqrt(1 - z * z), z])
        d, _, _ = surf.cast(o, np.broadcast_to(ray, o.shape))
        blocked += z * (d < 2.0)
        weights += z
    sky = np.zeros(hit.shape, "f4")
    sky[receiver] = blocked / weights
    np.savez_compressed(
        folder / "surface.npz",
        receiver_sky_blocked=sky,
        return_to_grade_blend_m=0.035,
        height=field,
        alpha=hit,
        width=ext[0],
        length=ext[1],
        defect_mask=part > 1,
        installation_mask=part > 1,
        first_part=part,
        unknown_beneath_sheet=part == 13,
        registered_relief=True,
        opening_mask=np.zeros_like(hit),
        water_mask=np.zeros_like(hit),
        photographic_rgb=True,
    )
    row = make_one(
        folder,
        cache,
        point_limit=650000,
        kernel_factors=(0.74, 0.15),
        part_importance={"Detached pavement edges": 6, "Photographed ribbed road insert": 4},
    )
    for k, f in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("surface", "surface.npz"),
        ("metadata", "metadata.json"),
    ]:
        row[k] = "pothole/P581/" + f
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
