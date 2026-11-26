"""Build the photographed stepped casting, exposed front webs and milled-road excavation."""

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from camera import HERE, load

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
    remap,
    rings_faces,
    write_json,
    xyz,
)


def periodic(x, lo, hi):
    width = hi - lo
    return lo + width - abs((x - lo) % (2 * width) - width)


def camera_textures(photo, cam, s):
    # Sample an orthographic atlas of the top. Only safe interior source pixels
    # survive the occluder mask. Hidden texels use photographed tread on this lid.
    dim = np.array([cam.width + 0.14, cam.length + 0.08])
    height, width = 480, 340
    y, x = np.indices((height, width))
    xy = (np.stack([(x + 0.5) / width, (y + 0.5) / height], axis=-1) - 0.5) * dim * [1, -1]
    uv = cam.project(xyz(xy, cam.grade))
    clear = np.ones(photo.shape[:2], np.uint8) * 255
    for region in s["top_occluders"]:
        cv2.fillPoly(clear, [np.int32(region)], 0)
    poly = cam.project(xyz(cam.profile(), cam.grade))
    footprint = np.zeros_like(clear)
    cv2.fillPoly(footprint, [np.int32(np.rint(poly * 16))], 255, shift=4)
    clear = cv2.erode(clear & footprint, np.ones((3, 3), "u1"))
    confidence = remap(clear, uv) / 255
    # The donor window was checked on the rectified top, below the worker's hands.
    donor_xy = np.stack(
        [periodic(xy[:, :, 0], 0.015, 0.221), periodic(xy[:, :, 1], -0.475, -0.115)], axis=-1
    )
    donor_uv = cam.project(xyz(donor_xy, cam.grade))
    if remap(clear, donor_uv).min() < 250:
        raise ValueError("Occlusion in tread donor")
    measured = remap(photo, uv)
    donor = remap(photo, donor_uv)
    weight = cv2.GaussianBlur(confidence, (0, 0), 1.4) * confidence
    tex = np.uint8(np.rint(measured * weight[:, :, None] + donor * (1 - weight[:, :, None])))
    return tex, dim, weight


def patched_texture(photo, cam, anchor, z, span, period, obstructions):
    y, x = np.indices((480, 480))
    xy = (np.stack([(x + 0.5) / 480, (y + 0.5) / 480], -1) - 0.5) * span * [1, -1]
    donor = cam.plane([anchor], z)[0, :2]
    donor_xy = periodic(xy, -np.array(period) / 2, np.array(period) / 2) + donor
    fallback = remap(photo, cam.project(xyz(donor_xy, z)))
    allowed = np.zeros(photo.shape[:2], "u1")
    allowed[250:424, 3:355] = 255
    for ob in obstructions:
        cv2.fillPoly(allowed, [np.asarray(ob, "i4")], 0)
    allowed = cv2.erode(allowed, np.ones((5, 5), "u1"))
    pixels = cam.project(xyz(xy, z))
    weights = (
        cv2.remap(
            allowed,
            pixels[:, :, 0].astype("f4"),
            pixels[:, :, 1].astype("f4"),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        / 255
    )
    weights = (cv2.GaussianBlur(weights, (0, 0), 2) * weights)[:, :, None]
    return np.uint8(np.rint(remap(photo, pixels) * weights + fallback * (1 - weights)))


def casting_atlas(photo, s, cam, low, high):
    # Lay out one physical strip around the ironwork. Restore obscured parts
    # from a clean rust swatch instead of projecting foreground tools onto them.
    valid = np.zeros(photo.shape[:2], "u1")
    for poly in s["observed_casting_faces_px"]:
        cv2.fillPoly(valid, [np.int32(poly)], 255)
    for top, foot in s["front_rib_pixels"]:
        cv2.line(valid, tuple(np.int32(top)), tuple(np.int32(foot)), 0, 4)
    valid = cv2.GaussianBlur(valid, (5, 5), 0.7)
    edges = np.roll(high, -1, axis=0) - high
    lengths = np.linalg.norm(edges, axis=1)
    widths = np.maximum(5, np.rint(lengths * 420).astype(int))
    bands = []
    for i, width in enumerate(widths):
        j = (i + 1) % len(widths)
        rows, cols = np.indices((72, width))
        u = (cols + 0.5) / width
        v = 1 - (rows + 0.5) / 72
        floor = low[i] + u[:, :, None] * (low[j] - low[i])
        top = high[i] + u[:, :, None] * (high[j] - high[i])
        world = floor + v[:, :, None] * (top - floor)
        px = cam.project(world)
        # Preserve the swatch's physical scale on inferred faces, too.
        tx = periodic(u * lengths[i] * 170 + i * 7, 0, 22) + 123
        ty = periodic((world[:, :, 2] - low[i, 2]) * 165, 0, 8) + 346
        fallback = remap(photo, np.stack([tx, ty], axis=-1))
        normal = np.cross(edges[i], high[i] - low[i])
        weight = remap(valid, px) / 255 if normal @ (cam.eye - high[i]) > 0 else np.zeros_like(u)
        bands.append(
            np.uint8(
                np.rint(weight[:, :, None] * remap(photo, px) + (1 - weight[:, :, None]) * fallback)
            )
        )
    return np.hstack(bands), np.r_[0, widths.cumsum()] / widths.sum()


def sides(mesh, name, low, high, mat, uvfun):
    # Both rings are counterclockwise seen from above; outward faces wrap the seam.
    n = len(low)
    i = np.arange(n)
    j = (i + 1) % n
    v = np.vstack([low, high])
    f = np.vstack([np.c_[i, j, n + j], np.c_[i, n + j, n + i]])
    mesh.add(name, v, f, uvfun(v), mat)


def prism(mesh, name, poly, bottom, top, material, uv):
    low, high = xyz(poly, bottom), xyz(poly, top)
    n = len(poly)
    i = np.arange(n)
    j = (i + 1) % n
    fc = face_up(high, rings_faces([poly]))
    v = np.vstack([low, high])
    f = np.vstack([np.c_[i, j, n + j], np.c_[i, n + j, n + i], n + fc, fc[:, ::-1]])
    mesh.add(name, v, f, uv(v), material)


def centred_uv(v, span):
    return v[..., :2] / span + 0.5


def build(out):
    s, cam = load()
    source = HERE / "references/A565.jpg"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == s["photo_sha256"]
    photo = np.asarray(Image.open(source).convert("RGB"))
    tex, top_dim, weight = camera_textures(photo, cam, s)
    out.mkdir(exist_ok=True, parents=True)
    Image.fromarray(np.uint8(weight * 255 + 0.5)).save(out / "top-source-weight.png")
    Image.fromarray(tex).save(out / "top-atlas.png")
    mesh = Mesh()
    top = mesh.material(
        "Observed stepped lid and completed hidden tread",
        tex,
        0.69,
        0.26,
        0.62,
        lossless_color=True,
    )
    span = np.array(s["road_patch_model_m"])
    pavement = mesh.material(
        "Planed asphalt from Hensting Lane",
        patched_texture(
            photo,
            cam,
            s["source_ground_donor_origin_px"],
            0.0,
            span,
            [0.72, 0.48],
            s["terrain_obstructions_px"],
        ),
        0.96,
        0.0,
        1.25,
    )
    bed = mesh.material(
        "Loose road aggregate from Hensting Lane",
        patched_texture(
            photo,
            cam,
            s["source_bed_donor_origin_px"],
            -0.10,
            span,
            [0.58, 0.35],
            s["terrain_obstructions_px"],
        ),
        0.98,
        0,
        0.85,
    )
    # Front rust comes from a cleaned photograph of the *same* vertical face.
    # Only use its registered front strip on forward faces. Elsewhere map a
    # real rust swatch, including surfaces hidden by earth, hands or tools.
    outline = cam.profile()
    z = cam.grade
    feet = cam.ribs()
    left, upper, right, lower_edge = s["face_donor_px"]
    rust_chip = photo[upper:lower_edge, left:right]
    rust = mesh.material("Oxidised exposed web", rust_chip, 0.82, 0.25, 0.65, lossless_color=True)
    shoe = mesh.material("Dark packers below flange", np.uint8(rust_chip * 0.52), 0.95, 0.04, 0.12)
    uv_top = lambda v: centred_uv(v, top_dim)
    uv_ground = lambda v: centred_uv(v, span)
    uv_rust = lambda v: np.stack(
        [
            0.5 + 0.47 * np.sin(7 * v[:, 0] + 4 * v[:, 1]),
            np.clip((v[:, 2] + 0.11) / 0.16, 0.01, 0.99),
        ],
        axis=-1,
    )
    # Narrow unobserved seat clearance is deliberately smaller than the photo's resolution.
    frame_w = 0.013

    def inset(poly, amount):
        # Shift each edge inward by a fixed distance. This preserves the step widths.
        delta = np.roll(poly, -1, axis=0) - poly
        unit = delta / np.linalg.norm(delta, axis=1, keepdims=True)
        normal = unit[:, [1, 0]] * [-1, 1]
        prev = np.roll(normal, 1, axis=0)
        return poly + amount * (normal + prev) / (1 + (normal * prev).sum(axis=1, keepdims=True))

    seat = inset(outline, frame_w)
    lid = inset(outline, frame_w + 0.0028)
    face_polygon(mesh, "Cast ring and cover seat", xyz(outline, z), [xyz(seat, z)], top, uv_top)
    face_polygon(mesh, "Lid face", xyz(lid, z), [], top, uv_top)
    face_polygon(
        mesh, "Circular cover rebate", xyz(seat, z - 0.004), [xyz(lid, z - 0.004)], rust, uv_rust
    )
    sides(mesh, "Lid rolled edge", xyz(lid, z - 0.020), xyz(lid, z), rust, uv_rust)
    sides(
        mesh,
        "Cast seat riser and ledge",
        xyz(seat[::-1], z - 0.004),
        xyz(seat[::-1], z),
        rust,
        uv_rust,
    )

    # Width of the feet and apparent base height change across the near face.
    def lower(poly):
        return np.interp(poly[:, 0], feet[:, 0], feet[:, 2])

    body = inset(outline, 0.004)
    low = xyz(body, lower(body))
    high = xyz(outline, z)
    wall_atlas, breaks = casting_atlas(photo, s, cam, low, high)
    wall_mat = mesh.material(
        "Registered ironwork around the stepped perimeter",
        wall_atlas,
        0.79,
        0.25,
        0.62,
        lossless_color=True,
    )
    for i in range(len(outline)):
        j = (i + 1) % len(outline)
        v = np.stack([low[i], low[j], high[j], high[i]])
        u0, u1 = breaks[i : i + 2]
        mesh.add(
            "Registered front casting" if j == 0 else "Casting perimeter wall",
            v,
            [[0, 1, 2], [0, 2, 3]],
            [[u0, 0], [u1, 0], [u1, 1], [u0, 1]],
            wall_mat,
        )
    flange = inset(outline, -0.026)
    face_polygon(mesh, "Lower casting flange", xyz(flange, lower(flange)), [low], rust, uv_rust)
    sides(
        mesh,
        "Lower casting flange edge",
        xyz(flange, lower(flange) - 0.011),
        xyz(flange, lower(flange)),
        rust,
        uv_rust,
    )
    # Triangular cast webs flare outward at the bottom; their three measured feet
    # lie on the front of the model and are independent of the overlay thickness.
    gussets = []
    for k, (x, y, foot_z) in enumerate(feet):
        y0 = -cam.length / 2 + 0.005
        ring = np.array(
            [[x - 0.007, y0, z - 0.016], [x - 0.007, y, foot_z], [x - 0.007, y0, foot_z]]
        )
        v = np.vstack([ring, ring + [0.014, 0, 0]])
        f = np.array(
            [[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]]
        )
        a, b, c = v[f].transpose(1, 0, 2)
        flip = (np.cross(b - a, c - a) * (a - v.mean(0))).sum(1) < 0
        f[flip] = f[flip, ::-1]
        gussets.append((v, f + 6 * k))
    v = np.vstack([g[0] for g in gussets])
    mesh.add("Front casting gussets", v, np.vstack([g[1] for g in gussets]), uv_rust(v), rust)
    sub_z = feet[:, 2].min() - 0.011 - s["subgrade_below_flange_m"]
    for x in [-0.30, 0.30]:
        for y in [-0.36, 0.36]:
            width = 0.23
            d = 0.16
            poly = np.array(
                [
                    [x - width / 2, y - d / 2],
                    [x + width / 2, y - d / 2],
                    [x + width / 2, y + d / 2],
                    [x - width / 2, y + d / 2],
                ]
            )
            prism(mesh, "Level-setting shims", poly, sub_z, lower(poly) - 0.011, shoe, uv_rust)
    # Trace the visible road cut and close the rear under the work crew.
    cut = np.array(s["excavation_ccw_xy_m"])
    inner = cut * np.array([0.92, 0.94])
    face_polygon(
        mesh, "Road adjoining the fracture", rect(span), [xyz(cut, 0)], pavement, uv_ground
    )
    sides(
        mesh,
        "Fracture sidewall in surface coordinates",
        xyz(inner[::-1], sub_z),
        xyz(cut[::-1], 0),
        bed,
        uv_ground,
    )
    face_polygon(mesh, "Fracture lip and exposed bed", xyz(inner, sub_z), [], bed, uv_ground)
    # Multiple entries share material/semantic IDs. Merge them before exporting GLB and PLY.
    groups = {}
    for name, v, f, uv, mat, road in mesh.parts:
        group = groups.setdefault((name, mat, road), [])
        group.append((v, f, uv))
    mesh.parts = []
    for (name, mat, road), pieces in groups.items():
        offsets = np.cumsum([0] + [len(v) for v, _, _ in pieces[:-1]])
        mesh.add(
            name,
            np.vstack([v for v, _, _ in pieces]),
            np.vstack([f + i for (_, f, _), i in zip(pieces, offsets)]),
            np.vstack([u for _, _, u in pieces]),
            mat,
            road,
        )
    series = out / "series"
    folder = series / "assets/raised/A565"
    cache = series / "mesh_cache"
    xy = grid(span, 0.004)
    ext = np.ptp(xy, axis=(0, 1))
    meta = dict(
        id="A565",
        family="hensting_cover",
        category="raised",
        state="resurfacing_preparation",
        seed=29565,
        label=s["label"],
        source=s["photo"],
        source_license=s["licence"],
        source_author=s["author"],
        source_sha256=s["photo_sha256"],
        surface_dimensions_m=ext.tolist(),
        lid_front_width_model_m=cam.width,
        lid_length_model_m=cam.length,
        top_above_milled_grade_model_m=z,
        frame_foot_xyz_m=feet.tolist(),
        camera_eye_m=cam.eye.tolist(),
        design_target="45 mm planned overlay; 30 mm approximate removal reported by photographer",
        assumed="Eye 1.55 m over lid; 36 mm sensor width; hidden side mirrored; shim positions and bed inferred.",
    )
    mesh.save(folder, cache, meta, dict(width=ext[0], length=ext[1], height=np.zeros(xy.shape[:2])))
    flatten_scene(folder / "asset.glb")
    start = xyz(xy, 1.0)
    ray = np.broadcast_to([0.0, 0.0, -1.0], start.shape)
    dist, part, _ = Surface(folder / "asset.glb").cast(start, ray)
    hit = np.isfinite(dist)
    np.savez_compressed(
        folder / "surface.npz",
        width=ext[0],
        length=ext[1],
        height=np.where(hit, 1 - dist, 0).astype("f4"),
        alpha=hit,
        installation_mask=(part > 1),
        defect_mask=(part > 1),
        opening_mask=np.zeros_like(hit),
        water_mask=np.zeros_like(hit),
        photographic_rgb=True,
    )
    row = make_one(
        folder,
        cache,
        point_limit=420000,
        kernel_factors=(0.72, 0.14),
        part_importance={"Front casting gussets": 8.0, "Lid face": 5.0},
    )
    for role, file in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("surface", "surface.npz"),
        ("metadata", "metadata.json"),
    ]:
        row[role] = "raised/A565/" + file
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    build(parser.parse_args().out)
