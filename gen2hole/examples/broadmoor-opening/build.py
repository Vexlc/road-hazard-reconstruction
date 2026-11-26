"""Rebuild the open street chamber, the detached iron arc and the photographed cone placement."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar, least_squares

from camera import HERE, S, SIZE, SOURCE, RADIUS, circle, fit, plane, proj, rays

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


def texture(photo, pixels):
    q = (pixels + 0.5) * np.array(photo.shape[1::-1]) / SIZE - 0.5
    return cv2.remap(
        photo,
        q[..., 0].astype("f4"),
        q[..., 1].astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def strip(mesh, name, low, high, mat, inward=False):
    n = len(low)
    i = np.arange(n)
    v = np.vstack([low, low[:1], high, high[:1]])
    f = np.vstack([np.c_[i, i + 1, i + n + 2], np.c_[i, i + n + 2, i + n + 1]])
    u = np.linspace(0, 1, n + 1)
    uv = np.column_stack([np.tile(u, 2), np.repeat([0, 1], n + 1)])
    mesh.add(name, v, f[:, ::-1] if inward else f, uv, mat)


def rounded(poly, cut=0.13):
    # Trim the four photographed base corners with tangent quadratic arcs.
    loop = []
    for i, b in enumerate(poly):
        a = b + cut * (poly[i - 1] - b)
        c = b + cut * (poly[(i + 1) % len(poly)] - b)
        t = np.linspace(0, 1, 7)[:, None]
        loop.extend((1 - t) ** 2 * a + 2 * t * (1 - t) * b + t**2 * c)
    loop = np.array(loop)
    return loop if np.cross(loop, np.roll(loop, -1, axis=0)).sum() > 0 else loop[::-1]


def foot_fit(photo, k, base, center, tip, z, index):
    img = cv2.resize(photo, tuple(SIZE), interpolation=cv2.INTER_AREA)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    mask = ((h < 18) | (h > 170)) & (s > 110) & (v > 85)
    if index == 1:
        # The rusty arc touches the left base in the image and shares its hue.
        yy, xx = np.indices(mask.shape)
        mask &= ~((xx < 350) & (yy < 614 - 0.34 * (xx - 317)))
    # Manually bounded strips isolate rubber from the detached rusty casting.
    boxes = [(315, 593, 467, 652), (594, 530, 702, 571), (729, 632, 898, 692)]
    x0, y0, x1, y1 = boxes[index - 1]
    cols = np.arange(x0 + 2, x1 - 2, 3)
    points = []
    tip_px = proj(xyz(tip, z), k)
    for x in cols:
        hits = np.flatnonzero(mask[y0:y1, x])
        if len(hits) > 2:
            points.append([x, y0 + hits[-1]])
            if abs(x - tip_px[0]) > [52, 35, 49][index - 1] and y0 + hits[0] > y0:
                points.append([x, y0 + hits[0]])
    p = np.array(points)
    train = np.arange(len(p)) % 3 != 0

    def geometry(parameters):
        corners = plane(base + parameters[:8].reshape(4, 2), k)
        loop = rounded(corners, parameters[8])
        # Soft rubber foot is widest at the middle of its rounded edge.
        return loop, [
            xyz(loop * 0.985 + center * 0.015, 0.0002),
            xyz(loop, 0.007),
            xyz(loop * 0.975 + center * 0.025, 0.022),
        ]

    def distances(par):
        loop, levels = geometry(par)
        hull = cv2.convexHull(proj(np.vstack(levels), k).astype("f4"))[:, 0]
        return np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in p])

    def loss(par):
        # Keep the rear corners close to the photograph where the body hides the outline.
        corners = plane(base + par[:8].reshape(4, 2), k)
        opposite = corners[0] + corners[2] - corners[1] - corners[3]
        ab = corners[1] - corners[0]
        bc = corners[2] - corners[1]
        cosine = ab @ bc / (np.linalg.norm(ab) * np.linalg.norm(bc))
        return np.r_[
            distances(par)[train], par[:8] * 0.065, opposite * 85, cosine * 4, (par[8] - 0.12) * 1.5
        ]

    result = least_squares(
        loss,
        np.r_[np.full(8, 0.3), 0.12],
        bounds=(np.r_[[-24.0] * 8, 0.04], np.r_[[24.0] * 8, 0.28]),
        loss="soft_l1",
        diff_step=0.001,
        max_nfev=300,
    )
    poly, levels = geometry(result.x)
    error = distances(result.x)[~train]
    return (
        poly,
        levels,
        dict(
            heldout_count=len(error),
            heldout_rms_px=float(np.sqrt(np.mean(error**2))),
            refined_corners_px=(base + result.x[:8].reshape(4, 2)).tolist(),
        ),
    )


def make_cone(mesh, photo, k, cone, base, index):
    center = np.array(cone["center_xy_m"])
    tip = np.array(cone["top_xy_m"])
    h = cone["height_m"]
    rb, rt = cone["base_body_radius_m"], cone["top_radius_m"]
    poly, levels, foot_metrics = foot_fit(photo, k, base, center, tip, h, index)
    zbase = 0.022
    cone["foot_fit"] = foot_metrics
    rows, cols = np.indices((384, 640))
    angle = 2 * np.pi * (cols + 0.5) / 640
    t = 1 - (rows + 0.5) / 384
    eye = -k[3:5]
    heading = np.arctan2(*(eye - center)[::-1])
    # Preserve the front scuffs. For the rear, borrow the visible half of this same cone.
    front = heading + np.clip(np.arcsin(np.sin(angle - heading)), -1.28, 1.28)
    radius = rb * (1 - t) + rt * t
    xy = (
        center
        + (tip - center) * t[..., None]
        + radius[..., None] * np.stack([np.cos(front), np.sin(front)], -1)
    )
    q = proj(xyz(xy, zbase + t * (h - zbase)), k)
    color = texture(photo, q)
    mat = mesh.material(
        f"Orange rubber from cone {index}", color, 0.88, 0.01, 0.3, lossless_color=True
    )
    lo = circle(center, rb, zbase)
    high = circle(tip, rt, h)
    name = f"Traffic cone {index}"
    strip(mesh, name + " tapered body", lo, high, mat)
    # Foot color is sampled from the exposed front of this base; the cone cannot leak onto the foot.
    # A rectified base atlas keeps the actual scuffs and edge shadow. Only project
    # onto the photographed top outside the body; inner/hidden rubber borrows the front lip.
    yy, xx = np.indices((256, 256))
    span = 0.42
    offsets = (np.stack([xx + 0.5, 255.5 - yy], axis=-1) / 256 - 0.5) * span
    world = xyz(center + offsets, zbase)
    facing = eye - center
    facing /= np.linalg.norm(facing)
    angle = np.arctan2(offsets[..., 1], offsets[..., 0])
    donor_angle = np.arctan2(facing[1], facing[0]) + 0.32 * np.sin(angle)
    donor = center + 0.108 * np.stack([np.cos(donor_angle), np.sin(donor_angle)], -1)
    direct = texture(photo, proj(world, k))
    donor_rgb = texture(photo, proj(xyz(donor, zbase), k))
    norm = np.linalg.norm(offsets, axis=-1)
    front_mask = (offsets @ facing) / np.maximum(norm, 0.001)
    w = np.clip((norm - rb - 0.005) / 0.016, 0, 1) * np.clip((front_mask + 0.1) / 0.3, 0, 1)
    hsv = cv2.cvtColor(direct, cv2.COLOR_RGB2HSV)
    in_rubber = (
        ((hsv[..., 0] < 19) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 90)
    )
    w *= cv2.erode(in_rubber.astype("u1"), np.ones((3, 3), "u1"))
    rgb = np.uint8(np.rint(w[..., None] * direct + (1 - w[..., None]) * donor_rgb))
    foot = mesh.material(name + " moulded base", rgb, 0.83, 0.01, 0.18)
    uv = lambda v: 0.5 + (v[:, :2] - center) / span
    lower, middle, top = levels
    # Photograph the near lip vertically, mapping hidden perimeter positions into that same side.
    u, v = np.meshgrid(np.linspace(-8, 8, 128), np.linspace(0, 1, 64))
    foot_pixel = proj(xyz(lower[np.argmax((lower[:, :2] - center) @ facing), :2], 0), k)
    # The side wall's colour and shadow come from its own front edge at the original pixel scale.
    local_center = np.rint(foot_pixel).astype(int)
    edge_pixels = np.stack([local_center[0] + u * 0.28, local_center[1] - 1 - (1 - v) * 3], -1)
    side_rgb = texture(photo, edge_pixels)
    for suffix, low_band, high_band in [
        (" lower bevel", lower, middle),
        (" base edge", middle, top),
    ]:
        # Sample the visible bevel along each front-facing edge; rear-facing edges borrow
        # the near-side rubber instead of projecting the opposite background into them.
        tt, ss = np.indices((48, 512))
        tt = 1 - (tt + 0.5) / 48
        u = (ss + 0.5) / 512 * len(poly)
        ix = u.astype(int)
        frac = (u - ix)[..., None]
        jx = (ix + 1) % len(poly)
        sampled_low = low_band[ix] * (1 - frac) + low_band[jx] * frac
        sampled_high = high_band[ix] * (1 - frac) + high_band[jx] * frac
        world = sampled_low * (1 - tt[..., None]) + sampled_high * tt[..., None]
        sampled = texture(photo, proj(world, k))
        orange = cv2.cvtColor(sampled, cv2.COLOR_RGB2HSV)
        edge = high_band[jx, :2] - high_band[ix, :2]
        normal = edge[..., [1, 0]] * [1, -1]
        seen = (normal * (eye - world[..., :2])).sum(-1) > 0
        weight = seen & ((orange[..., 0] < 18) | (orange[..., 0] > 170)) & (orange[..., 1] > 105)
        fallback = cv2.resize(side_rgb, (512, 48))
        skin = np.where(weight[..., None], sampled, fallback)
        side_mat = mesh.material(name + suffix + " photograph", skin, 0.88, 0.01, 0.23)
        strip(mesh, name + suffix, low_band, high_band, side_mat)
    face_polygon(mesh, name + " foot", top, [lo], foot, uv)
    mesh.add(
        name + " sole", lower, face_up(lower, rings_faces([lower[:, :2]]))[:, ::-1], uv(lower), foot
    )
    u, v = np.meshgrid(np.linspace(-4, 4, 96), np.linspace(-1.1, 1.1, 32))
    # The upper surface shows a small inset. Its internal depth is not observable.
    inner = circle(tip, rt * 0.70, h - 0.0015)
    cap = mesh.material(
        name + " tip",
        texture(photo, proj(xyz(tip + np.stack([u * 0.0012, v * 0.003], axis=-1), h), k)),
        0.88,
        0.01,
        0.22,
    )
    strip(mesh, name + " tip bevel", high, inner, cap)
    face_polygon(mesh, name + " tip centre", inner, [], cap, uv)


def cylinder_depth(p, k, r):
    q = rays(np.asarray(p), k)
    eye = np.r_[-k[3:5], k[2]]
    a = (q[:, :2] ** 2).sum(1)
    b = q[:, :2] @ eye[:2]
    c = eye[:2] @ eye[:2] - r * r
    return eye + q * ((-b + np.sqrt(b * b - a * c)) / a)[:, None]


def road(mesh, photo, k, span, outer):
    yy, xx = np.indices((824, 1152))
    xy = (np.stack([(xx + 0.5) / 1152, (yy + 0.5) / 824], axis=-1) - 0.5) * span * [1, -1]
    pp = proj(xyz(xy, 0), k)
    working = cv2.resize(photo, tuple(SIZE), interpolation=cv2.INTER_AREA)
    h, s, v = cv2.split(cv2.cvtColor(working, cv2.COLOR_RGB2HSV))
    yy0, xx0 = np.indices(h.shape)
    obstacle = (
        ((h < 18) | (h > 170)) & (s > 83) & (v > 55) & (yy0 > 380) & (xx0 > 305) & (xx0 < 913)
    ).astype("u1")
    cv2.fillPoly(
        obstacle,
        [
            np.int32(
                [[257, 627], [319, 629], [354, 615], [355, 574], [328, 575], [323, 583], [261, 598]]
            )
        ],
        1,
    )
    cv2.fillPoly(obstacle, [np.rint(proj(outer, k)).astype("i4")], 1)
    obstacle = cv2.dilate(obstacle, np.ones((9, 9), "u1"))
    # Copy only measured pavement. Hidden support gets a metrically rectified clean asphalt patch.
    keep = cv2.remap(
        (1 - obstacle).astype("f4"),
        pp[..., 0].astype("f4"),
        pp[..., 1].astype("f4"),
        cv2.INTER_LINEAR,
    )
    mirror = lambda z, l: l - np.abs(z % (2 * l) - l)
    donor = np.stack(
        [0.12 + mirror(xy[..., 0] + 0.25, 0.36), -0.84 + mirror(xy[..., 1], 0.20)], axis=-1
    )
    if (
        cv2.remap(obstacle, proj(xyz(donor, 0), k).astype("f4"), None, cv2.INTER_NEAREST) > 0
    ).any():
        raise ValueError("The asphalt donor touches an obstacle")
    wt = (cv2.GaussianBlur(keep, (0, 0), 2) * keep)[..., None]
    rgb = np.uint8(
        np.rint(wt * texture(photo, pp) + (1 - wt) * texture(photo, proj(xyz(donor, 0), k)))
    )
    asphalt = mesh.material("Broadmoor cracked lane pavement", rgb, 0.95, 0.01, 0.73)
    face_polygon(
        mesh,
        "Road around the aperture",
        rect(span),
        [outer],
        asphalt,
        lambda q: q[:, :2] / span + 0.5,
    )


def chamber(mesh, photo, k):
    radius = RADIUS
    inner = S["bore_radius_assumed_m"]
    n = 384
    bearing = cylinder_depth(S["bearing_far_arc_px"], k, inner)
    wall = cylinder_depth(S["lower_wall_far_arc_px"], k, inner)
    seat_z = np.median(bearing[:, 2])
    t = np.arange(n) * 2 * np.pi / n
    limits = plane(np.asarray(S["chip_edge_angles_from_pixels"]), k)
    left, right = sorted(np.arctan2(limits[:, 1], limits[:, 0]))
    # The back rim has a local fracture down to the ledge. Keep its sharp asymmetric edges.
    missing = np.clip((t - left) / 0.012, 0, 1) * np.clip((right - t) / 0.015, 0, 1)
    crest = seat_z * 0.88 * missing

    def ring(r, z):
        return xyz(r * np.stack([np.cos(t), np.sin(t)], -1), z)

    outer = ring(radius + 0.005, 0)
    top = ring(radius, crest)
    lower = ring(radius, seat_z)
    seat = ring(inner, seat_z)
    wa = np.arctan2(wall[:, 1], wall[:, 0])
    order = np.argsort(wa)
    wall_lo = ring(inner, np.interp(t, wa[order], wall[order, 2], period=2 * np.pi))

    def iron(name, lo, hi):
        # Sample the far-side rust around the circumference, retaining radial and vertical shifts.
        row, col = np.indices((72, 512))
        t0 = 1 - (row + 0.5) / 72
        az = 2 * np.pi * (col + 0.5) / 512
        angle = np.pi / 2 + np.arcsin(0.82 * np.sin(az - np.pi / 2))
        i = np.rint(angle / (2 * np.pi) * n).astype(int) % n
        world = lo[i] * (1 - t0[..., None]) + hi[i] * t0[..., None]
        return mesh.material(name, texture(photo, proj(world, k)), 0.83, 0.30, 0.5)

    strip(mesh, "Cast ring and cover seat", outer, top, iron("Fractured ring crest", outer, top))
    strip(mesh, "Casting perimeter wall", lower, top, iron("Recessed riser", lower, top), True)
    strip(mesh, "Cast seat riser and ledge", lower, seat, iron("Bright bearing arc", lower, seat))
    strip(
        mesh,
        "Inspection shaft",
        wall_lo,
        seat,
        iron("Rusted sleeve in source light", wall_lo, seat),
        True,
    )
    bottom = ring(inner, -S["closure_visual_m"])
    dark = mesh.material("Photographed shaft shadow", photo[938:955, 934:960], 0.98, 0, 0.09)
    strip(mesh, "Cavity inferred closure", bottom, wall_lo, dark, True)
    face_polygon(mesh, "Shaft bottom", bottom, [], dark, lambda v: v[:, :2] + 0.5)
    return outer, dict(
        bearing_z_m=seat_z,
        visible_far_edge_xyz_m=wall.tolist(),
        notch_angle_rad=[left, right],
        nominal_radius_m=radius,
    )


def fragment(mesh, photo, k):
    front = plane(np.array(S["fragment_foot_front_px"]), k)
    tangent = np.gradient(front, axis=0)
    normal = tangent[:, [1, 0]] * [-1, 1]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    back = front + normal * S["fragment_between_faces_depth_assumed_m"]

    def height(pt, target):
        return minimize_scalar(
            lambda z: np.linalg.norm(proj(xyz(pt, z), k) - target),
            bounds=(0.002, 0.13),
            method="bounded",
        ).x

    flange = height(front[0], S["fragment_front_at_flange_px"])
    ridge = height(back[0], S["fragment_front_at_back_ridge_px"])
    # Sweep an L-section along the retained, visibly curved fragment. Its hidden end is truncated.
    section = [
        xyz(front, 0),
        xyz(front, flange),
        xyz(back - normal * 0.014, flange),
        xyz(back - normal * 0.014, ridge),
        xyz(back, ridge),
        xyz(back, 0),
    ]
    v = np.vstack(section)
    n = len(front)
    f = []
    for j in range(len(section)):
        a = j * n + np.arange(n - 1)
        b = ((j + 1) % len(section)) * n + np.arange(n - 1)
        f.extend(np.vstack([np.c_[a, a + 1, b + 1], np.c_[a, b + 1, b]]))
    # Cap the concave cross-section in its own 2D plane.
    section2 = np.array(
        [[0, 0], [0, flange], [0.037, flange], [0.037, ridge], [0.051, ridge], [0.051, 0]]
    )
    cap = rings_faces([section2]) * n
    f = np.vstack([f, cap[:, ::-1], cap + n - 1])
    # The profile is clockwise when viewed from +tangent.
    from trimesh import Trimesh

    body = Trimesh(v, f, process=False)
    if body.volume < 0:
        f = f[:, ::-1]
    src = photo[852:950, 382:546]
    mat = mesh.material("Photographed detached iron section", src, 0.88, 0.25, 0.44)
    q = proj(v, k) * 1.5 - [382, 852]
    uv = (np.clip(q / [164, 98], 0.005, 0.995)) * [1, -1] + [0, 1]
    mesh.add("Fracture debris iron arc", v, f, uv, mat)
    return dict(front_foot_xy_m=front.tolist(), flange_height_m=flange, back_ridge_height_m=ridge)


def build(out):
    k, cones, fit_errors = fit()
    photo = np.asarray(Image.open(SOURCE).convert("RGB"))
    mesh = Mesh()
    span = np.array([2.32, 1.72])
    outer, rim = chamber(mesh, photo, k)
    road(mesh, photo, k, span, outer)
    piece = fragment(mesh, photo, k)
    for j, (cone, base) in enumerate(zip(cones, S["cone_feet_px"]), 1):
        make_cone(mesh, photo, k, cone, np.array(base), j)
    series = out / "series"
    folder = series / "assets/missing/M571"
    cache = series / "mesh_cache"
    xy = grid(span, 0.004)
    ext = np.ptp(xy, axis=(0, 1))
    meta = dict(
        id="M571",
        family="broadmoor_opening",
        category="missing",
        state="missing_cover_three_cones",
        seed=29571,
        label="Broadmoor 机动车道缺盖井与三只锥桶",
        source=S["source"],
        source_author=S["author"],
        source_license=S["licence"],
        source_sha256=S["source_sha256"],
        surface_dimensions_m=ext.tolist(),
        camera_params=k.tolist(),
        focal_source="5.8 mm, 12515.56 px per inch, full sensor width 2816 px",
        cones=cones,
        registration=fit_errors,
        ring=rim,
        fragment=piece,
        visual_only_depth_m=S["closure_visual_m"],
    )
    mesh.save(folder, cache, meta, dict(width=ext[0], length=ext[1], height=np.zeros(xy.shape[:2])))
    flatten_scene(folder / "asset.glb")
    mesh.parts = [p for p in mesh.parts if p[0] not in {"Cavity inferred closure", "Shaft bottom"}]
    (folder / "observed.glb").write_bytes(mesh.glb_bytes())
    flatten_scene(folder / "observed.glb")
    origin = xyz(xy, 1)
    direction = np.broadcast_to([0, 0, -1.0], origin.shape)
    dist, part, _ = Surface(folder / "asset.glb").cast(origin, direction)
    support, _, _ = Surface(folder / "observed.glb").cast(origin, direction)
    hit = np.isfinite(dist)
    valid = np.isfinite(support)
    np.savez_compressed(
        folder / "surface.npz",
        width=ext[0],
        length=ext[1],
        height=np.where(hit, 1 - dist, 0).astype("f4"),
        alpha=hit,
        observed_depth_valid=valid,
        opening_mask=hit & ~valid,
        installation_mask=part > 1,
        defect_mask=part > 1,
        cone_mask=part == 25,
        water_mask=np.zeros_like(hit),
        photographic_rgb=True,
    )
    row = make_one(
        folder,
        cache,
        point_limit=410000,
        kernel_factors=(0.7, 0.16),
        part_importance={
            **{f"Traffic cone {i} tapered body": 4.0 for i in (1, 2, 3)},
            "Fracture debris iron arc": 6.0,
        },
    )
    for key, name in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("metadata", "metadata.json"),
        ("surface", "surface.npz"),
    ]:
        row[key] = "missing/M571/" + name
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
