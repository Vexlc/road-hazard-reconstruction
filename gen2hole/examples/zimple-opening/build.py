"""Construct the Zimple intersection field asset from its native-pixel survey."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from camera import Camera, HERE, S, circle, photograph

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    face_polygon,
    face_up,
    rings_faces,
    flatten_scene,
    grid,
    make_one,
    normalize,
    rect,
    write_json,
    xyz,
)


def swatch(xmin, ymin, xmax, ymax):
    return photo[ymin - offset[1] : ymax - offset[1], xmin - offset[0] : xmax - offset[0]]


def sample(p):
    p = np.asarray(p) - offset
    return cv2.remap(
        photo,
        p[..., 0].astype("f4"),
        p[..., 1].astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def loop_strip(mesh, name, low, high, mat, inward=False):
    n = len(low)
    i = np.arange(n)
    v = np.vstack([low, low[:1], high, high[:1]])
    f = np.vstack([np.c_[i, i + 1, i + n + 2], np.c_[i, i + n + 2, i + n + 1]])
    uv = np.c_[np.tile(np.linspace(0, 1, n + 1), 2), np.repeat([0, 1], n + 1)]
    mesh.add(name, v, f[:, ::-1] if inward else f, uv, mat)


def chamber(mesh, cam):
    r = cam.radius

    def ring(rad, z):
        return xyz(circle(rad), z)

    seat_z = float(np.median(cam.cylinder(S["shaft_shelf_far_px"], r - 0.010)[:, 2]))
    lip = mesh.material("Cast iron lip", swatch(2898, 2944, 2928, 2949), 0.8, 0.35, 0.5)
    shelf = mesh.material("Bearing face", swatch(2904, 2949, 2928, 2952), 0.85, 0.24, 0.5)
    wall = mesh.material("Dark casting", swatch(2923, 2956, 2970, 2962), 0.88, 0.17, 0.4)
    loop_strip(mesh, "Cast ring and cover seat", ring(r + 0.005, 0), ring(r, -0.001), lip)
    loop_strip(mesh, "Casting perimeter wall", ring(r, seat_z), ring(r, -0.001), wall, True)
    loop_strip(
        mesh, "Cast seat riser and ledge", ring(r, seat_z), ring(r - 0.016, seat_z - 0.002), shelf
    )
    # End the measured-depth mesh at the ledge. Only the visual product needs a deep dark pocket.
    deep = ring(r - 0.016, -S["visual_only_closure_m"])
    mat = mesh.material("Unresolved shaft shadow", swatch(3030, 2956, 3060, 2974), 0.99, 0, 0.05)
    loop_strip(mesh, "Cavity inferred closure", deep, ring(r - 0.016, seat_z - 0.002), mat, True)
    disk = face_up(deep, rings_faces([deep[:, :2]]))
    mesh.add("Shaft bottom", deep, disk, deep[:, :2] + 0.5, mat)
    return ring(r + 0.005, 0), seat_z


def receiver_lighting(mesh, extent):
    import open3d as o3d

    # Only raised apparatus obstructs light arriving above the local ground plane.
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    for name, v, f, *_ in mesh.parts:
        if name.startswith("Delineator ") or name == "Yellow braided line":
            scene.add_triangles(o3d.core.Tensor(v.astype("f4")), o3d.core.Tensor(f.astype("u4")))
    xy = grid(extent, 0.01)
    o = xyz(xy, 0.001).astype("f4")

    def occlusion(ray):
        rays = np.concatenate((o, np.broadcast_to(ray, o.shape)), axis=-1)
        return scene.test_occlusions(
            o3d.core.Tensor(rays.astype("f4")), tfar=2.1, nthreads=2
        ).numpy()

    blocked = np.zeros(o.shape[:2], "f4")
    for i in range(32):
        z = (i + 0.5) / 32
        az = i * np.pi * (3 - np.sqrt(5))
        ray = [np.cos(az) * np.sqrt(1 - z * z), np.sin(az) * np.sqrt(1 - z * z), z]
        blocked += z * occlusion(ray) / 16
    # Field photograph casts the post shadow leftward behind the ballast.
    direct = occlusion(normalize(np.array([0.54, -0.38, 1.0]))).astype("f4")
    return blocked, cv2.GaussianBlur(direct, (5, 5), 0.75)


def pavement(mesh, cam, outer, lighting, extent):
    size = np.array([2.80, 2.70])
    rows, cols = np.indices((1024, 1152))
    xy = np.stack(
        [((cols + 0.5) / 1152 - 0.5) * size[0], (0.5 - (rows + 0.5) / 1024) * size[1]], axis=-1
    )
    # Reproject cleared road from the original wherever it is visible. In the shadow,
    # post and rope footprints take real ground swatches from the same intersection.
    pix = cam.project(xyz(xy, 0))
    x, y = pix.transpose(2, 0, 1)

    def smooth(d, width):
        t = np.clip(d / width, 0, 1)
        return t * t * (3 - 2 * t)

    border = np.minimum.reduce([x - offset[0], S["crop_xyxy"][2] - 1 - x, y - 2650, 3179 - y])
    marker_distance = np.maximum.reduce([2844 - x, x - 3315, y - 3007])
    picks = np.array(S["rope_xyz_constraints_px_zm"])
    road_line = np.interp(x, picks[:-14, 0], picks[:-14, 1])
    rope_distance = np.maximum(3160 - x, abs(y - road_line) - 24)
    use = smooth(border, 42) * smooth(marker_distance, 26) * smooth(rope_distance, 17)

    def reflected(q):
        return 1 - abs((q % 2) - 1)

    period = np.array([0.32, 0.38])
    tile = reflected(xy / period) - 0.5
    centers = [[2740, 3057], [3584, 3037], [3970, 2990]]
    weights = [
        0.5 + 0.5 * np.cos(4 * xy[..., 0] + 3 * xy[..., 1]),
        0.5 + 0.5 * np.cos(3 * xy[..., 0] - 5 * xy[..., 1]),
    ]
    colors = []
    for i, center in enumerate(centers):
        t = np.roll(tile, i % 2, axis=-1)
        pos = cam.plane(center)[:2] + t * period
        colors.append(sample(cam.project(xyz(pos, 0))).astype(float))
    fill = colors[0] * weights[0][..., None] + colors[1] * (1 - weights[0][..., None])
    fill = fill * (0.5 + 0.5 * weights[1][..., None]) + colors[2] * (
        0.5 - 0.5 * weights[1][..., None]
    )
    tex = fill * (1 - use[..., None]) + sample(pix) * use[..., None]
    uv = (xy / extent * [1, -1] + 0.5) * (np.array(lighting[0].shape[::-1]) - 1)
    sky, sun = [
        cv2.remap(a, uv[..., 0].astype("f4"), uv[..., 1].astype("f4"), cv2.INTER_LINEAR)
        for a in lighting
    ]
    tex *= np.power(np.maximum(0.08, 0.47 * (1 - 0.7 * sky) + 0.53 * (1 - sun)), 1 / 2.2)[..., None]
    mat = mesh.material("Rectified intersection gravel", tex, 0.96, 0, 0.75)
    # All vertices lie in one plane and the rectified UVs are affine. Preserve
    # the exact circular aperture without subdividing over a million coplanar faces.
    perimeter = rect(size)
    v = np.vstack((perimeter, outer))
    faces = face_up(v, rings_faces([perimeter[:, :2], outer[:, :2]]))
    mesh.add("Road around the aperture", v, faces, v[:, :2] / size + 0.5, mat, road=True)


def base(mesh, cam):
    top = 0.116
    poly = cam.plane(S["base_top_outline_px"], top)[:, :2]
    holes = [cam.plane(a, top)[:, :2] for a in S["base_handles_px"]]

    def ccw(x):
        return x if np.cross(x, np.roll(x, -1, axis=0)).sum() > 0 else x[::-1]

    poly = ccw(poly)
    holes = [ccw(a) for a in holes]
    mid = cam.plane(S["marker_contact_center_px"])[:2]
    surf = swatch(2845, 2814, 3310, 2877).copy()
    # Mask the post/tape and its hard shadow in the plane material. The free-standing
    # meshes carry that information. Sample neighbouring clear top rubber at equal rows.
    for j in range(len(surf)):
        # Neutral moulded rubber sampled inside the upper face avoids importing the
        # post silhouette or sky onto portions hidden by the marker in the field image.
        for x in range(130, 354):
            a = np.clip(min(x - 130, 354 - x) / 16, 0, 1)
            rub = swatch(3168, 2846, 3208, 2854)[j % 8, x % 40]
            surf[j, x] = surf[j, x] * (1 - a) + rub * a
    topmat = mesh.material("Black ballast top rubber", surf, 0.94, 0, 0.7)
    sd = mesh.material("Moulded rubber edge", swatch(2900, 2863, 2960, 2874), 0.96, 0, 0.7)
    dark = mesh.material("Recessed ballast support", swatch(2976, 2886, 3018, 2896), 0.98, 0, 0.4)
    uv = lambda v: (cam.project(v) - [2845, 2814]) / [465, -63] + [0, 1]
    face_polygon(
        mesh, "Delineator ballast top", xyz(poly, top), [xyz(a, top) for a in holes], topmat, uv
    )
    for i, a in enumerate(holes):
        loop_strip(
            mesh, "Delineator ballast slot " + str(i), xyz(a, top - 0.043), xyz(a, top), sd, True
        )
    low = 0.073
    loop_strip(
        mesh,
        "Delineator ballast bevel",
        xyz(mid + (poly - mid) * 1.018, top - 0.009),
        xyz(poly, top),
        sd,
    )
    loop_strip(
        mesh,
        "Delineator ballast edge",
        xyz(mid + (poly - mid) * 1.018, low),
        xyz(mid + (poly - mid) * 1.018, top - 0.009),
        sd,
    )
    # Underside has the same two genuine through slots.
    before = len(mesh.parts)
    face_polygon(
        mesh,
        "Delineator ballast underside",
        xyz(mid + (poly - mid) * 1.018, low),
        [xyz(a, low) for a in holes],
        dark,
        lambda v: v[:, :2],
    )
    name, v, f, uv, mat, flag = mesh.parts[before]
    mesh.parts[before] = (name, v, f[:, ::-1], uv, mat, flag)
    # Exposed lower support is recessed and fenestrated. The front has a daylight slit.
    for i, px in enumerate(
        [
            [[2920, 2907], [2988, 2914], [2975, 2924], [2918, 2919]],
            [[3110, 2911], [3182, 2909], [3192, 2917], [3107, 2924]],
        ]
    ):
        footprint = ccw(cam.plane(px)[:, :2])
        loop_strip(
            mesh, "Delineator ballast foot " + str(i), xyz(footprint, 0), xyz(footprint, low), dark
        )
        face_polygon(
            mesh,
            "Delineator ballast foot cap " + str(i),
            xyz(footprint, low),
            [],
            dark,
            lambda v: v[:, :2],
        )
    # The lower front rail is visible beneath the ballast with a narrow
    # daylight window above it. Its plan depth is inferred from the two feet.
    rail_px = [
        [2918, 2918],
        [2943, 2922],
        [3050, 2927],
        [3137, 2928],
        [3187, 2922],
        [3195, 2918],
        [3179, 2913],
        [3070, 2914],
        [2978, 2911],
        [2926, 2911],
    ]
    rail = ccw(cam.plane(rail_px, 0.009)[:, :2])
    loop_strip(mesh, "Delineator ballast lower rail", xyz(rail, 0.001), xyz(rail, 0.019), dark)
    face_polygon(
        mesh, "Delineator ballast lower rail top", xyz(rail, 0.019), [], dark, lambda v: v[:, :2]
    )
    return mid, top


def post(mesh, cam, mid, top):
    rows = np.array(S["marker_body_rows_y_left_right"], float)[::-1]
    rows = np.row_stack(([2851, 3035, 3133], rows))
    ys, left, right = rows.T
    z = cam.upright_height(mid, ys)
    z[0] = top
    xy = cam.plane(np.c_[(left + right) / 2, ys], z)[:, :2]
    xy[:, 1] = mid[1]
    radius = (cam.plane(np.c_[right, ys], z)[:, 0] - cam.plane(np.c_[left, ys], z)[:, 0]) / 2
    axis = normalize(cam.eye[:2] - mid)
    lateral = np.array([-axis[1], axis[0]])
    angles = np.linspace(0, 2 * np.pi, 97)
    dirs = np.outer(np.cos(angles), axis) + np.outer(np.sin(angles), lateral)
    # Rotate azimuth counterclockwise and unroll the front photograph across both sides.
    u = np.linspace(0, 1, 512)
    yp = np.linspace(ys[-1], ys[0], 1024)
    le = np.interp(yp, ys[::-1], left[::-1])
    ri = np.interp(yp, ys[::-1], right[::-1])
    x = (le + ri)[:, None] / 2 + np.sin(2 * np.pi * u)[None, :] * (ri - le)[:, None] * 0.455
    tex = sample(np.stack(np.broadcast_arrays(x, np.minimum(yp, 2782)[:, None]), axis=-1))
    mat = mesh.material("Scuffed green post and two reflective bands", tex, 0.62, 0, 0.4)
    v = np.stack([xyz(c + r * dirs, h) for c, r, h in zip(xy, radius, z)])
    uv = np.stack(
        np.broadcast_arrays(
            np.linspace(0, 1, 97)[None, :], ((ys[0] - ys) / (ys[0] - ys[-1]))[:, None]
        ),
        axis=-1,
    )
    i, j = np.indices((len(rows) - 1, 96))
    a = (i * 97 + j).ravel()
    f = np.vstack((np.c_[a, a + 1, a + 98], np.c_[a, a + 98, a + 97]))
    mesh.add("Delineator green body", v, f, uv, mat)
    cap = mesh.material("Rounded post tip", swatch(3070, 2126, 3077, 2131), 0.68, 0, 0.2)
    face_polygon(mesh, "Delineator tip", v[-1, :-1], [], cap, lambda w: w[:, :2])
    return float(z[-1])


def tape(mesh, cam, mid):
    # Use the visible folded, printed CAUTION band around the foot of the post.
    rows, cols = np.indices((88, 512))
    t = cols / 511 * 2 * np.pi
    h = rows / 87
    x = 3080 + np.sin(t) * 72
    y = 2791 + h * (49 + 5 * np.sin(t)) - 4 * np.sin(t * 2)
    tex = sample(np.stack([x, y], axis=-1))
    mat = mesh.material("Photographed caution tape", tex, 0.73, 0, 0.45)
    t = np.linspace(0, 2 * np.pi, 97)
    axis = normalize(cam.eye[:2] - mid)
    tangent = np.array([-axis[1], axis[0]])
    dirs = np.outer(np.cos(t), axis) + np.outer(np.sin(t), tangent)
    q = np.maximum(0, np.sin(t)) ** 6
    zlo = 0.125 + 0.009 * np.sin(t * 2 + 0.5)
    rad = 0.103 + 0.019 * np.sin(3 * t + 0.2) + 0.035 * q
    v = np.stack(
        (
            xyz(mid + dirs * rad[:, None], zlo),
            xyz(mid + dirs * (rad * 0.92)[:, None], zlo + 0.048 + 0.016 * np.cos(2 * t)),
        )
    )
    a = np.arange(96)
    mesh.add(
        "Delineator caution tape",
        v,
        np.vstack([np.c_[a, a + 1, a + 98], np.c_[a, a + 98, a + 97]]),
        np.stack(np.meshgrid(np.linspace(0, 1, 97), [0, 1]), -1),
        mat,
    )


def rope(mesh, cam):
    picks = np.array(S["rope_xyz_constraints_px_zm"])
    path = cam.plane(picks[:, :2], picks[:, 2])
    chord = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    station = np.linspace(0, chord[-1], int(chord[-1] / 0.005) + 1)
    center = PchipInterpolator(chord, path)(station)
    forward = normalize(np.gradient(center, axis=0))
    horizontal = normalize(np.cross(forward, [0, 0, 1]))
    normal = normalize(np.cross(forward, horizontal))
    # Rope width is roughly 12–18 px near the rim. Projected diameter uses a 22 mm hypothesis.
    t = np.linspace(0, 2 * np.pi, 25)
    angle = np.broadcast_to(t, (len(station), len(t)))
    radius = 0.011 * (1 + 0.08 * np.cos(angle * 3 - station[:, None] * 2 * np.pi / 0.042))
    v = center[:, None, :] + radius[..., None] * (
        np.cos(angle)[..., None] * horizontal[:, None, :]
        + np.sin(angle)[..., None] * normal[:, None, :]
    )
    # Bake the measured fibre grain along the complete line, including the two
    # folded segments at the far end. The enclosed UV domain also suits splat export.
    ntex = 3072
    mark = np.linspace(0, station[-1], ntex)
    p = cam.project(PchipInterpolator(chord, path)(mark))
    tangent = normalize(np.gradient(p, axis=0))
    image_normal = tangent[:, [1, 0]] * [-1, 1]
    w = 4 * np.sin(np.linspace(0, 2 * np.pi, 64))[:, None, None]
    tex = sample(p[None, :, :] + image_normal[None, :, :] * w)
    mat = mesh.material("Yellow braided line", tex, 0.91, 0, 0.65)
    a = (np.arange(len(station) - 1)[:, None] * len(t) + np.arange(len(t) - 1)).ravel()
    f = np.vstack([np.c_[a, a + 1, a + len(t) + 1], np.c_[a, a + len(t) + 1, a + len(t)]])
    # Around the cross-section then forward is outward.
    mesh.add(
        "Yellow braided line",
        v,
        f,
        np.stack(
            np.broadcast_arrays(station[:, None] / station[-1], t[None, :] / (2 * np.pi)), axis=-1
        ),
        mat,
    )
    return dict(
        traced_length_m=float(np.linalg.norm(np.diff(center, axis=0), axis=1).sum()),
        visible_endpoints_xyz_m=center[[0, -1]].tolist(),
    )


def build(out):
    global photo, offset
    photo = photograph()
    offset = np.array(S["crop_xyxy"][:2])
    cam = Camera()
    mesh = Mesh()
    outer, seat = chamber(mesh, cam)
    mid, top = base(mesh, cam)
    height = post(mesh, cam, mid, top)
    tape(mesh, cam, mid)
    line = rope(mesh, cam)
    series = out / "series"
    folder = series / "assets/missing/M582"
    cache = series / "mesh_cache"
    vertices = np.vstack([p[1] for p in mesh.parts])
    half = np.maximum(np.max(abs(vertices[:, :2]), axis=0) + 0.035, [1.4, 1.35])
    xy = grid(np.ceil(half * 100) * 0.02, 0.005)
    ext = np.ptp(xy, axis=(0, 1))
    lighting = receiver_lighting(mesh, ext)
    pavement(mesh, cam, outer, lighting, ext)
    meta = dict(
        id="M582",
        family="zimple_opening",
        category="missing",
        state="open_marked_intersection",
        seed=29582,
        label="Zimple 街口缺盖井、警示柱与黄色绳索",
        source=S["source"],
        source_author=S["author"],
        source_license=S["license"],
        source_sha256=S["original_sha256"],
        surface_dimensions_m=ext.tolist(),
        assumed_outer_diameter_m=cam.radius * 2,
        source_eye_m=cam.eye.tolist(),
        source_focal_px=cam.f.tolist(),
        marker_height_m=height,
        bearing_height_m=seat,
        rope=line,
        visual_only_depth_m=S["visual_only_closure_m"],
    )
    mesh.save(folder, cache, meta, dict(width=ext[0], length=ext[1], height=np.zeros(xy.shape[:2])))
    flatten_scene(folder / "asset.glb")
    mesh.parts = [p for p in mesh.parts if p[0] not in {"Shaft bottom", "Cavity inferred closure"}]
    (folder / "observed.glb").write_bytes(mesh.glb_bytes())
    flatten_scene(folder / "observed.glb")
    origins = xyz(xy, 2)
    down = np.broadcast_to([0.0, 0.0, -1.0], origins.shape)
    d, part, _ = Surface(folder / "asset.glb").cast(origins, down)
    observed, _, _ = Surface(folder / "observed.glb").cast(origins, down)
    hit = np.isfinite(d)
    np.savez_compressed(
        folder / "surface.npz",
        width=ext[0],
        length=ext[1],
        height=np.where(hit, 2 - d, 0).astype("f4"),
        alpha=hit,
        first_part=part,
        observed_depth_valid=np.isfinite(observed),
        opening_mask=hit & ~np.isfinite(observed),
        defect_mask=part > 1,
        installation_mask=part > 1,
        water_mask=np.zeros_like(hit),
        photographic_rgb=True,
        receiver_sky_blocked=lighting[0],
        cast_receiver_sunlight=True,
        registered_relief=True,
    )
    row = make_one(
        folder,
        cache,
        point_limit=430000,
        kernel_factors=(0.70, 0.16),
        part_importance={
            "Yellow braided line": 9.0,
            "Delineator caution tape": 5.0,
            "Delineator green body": 4.0,
        },
    )
    for k, f in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("metadata", "metadata.json"),
        ("surface", "surface.npz"),
    ]:
        row[k] = "missing/M582/" + f
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
