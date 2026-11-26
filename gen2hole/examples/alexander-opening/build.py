"""Build the two-view displaced-cover scene, undercut shell, slotted post and sensor surfaces."""

import argparse, hashlib
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from geometry import HERE, S, SIZE, RADIUS, fit, unit
from fixtures import circle, marker_pixel, wall_pixel, preview, make_step, step_path, delineator
from stereo import solve

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    flatten_scene,
    grid,
    grid_faces,
    make_one,
    rect,
    rings_faces,
    face_up,
    write_json,
    xyz,
)
import open3d as o3d


def orient(v, f, d):
    a, b, c = v[f].transpose(1, 0, 2)
    flip = (np.cross(b - a, c - a) * d).sum(-1) < 0
    f = f.copy()
    f[flip] = f[flip, ::-1]
    return f


def photographs():
    result = []
    for name in ("detail", "street"):
        path = HERE / "references" / (name + ".jpg")
        if hashlib.sha256(path.read_bytes()).hexdigest() != S[name]["source_sha256"]:
            raise ValueError("Original photograph checksum mismatch")
        im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        def sample(q, im=im):
            p = ((np.asarray(q) + 0.5) * im.shape[1::-1] / SIZE - 0.5).astype("f4")
            return cv2.remap(
                im, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )

        result.append(sample)
    return result


def lattice(mesh, name, v, mat, uv, towards):
    f = grid_faces(*v.shape[:2])
    v = v.reshape(-1, 3)
    mesh.add(name, v, orient(v, f, towards(v[f].mean(1))), uv, mat)


def cyl_grid(depths, radii, n, seam=True):
    a = np.linspace(0, 2 * np.pi, n, endpoint=seam)
    return xyz(
        np.stack((np.cos(a), np.sin(a)), -1)[None] * np.array(radii)[:, None, None],
        -np.array(depths)[:, None],
    )


def interior(mesh, cams, reg, photos, stereo):
    bore = reg["bore_radius"]
    seat = -reg["seat_z"]
    depths = np.array([seat, 0.075, 0.18, 0.34, 0.52, 0.68, 0.84, 1.0, 1.22, 1.48])
    prior = np.array([bore, bore + 0.003, 0.30, 0.36, 0.43, 0.52, 0.60, 0.68, 0.75, 0.79])
    # Use only two-way consistent pixels well inside both lips. Never fit the barrel
    # to the interior step, or to glints on the bearing edge.
    cloud = []
    for i in (0, 1):
        use = stereo["valid" + str(i)]
        y, x = np.where(use)
        p = np.c_[x, y] + stereo["offset" + str(i)]
        v = cams[i].plane(p, stereo["z" + str(i)][use])
        keep = np.ones(len(v), bool)
        for j in (0, 1):
            uv = cams[j].project(v)
            h = reg["traces"][j]["bore"].astype("f4")
            keep &= np.array([cv2.pointPolygonTest(h, tuple(q), True) > 10 for q in uv])
            region = wall_pixel(
                (
                    [[117, 49], [213, 49], [213, 94], [117, 94]]
                    if j == 0
                    else [[46, 118], [98, 118], [98, 191], [46, 191]]
                ),
                j,
            )
            keep &= ~((uv >= region.min(0)) & (uv <= region.max(0))).all(1)
        cloud.extend(v[keep])
    cloud = np.array(cloud)
    # Sparse shared masonry constrains an otherwise extrapolated flaring envelope.
    # Count spatially distinct 2-cm voxels instead of pooling correlated neighbouring pixels.
    vox = np.unique(np.floor(cloud / 0.02).astype(int), axis=0, return_index=True)[1]
    v = cloud[vox]
    r = np.linalg.norm(v[:, :2], axis=1)
    dz = -v[:, 2]
    radii = prior.copy()
    for j in range(2, len(depths) - 1):
        sel = (abs(dz - depths[j]) < 0.12) & (abs(r - np.interp(dz, depths, prior)) < 0.12)
        if sel.any():
            offset = np.median(r[sel] - np.interp(dz[sel], depths, prior))
            radii[j] += offset * min(sel.sum() / 8, 0.75)
    radii = np.maximum.accumulate(radii)
    profile = PchipInterpolator(depths, radii)
    rung, zr, _ = step_path(cams)

    # Free space has direct evidence too. Carve a recess around both sightlines to
    # the shared rung. Otherwise a rotationally symmetric cone hides it entirely.
    def envelope(d, width):
        theta = np.linspace(0, 2 * np.pi, width)
        rr = np.broadcast_to(profile(d)[:, None], (len(d), width)).copy()
        for c in cams:
            for point in rung:
                v = point - c.eye
                cross = c.eye + v * ((-d - c.eye[2]) / v[2])[:, None]
                angle = np.arctan2(cross[:, 1], cross[:, 0])
                delta = np.arctan2(
                    np.sin(theta[None] - angle[:, None]), np.cos(theta[None] - angle[:, None])
                )
                rad = np.linalg.norm(cross[:, :2], axis=1)
                excess = np.maximum(0, rad + 0.035 - profile(d))
                strength = np.clip((d - seat) / 0.045, 0, 1) * np.clip(
                    (0.77 - d) / (0.77 + zr - 0.025), 0, 1
                )
                rr = np.maximum(
                    rr,
                    profile(d)[:, None]
                    + (strength * excess)[:, None] * np.exp(-0.5 * (delta / 0.17) ** 2),
                )
        return xyz(rr[..., None] * np.stack([np.cos(theta), np.sin(theta)], -1), -d[:, None])

    def wall_tex(p):
        accum = np.zeros(p.shape)
        total = np.zeros(p.shape[:-1])
        for i in (1, 0):
            cam = cams[i]
            q = cam.project(p)
            weights = np.ones(p.shape[:-1])
            for z, r in [(0, RADIUS - 0.018), (-seat, bore)]:
                ray = p - cam.eye
                crossing = cam.eye + ray * ((z - cam.eye[2]) / ray[..., 2])[..., None]
                weights *= np.clip((r - np.linalg.norm(crossing[..., :2], axis=-1)) / 0.012, 0, 1)
            # Exclude the visible rusty step from the projected masonry.
            coords = wall_pixel([[140, 56], [208, 92]] if i == 0 else [[49, 124], [95, 187]], i)
            near = (q - coords.min(0)).min(-1)
            far = (coords.max(0) - q).min(-1)
            weights *= np.clip(1 - np.minimum(near, far) / 4, 0, 1)
            weights *= 3 if i == 0 else 1
            color = photos[i](q)
            accum += color * weights[..., None]
            total += weights
        # Local unlit masonry swatch for the angular sectors that neither camera sees.
        yy, xx = np.indices(total.shape)
        t = np.stack([1 - abs((xx / 89 % 2) - 1), 1 - abs((yy / 67 % 2) - 1)], -1)
        filler = photos[0](wall_pixel(t * [46, 26] + [112, 111], 0)) * 0.55
        return (accum + filler * np.maximum(0, 0.20 - total)[..., None]) / np.maximum(total, 0.20)[
            ..., None
        ]

    strip_uv = lambda h, w: np.stack(
        np.broadcast_arrays(np.linspace(0, 1, w), np.linspace(1, 0, h)[:, None]), -1
    )
    zz = np.linspace(seat, depths[-1], 96)
    tt = np.linspace(seat, depths[-1], 640)
    tex = wall_tex(envelope(tt, 1152))
    mat = mesh.material("Two-view masonry", tex, 0.95, 0, 0.75)
    inward = lambda c: np.c_[-c[:, :2], np.zeros(len(c))]
    lattice(
        mesh, "Alexander masonry envelope", envelope(zz, 241), mat, strip_uv(len(zz), 241), inward
    )
    last = cyl_grid([zz[-1]], [radii[-1]], 240, False)[0]
    side = 2 * radii[-1]
    yy, xx = np.indices((640, 640))
    xy = np.stack([xx / 639 - 0.5, 0.5 - yy / 639], -1) * side
    tex = wall_tex(xyz(xy, -zz[-1]))
    mat = mesh.material("Deep cavity appearance", tex, 0.99, 0, 0.3)
    mesh.add(
        "Alexander optical termination",
        last,
        face_up(last, rings_faces([last[:, :2]])),
        last[:, :2] / side + 0.5,
        mat,
    )
    # Annular casting, inset bearing shoulder and sleeve use adjacent source colors.
    dd = np.array([0, 0.004, 0.007, seat * 0.70, seat * 0.73, seat])
    rr = np.array([RADIUS, RADIUS - 0.011, bore + 0.021, bore + 0.017, bore + 0.004, bore])
    for i in range(len(dd) - 1):
        d = np.linspace(dd[i], dd[i + 1], 16)
        r = np.linspace(rr[i], rr[i + 1], 16)
        p = cyl_grid(d, r, 960)
        tex = photos[0](cams[0].project(p))
        # On the near face the outer rim occludes the camera's ray. Mirror rust from
        # the opposite azimuth where the inner edge is directly visible.
        front = unit(cams[0].eye[:2])
        dot = p[..., :2] @ front
        reverse = p.copy()
        reverse[..., :2] -= 2 * dot[..., None] * front
        if i > 1:
            tex = np.where((dot > 0)[..., None], photos[0](cams[0].project(reverse)), tex)
        mat = mesh.material("Iron rim course " + str(i), tex, 0.78, 0.28, 0.5)
        v = cyl_grid(dd[i : i + 2], rr[i : i + 2], 241)
        lattice(
            mesh,
            "Alexander casting " + str(i),
            v,
            mat,
            strip_uv(2, 241),
            lambda c: inward(c) + [0, 0, 0.2],
        )
    return dict(
        profile_depth_m=depths.tolist(),
        profile_radius_m=radii.tolist(),
        fitting_voxels=len(vox),
        visual_termination_depth_m=depths[-1],
    )


def lid(mesh, cams, reg, photos):
    cam = cams[0]
    photo = photos[0]
    n = reg["lid_normal"]
    center = reg["lid_center"]
    r = reg["lid_radius"]
    u = unit(np.cross([0, 1, 0], n))
    v = np.cross(n, u)

    def world(xy):
        return center + xy[..., :1] * u + xy[..., 1:] * v

    y, x = np.indices((896, 896))
    q = np.stack([x / 895 - 0.5, 0.5 - y / 895], -1) * (2 * r)
    tex = photo(cam.project(world(q)))
    mat = mesh.material("Photographic SEWER casting", tex, 0.76, 0.38, 1.7)
    # Real lettering and lozenge wear are carried by the rectified photographic finish.
    rr = np.linspace(0, r, 85)[:, None, None]
    theta = np.linspace(0, 2 * np.pi, 257)
    disk = rr * np.stack([np.cos(theta), np.sin(theta)], -1)
    verts = world(disk)
    lattice(mesh, "Alexander SEWER lid", verts, mat, disk / (2 * r) + 0.5, lambda c: n)
    ring = verts[-1]
    lip = ring - 0.015 * n
    # The street view resolves the lower silhouette under the top plate. A narrow
    # exposed rim overhangs a darker, recessed support that descends to the road.
    inset = center + (ring - center) * 0.86
    side = np.stack([ring, lip, inset - 0.018 * n, xyz(inset[:, :2], 0.002)])
    a = np.linspace(0, 2 * np.pi, 769)
    edge = world(r * np.stack((np.cos(a), np.sin(a)), -1))
    tt = np.linspace(0, 1, 80)
    sample_z = edge[None] - tt[:, None, None] * 0.016 * n
    street = photos[1](cams[1].project(sample_z))
    toward = unit((cams[1].eye - center)[:2])
    visible = (edge[:, :2] - center[:2]) @ toward > 0
    rust = photos[0](cam.project(world((r - 0.006) * np.stack([np.cos(a), np.sin(a)], -1))[None]))
    band = np.where(visible[None, :, None], street, rust * 0.67)
    tex = np.vstack([band, np.broadcast_to(band[-1:] * 0.34, (76, len(edge), 3))])
    material = mesh.material("Recessed rusty lid edge", tex, 0.87, 0.24, 0.65)
    uv = np.stack(
        np.broadcast_arrays(np.linspace(0, 1, len(theta)), np.array([1, 0.49, 0.10, 0])[:, None]),
        -1,
    )
    away = lambda c: xyz(c[:, :2] - center[:2], -0.02)
    lattice(mesh, "Alexander SEWER lid edge", side[:2], material, uv[:2], away)
    lattice(mesh, "Alexander hidden lid support", side[1:], material, uv[1:], away)
    under = side[-1, :-1]
    mesh.add(
        "Alexander hidden lid underside",
        under,
        face_up(under, rings_faces([under[:, :2]]))[:, ::-1],
        under[:, :2] - center[:2] + 0.5,
        material,
    )
    return ring


def road(mesh, cams, reg, photos, span):
    xy = grid(span, 0.002)
    v = xyz(xy, 0)
    regions = reg["traces"][0]
    mask = np.ones(tuple(SIZE[::-1]), "u1")
    for k in ["lid", "rim"]:
        cv2.fillPoly(mask, [np.int32(regions[k])], 0)
    cv2.fillPoly(mask, [np.int32(preview([[534, 38], [612, 25], [612, 225], [539, 137]]))], 0)
    gap = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    q = cams[0].project(v).astype("f4")
    x, y = q.transpose(2, 0, 1)
    available = cv2.remap(gap, x, y, cv2.INTER_LINEAR)
    w = np.clip(np.minimum.reduce([x, y, SIZE[0] - x, SIZE[1] - y, available - 9]) / 18, 0, 1)
    tile = 1 - abs((xy / [0.27, 0.23] % 2) - 1)
    base = photos[0](preview(tile * [98, 82] + [75, 34]))
    tex = base * (1 - w[..., None]) + photos[0](q) * w[..., None]
    mat = mesh.material("Alexander Street asphalt", tex, 0.95, 0, 0.9)
    outer = rect(span)
    inner = xyz(circle() * RADIUS, 0)
    v = np.vstack([outer, inner])
    mesh.add(
        "Road around the aperture",
        v,
        face_up(v, rings_faces([outer[:, :2], inner[:, :2]])),
        v[:, :2] / span + 0.5,
        mat,
        road=True,
    )


def sky_map(surface, span):
    orig = xyz(grid(span, 0.012), 0.001)
    total = np.zeros(orig.shape[:2], "f4")
    for i in range(32):
        z = (i + 0.5) / 32
        a = i * np.pi * (3 - np.sqrt(5))
        d = [np.cos(a) * np.sqrt(1 - z * z), np.sin(a) * np.sqrt(1 - z * z), z]
        rays = np.concatenate([orig, np.broadcast_to(d, orig.shape)], -1)
        total += (
            z
            / 16
            * surface.scene.test_occlusions(
                o3d.core.Tensor(rays.astype("f4")), tfar=2.0, nthreads=2
            ).numpy()
        )
    return total


def export_subset(mesh, path, parts, origin):
    subset = Mesh()
    keys = {p[4] for p in parts}
    subset.materials = {k: v for k, v in mesh.materials.items() if k in keys}
    subset.parts = [(name, v - origin, f, uv, mat, road) for name, v, f, uv, mat, road in parts]
    path.write_bytes(subset.glb_bytes())
    flatten_scene(path)


def build(out, stereo):
    cams, reg = fit()
    photo = photographs()
    mesh = Mesh()
    if not stereo.is_file():
        solve(stereo)
    with np.load(stereo) as data:
        shell = interior(mesh, cams, reg, photo, data)
    lid(mesh, cams, reg, photo)
    rung = make_step(mesh, cams, photo, grid_faces, orient)
    post = delineator(mesh, cams[1], photo[1], xyz, orient, grid_faces, face_up, rings_faces)
    parts = np.vstack(
        [
            p[1]
            for p in mesh.parts
            if not p[0].startswith(("Alexander masonry", "Alexander optical"))
        ]
    )
    span = np.ceil((np.max(abs(parts[:, :2]), axis=0) + 0.13) * 100) * 0.02
    # Include the surveyed replay's full post shadow on the receiving road. The
    # inspection apron also covers the plan footprint of the flaring shell.
    span = np.maximum(span, [3.48, 2.32])
    road(mesh, cams, reg, photo, span)
    folder = out / "series/assets/missing/M585"
    cache = out / "series/mesh_cache"
    meta = dict(
        id="M585",
        family="alexander_opening",
        category="missing",
        state="displaced_sewer_lid",
        label="亚历山大街移开的污水井盖、井内踏步与警示柱",
        seed=29585,
        source=[S["detail_url"], S["street_url"]],
        source_author=S["reference_author"],
        source_license=S["image_license"],
        surface_dimensions_m=span.tolist(),
        assumed_casting_outer_diameter_m=2 * RADIUS,
        camera_centres_m=[c.eye.tolist() for c in cams],
        lid_center_m=reg["lid_center"].tolist(),
        lid_radius_m=reg["lid_radius"],
        lid_normal=reg["lid_normal"].tolist(),
        bore_radius_m=reg["bore_radius"],
        bearing_z_m=reg["seat_z"],
        step=rung,
        marker=post,
        masonry=shell,
    )
    mesh.save(folder, cache, meta, dict(width=span[0], length=span[1]))
    flatten_scene(folder / "asset.glb")
    cover_parts = [
        p for p in mesh.parts if p[0].startswith(("Alexander SEWER", "Alexander hidden lid"))
    ]
    marker_parts = [p for p in mesh.parts if p[0].startswith("Alexander marker")]
    export_subset(mesh, folder / "cover.glb", cover_parts, np.r_[reg["lid_center"][:2], 0])
    export_subset(mesh, folder / "marker.glb", marker_parts, np.asarray(post["base_origin_m"]))
    names = {p[0] for p in cover_parts}
    export_subset(
        mesh, folder / "opening.glb", [p for p in mesh.parts if p[0] not in names], np.zeros(3)
    )
    bounded = [
        p
        for p in mesh.parts
        if not p[0].startswith(("Alexander masonry", "Alexander optical", "Alexander hidden lid"))
    ]
    export_subset(mesh, folder / "bounded.glb", bounded, np.zeros(3))
    field = Surface(folder / "asset.glb")
    xy = grid(span, 0.005)
    orig = xyz(xy, 2.0)
    rays = np.broadcast_to([0.0, 0.0, -1.0], orig.shape)
    d, part, _ = field.cast(orig, rays)
    hit = np.isfinite(d)
    lower = Surface(folder / "bounded.glb", exclude={1, 5, 187})
    with np.load(stereo) as data:
        np.savez_compressed(folder / "stereo.npz", **dict(data))
    np.savez_compressed(
        folder / "surface.npz",
        width=span[0],
        length=span[1],
        height=np.where(hit, 2 - d, 0).astype("f4"),
        alpha=hit,
        first_part=part,
        opening_mask=np.linalg.norm(xy, axis=2) < reg["bore_radius"],
        installation_mask=part > 1,
        defect_mask=part > 1,
        water_mask=np.zeros_like(hit),
        unconstrained_depth=np.isin(part, [186, 189, 190]),
        photographic_rgb=True,
        receiver_sky_blocked=sky_map(lower, span),
        cast_receiver_sunlight=True,
        registered_relief=True,
    )
    row = make_one(
        folder,
        cache,
        point_limit=520000,
        kernel_factors=(0.79, 0.15),
        part_importance={
            "Alexander step": 6.0,
            "Alexander SEWER lid": 20.0,
            "Alexander optical termination": 0.22,
            "Alexander masonry envelope": 0.84,
            "Alexander marker striped sleeve": 4.0,
            "Road around the aperture": 1.2,
        },
    )
    for k, f in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("metadata", "metadata.json"),
        ("surface", "surface.npz"),
    ]:
        row[k] = "missing/M585/" + f
    write_json(out / "series/assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("stereo", type=Path)
    args = p.parse_args()
    args.stereo.parent.mkdir(parents=True, exist_ok=True)
    build(args.out, args.stereo)
