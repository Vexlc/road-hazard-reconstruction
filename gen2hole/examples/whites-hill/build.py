"""Reconstruct M576 from the February 2018 Whites Hill photographs."""

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from cameras import Cameras, S, photograph
from geometry import Parts, rect, rounded, sample, strip, unit, xyz
from photo_mesh import Surface, flatten_scene, grid, make_one, write_json

UP = np.array([0.0, 0.0, 1.0])


def xy(z):
    return lambda q: xyz(q, z)


def make(out):
    cam = Cameras(S["working_focal_px"])
    near = photograph("5710912")
    wide = photograph("5710915")
    parts = Parts()
    mesh = parts.mesh

    def material(name, img, r=0.9, m=0.0, b=0.5):
        return mesh.material(name, np.ascontiguousarray(img), r, m, b, lossless_color=True)

    def chip(name, box, **kw):
        x0, y0, x1, y1 = box
        return material(name, near[y0:y1, x0:x1], **kw)

    road = chip("Damp road aggregate", (7, 269, 43, 305))
    iron = chip("Weathered iron frame", (149, 149, 168, 161), r=0.78, m=0.25)
    cement = chip("Chamber wall concrete", (391, 202, 452, 277))
    lid = chip("Mottled slab sample", (342, 567, 426, 631))
    edge = chip("Cover edge sample", (150, 148, 165, 158), r=0.86)
    beam_edge = chip("Dark rail side", (291, 239, 296, 251), r=0.77, m=0.28)

    # Assign angle-dependent source reflections only in a separate capture-context mesh.
    def plane_texture(name, place, bounds, allowed, base, detail=500, wide_view=False, skip=()):
        x0, y0, x1, y1 = bounds
        nx, ny = np.maximum(8, np.ceil(np.array([x1 - x0, y1 - y0]) * detail).astype(int))
        px, py = np.meshgrid(
            x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx, y1 - (np.arange(ny) + 0.5) * (y1 - y0) / ny
        )
        q = np.stack((px, py), axis=-1)
        proj = cam.project(place(q.reshape(-1, 2)), wide=wide_view).reshape(ny, nx, 2)
        source = wide if wide_view else near
        mask = np.zeros(source.shape[:2], "u1")
        for poly in allowed:
            cv2.fillPoly(mask, [np.int32(poly)], 255)
        for poly in skip:
            cv2.fillPoly(mask, [np.int32(poly)], 0)
        mask = cv2.GaussianBlur(mask, (3, 3), 0.65)
        weight = (
            cv2.remap(
                mask, proj.astype("f4"), None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            ).astype(float)[:, :, None]
            / 255
        )
        iy, ix = np.indices((ny, nx))
        fallback = sample(mesh.materials[base][0], np.stack([ix, iy], -1), cv2.BORDER_REFLECT_101)
        img = (sample(source, proj) * weight + fallback * (1 - weight)).astype("u1")
        return material(name, img), lambda p, v: (p - [x0, y0]) / [x1 - x0, y1 - y0]

    sx, sy = cam.size / 2
    mouth = rect(-sx, -sy, sx, sy)
    measured = cam.plane([[58, 150], [451, 57], [203, 516]], 0)
    outer = rect(
        min(-sx - 0.025, measured[0, 0]),
        min(-sy - 0.025, measured[2, 1]),
        max(sx + 0.025, measured[1, 0]),
        max(sy + 0.025, measured[0, 1]),
    )
    # Slab thickness is set by the pixel offset between its top and lower edge.
    top_px, base_px = S["wide_slab_side_px"]
    foot = cam.plane([base_px], 0, wide=True)[0]

    def slab_loss(z):
        return np.linalg.norm(cam.project((foot + [0, 0, z])[None], True)[0] - top_px)

    slab_z = minimize_scalar(slab_loss, bounds=(0.015, 0.14), method="bounded").x
    corners = np.asarray(S["wide_slab_top"], float)
    estimate = cam.plane(corners, slab_z, wide=True)[:, :2].mean(0)
    unit_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])

    def footprint(q, points):
        x, y, w, l, angle = q
        cs, sn = np.cos(angle), np.sin(angle)
        return points * np.array([w, l]) @ np.array([[cs, sn], [-sn, cs]]) + [x, y]

    def residual(q):
        return (cam.project(xyz(footprint(q, unit_corners), slab_z), True) - corners).ravel()

    # A single displaced cover is modelled large enough to rest on the opening.
    shape = least_squares(
        residual,
        [*estimate, 0.91, 0.69, 0.0],
        bounds=([-2, -3, cam.size[0] + 0.014, cam.size[1] + 0.014, -0.8], [2, 0, 1.8, 1.8, 0.8]),
    ).x
    slab = footprint(shape, rounded(-0.5, -0.5, 0.5, 0.5, 0.022))
    apron_span = np.ceil(np.maximum(abs(slab).max(0), abs(cam.guard).max(0)) * 2 / 0.1) * 0.1 + 0.24
    apron = rect(*(-apron_span / 2), *(apron_span / 2))
    road_mask = [
        [
            [0, 0],
            [479, 0],
            [479, 82],
            [454, 54],
            [59, 148],
            [216, 513],
            [478, 396],
            [479, 479],
            [265, 561],
            [283, 639],
            [0, 639],
        ]
    ]
    road_mat, road_uv = plane_texture(
        "Photographed road and verge",
        xy(-0.001),
        [*apron.min(0), *apron.max(0)],
        road_mask,
        road,
        detail=480,
        skip=[
            [[36, 0], [155, 0], [119, 89], [56, 96]],
            [[369, 0], [479, 0], [479, 65], [379, 34]],
            [[253, 522], [479, 394], [479, 475], [264, 549]],
        ],
    )
    parts.face("Road around the aperture", [apron, outer], xy(-0.001), road_mat, road_uv, UP)
    rim_box = [*outer.min(0), *outer.max(0)]
    ring_mat, ruv = plane_texture(
        "Photographed rim",
        xy(0),
        rim_box,
        [
            [
                [59, 148],
                [451, 56],
                [479, 79],
                [479, 181],
                [404, 99],
                [103, 179],
                [238, 477],
                [479, 369],
                [479, 397],
                [224, 515],
            ],
            [[60, 148], [104, 176], [238, 478], [218, 508]],
        ],
        iron,
        skip=[[[259, 518], [478, 397], [478, 456], [266, 543]]],
    )
    parts.face("Cast ring and cover seat", [outer, mouth], xy(0), ring_mat, ruv, UP, 0.035)
    strip(parts, "Cast iron throat skirt", mouth[::-1], -0.035, 0, iron)
    z = float(np.median(cam.report()["waterline_on_right_wall_m"]))
    far_wall = cam.plane([[283, 180]], sx, axis=0)[0, 1]
    cavity = rect(-sx, -sy, sx, far_wall)
    for i, (a, b) in enumerate(zip(cavity, np.roll(cavity, -1, axis=0))):
        side = unit(b - a)
        length = np.linalg.norm(b - a)

        def place(q):
            return np.column_stack([a + q[:, 0, None] * side, q[:, 1]])

        domain = rect(0, z, length, -0.035)
        if i == 1:
            mat, uv = plane_texture(
                "Observed mottled right wall",
                place,
                [0, z, length, -0.035],
                [
                    [
                        [288, 133],
                        [401, 101],
                        [478, 190],
                        [478, 362],
                        [447, 380],
                        [385, 348],
                        [319, 231],
                        [284, 177],
                    ]
                ],
                cement,
                skip=[
                    [[341, 109], [345, 108], [362, 314], [355, 311]],
                    [[376, 341], [451, 287], [472, 296], [411, 360], [392, 364]],
                ],
            )
        else:
            dark = material("Sheltered wall", np.uint8(mesh.materials[cement][0] * 0.31))
            back_roi = [[101, 179], [280, 137], [280, 177], [250, 197], [208, 211], [142, 231]]
            mat, uv = plane_texture(
                "Sheltered masonry " + str(i),
                place,
                [0, z, length, -0.035],
                [back_roi] if i == 2 else [],
                dark,
                skip=[
                    [
                        [165, 213],
                        [196, 192],
                        [207, 164],
                        [217, 158],
                        [212, 196],
                        [182, 227],
                        [162, 225],
                    ],
                    [[230, 152], [251, 147], [278, 183], [256, 191]],
                ],
            )
        parts.face(
            [
                "Chamber joint recess",
                "Inspection shaft",
                "Coursed concrete chamber back",
                "Curb return left",
            ][i],
            [domain],
            place,
            mat,
            uv,
            np.r_[-side[1], side[0], 0],
        )
    # Roof over the recessed far wall; there is no floor below the visible water.
    roof = rect(-sx, sy, sx, far_wall)
    parts.face(
        "Photographed soffit", [roof], xy(-0.035), iron, lambda q, v: q / cam.size + 0.5, -UP
    )
    water_base = chip("Still brown water", (193, 263, 239, 305), r=0.12, b=0)
    water_bounds = [*cavity.min(0), *cavity.max(0)]
    capture_water, wuv = plane_texture(
        "Water in the close exposure",
        xy(z),
        water_bounds,
        [
            [
                [133, 230],
                [157, 217],
                [168, 224],
                [185, 222],
                [200, 207],
                [216, 197],
                [240, 201],
                [266, 221],
                [312, 285],
                [351, 347],
                [374, 371],
                [310, 397],
                [329, 408],
                [391, 378],
                [431, 387],
                [374, 421],
                [303, 454],
                [239, 475],
                [205, 383],
                [172, 312],
            ]
        ],
        water_base,
        detail=650,
    )
    parts.face("Water surface", [cavity], xy(z), water_base, wuv, UP)
    # The squared rail is intercepted by a cross bracket on the right wall.
    rail_z = z + 0.16
    rail_px = [
        [232, 151],
        [251, 146],
        [274, 179],
        [307, 226],
        [337, 274],
        [361, 309],
        [384, 344],
        [392, 359],
        [378, 370],
        [365, 349],
        [348, 321],
        [325, 284],
        [302, 249],
        [277, 210],
        [253, 175],
    ]
    rail = cam.plane(rail_px, rail_z)[:, :2]
    if cv2.contourArea(rail.astype("f4"), oriented=True) < 0:
        rail = rail[::-1]
    rail_bounds = [*rail.min(0), *rail.max(0)]
    rmat, uv = plane_texture(
        "Yellow and rust on the cross member",
        xy(rail_z),
        rail_bounds,
        [rail_px],
        beam_edge,
        detail=950,
    )
    parts.face("Central bearing beam", [rail], xy(rail_z), rmat, uv, UP, 0.045)
    strip(parts, "Recessed grate crossbars", rail, rail_z - 0.025, rail_z, beam_edge)
    # A short cast crosspiece inclines from the wall bolt down to the beam.
    bracket_pts = [[307, 387], [455, 292], [466, 302], [316, 405], [307, 400]]
    attachment = cam.plane([[455, 295]], sx, axis=0)[0]
    q = cam.rays(bracket_pts)
    rail_tip = cam.plane([[312, 396]], rail_z - 0.03)[0]
    horizontal = unit((rail_tip - attachment)[:2])
    normal = unit(
        np.r_[
            -horizontal
            * (rail_tip[2] - attachment[2])
            / np.linalg.norm((rail_tip - attachment)[:2]),
            1.0,
        ]
    )
    v = cam.eye + q * ((attachment - cam.eye) @ normal / (q @ normal))[:, None]
    ptex = material("Observed bracket patina", near)

    def photo_uv(q, v):
        return (cam.project(v) + 0.5) / [480, 640] * [1, -1] + [0, 1]

    from geometry import rings_faces

    tri = rings_faces([v[:, :2]])
    parts.add("Wall cable fitting", v, tri, photo_uv(None, v), ptex)
    # Visible end of the duct and its stepped open coupling. The cable disappears under the far frame.
    from inspection import coupling, observed_context

    end, axial, r = coupling(cam)
    jacket = chip("Black duct jacket", (191, 195, 195, 202), r=0.63)
    stripe = chip("Yellow band on duct", (202, 184, 206, 192), r=0.68)
    grey = chip("Grey coupling sleeve", (173, 205, 180, 210), r=0.51)
    path = np.stack(
        [
            cam.plane([p], zz)[0]
            for p, zz in zip(
                [[210, 158], [210, 176], [206, 188], [200, 196]], [-0.36, -0.38, -0.39, -0.39]
            )
        ]
        + [end - axial * 4.3 * r]
    )
    parts.tube("Black insulated cables", path, 0.64 * r, jacket)
    parts.tube("Tracer wire", path[1:3], 0.68 * r, stripe)
    parts.tube(
        "Pipe coupling",
        end
        + np.array([-4.3, -3.6, -3.1, -2.4, -2.0, -1.5, -1.1, -0.8, -0.32, 0])[:, None] * r * axial,
        r * np.array([0.89, 1.0, 1.16, 1.16, 0.84, 0.85, 1.17, 1.18, 1.06, 1.0]),
        grey,
    )
    parts.tube(
        "Electrical conduit 1 bore",
        end + np.array([0, -1.8])[:, None] * r * axial,
        0.58 * r,
        jacket,
        inward=True,
    )
    a = unit(np.cross(axial, [0, 0, 1]))
    b = np.cross(axial, a)
    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    circ = np.c_[np.cos(theta), np.sin(theta)]
    parts.face(
        "Electrical conduit 1 end",
        [circ * r, circ * 0.58 * r],
        lambda p: end + p[:, 0, None] * a + p[:, 1, None] * b,
        grey,
        lambda q, v: q / (2 * r) + 0.5,
        axial,
    )
    # Narrow grey line from the verge to the fittings; intermediate sag is inferred.
    tether_px = [
        [304, 0],
        [323, 22],
        [333, 43],
        [330, 54],
        [337, 94],
        [344, 128],
        [349, 157],
        [345, 193],
        [350, 233],
        [355, 268],
        [359, 312],
    ]
    levels = np.interp(np.arange(len(tether_px)), [0, 4, 10], [0.002, -0.002, rail_z])
    line = np.stack([cam.plane([p], zz)[0] for p, zz in zip(tether_px, levels)])
    parts.tube(
        "Grey insulated cables", line, 0.0027, chip("Grey line fibres", (338, 98, 341, 107)), 14
    )
    # The separate slab is positioned from its own outline in the lane approach.
    bounds = np.ptp(slab, axis=0)
    # Wide-view low frequencies preserve the recorded shadow; the close photo supplies grain.
    ny, nx = np.maximum(8, (bounds[::-1] * 480).astype(int))
    xx, yy = np.meshgrid((np.arange(nx) + 0.5) / nx, 1 - (np.arange(ny) + 0.5) / ny)
    surface_xy = slab.min(0) + np.stack([xx, yy], -1) * bounds
    wx = cam.project(xyz(surface_xy, slab_z), True)
    low = sample(wide, wx).astype(float)
    chip_img = mesh.materials[lid][0].astype("f4")
    granular = chip_img - cv2.GaussianBlur(chip_img, (0, 0), 2.6)
    hy, hx = np.indices((ny, nx))
    slab_img = np.clip(
        low + 0.47 * sample(granular, np.stack([hx, hy], -1), cv2.BORDER_REFLECT_101), 0, 255
    )
    slab_mat = material("Displaced slab in roadside shade", slab_img.astype("u1"))
    edge = material("Slab shadow edge", np.uint8(mesh.materials[edge][0] * 0.40))
    parts.face(
        "Lid face", [slab], xy(slab_z), slab_mat, lambda q, v: (q - slab.min(0)) / bounds, UP
    )
    strip(parts, "Lid edge", slab, 0.0, slab_z, edge)
    add_guard(parts, cam, wide)

    mesh = parts.finish()
    road_source = mesh.materials[road_mat]
    # Match the road in the shaded wide exposure. Keep the close exposure for inspection.
    near_patch = near[305:350, 10:56].reshape(-1, 3)
    wide_patch = wide[261:273, 189:223].reshape(-1, 3)
    gain = np.median(wide_patch, 0) / np.maximum(np.median(near_patch, 0), 1)
    mesh.materials[road_mat] = (np.uint8(np.clip(road_source[0] * gain, 0, 255)), *road_source[1:])
    series = out / "series"
    folder = series / "assets/missing/M576"
    cache = series / "mesh_cache"
    xygrid = grid(apron_span, 0.006)
    ext = np.ptp(xygrid, axis=(0, 1))
    meta = dict(
        id="M576",
        label="Whites Hill 移开盖板与积水通信井",
        seed=29576,
        category="missing",
        state="field_open_guarded",
        family="whites_hill",
        source=[r["page"] for r in S["references"].values()],
        source_author=S["photographer"],
        source_license="CC BY-SA 2.0",
        surface_dimensions_m=ext.tolist(),
        camera_fit=cam.report(),
        water_plane_m=z,
        slab_thickness_m=slab_z,
        slab_length_width_m=shape[2:4].tolist(),
        slab_corner_rms_px=float(
            np.sqrt(np.mean(np.sum(residual(shape).reshape(-1, 2) ** 2, axis=1)))
        ),
        slab_xy_m=slab.tolist(),
        guard_foot_centres_m=cam.guard.tolist(),
        beam_top_m=rail_z,
        far_wall_y_m=far_wall,
        close_optics="inspection.glb uses only the photographed panel fragments and slab corner, with source-view water reflections.",
        general_optics="asset.glb uses the enclosure and moved cover from the wide photograph and a neutral water material.",
        unknown="No surveyed dimensions, bottom range or underwater cable positions. Hardware heights and unseen wall continuation are modelled.",
        part_codes={"guard": 174, "water": 4, "frame": 5, "wall": 6, "slab_top": 9, "beam": 20},
    )
    field = dict(
        width=ext[0],
        length=ext[1],
        height=np.zeros(xygrid.shape[:2]),
        alpha=np.ones(xygrid.shape[:2]),
    )
    mesh.save(folder, cache, meta, field)
    flatten_scene(folder / "asset.glb")
    capture = copy.copy(mesh)
    capture.parts = [
        (n, v, f, uv, capture_water if n == "Water surface" else mat, road)
        for n, v, f, uv, mat, road in mesh.parts
        if not n.startswith("Folding manhole guard") and n not in ("Lid face", "Lid edge")
    ]
    capture.materials = dict(mesh.materials)
    capture.materials["Near exposure"] = mesh.materials[ptex]
    capture.materials[road_mat] = road_source
    capture.parts += observed_context(cam, near, capture.materials["Near exposure"], slab_z)
    (folder / "inspection.glb").write_bytes(capture.glb_bytes())
    flatten_scene(folder / "inspection.glb")
    ray = np.broadcast_to([0.0, 0.0, -1.0], (*xygrid.shape[:2], 3))
    dist, part, _ = Surface(folder / "asset.glb").cast(xyz(xygrid, 2.0), ray)
    below, _, _ = Surface(folder / "asset.glb", exclude={4}).cast(xyz(xygrid, 2.0), ray)
    aperture = (abs(xygrid[:, :, 0]) < sx) & (abs(xygrid[:, :, 1]) < sy)
    guard_zone = (part == 174) | (part == 9) | (part == 11)
    installation = (
        aperture
        | guard_zone
        | (
            (abs(xygrid[:, :, 0]) <= abs(outer[:, 0]).max())
            & (abs(xygrid[:, :, 1]) <= abs(outer[:, 1]).max())
        )
    )
    from scipy.ndimage import distance_transform_edt

    alpha = np.clip(1 - distance_transform_edt(~installation) * 0.006 / 0.045, 0, 1)
    np.savez_compressed(
        folder / "surface.npz",
        width=ext[0],
        length=ext[1],
        height=np.where(np.isfinite(dist), 2.0 - dist, 0.0),
        alpha=alpha,
        photographic_rgb=True,
        defect_mask=installation,
        installation_mask=installation,
        opening_mask=aperture,
        water_mask=part == 4,
        first_part=part,
        solid_hit=np.isfinite(below),
        solid_range_m=np.where(np.isfinite(below), below, np.nan),
    )
    row = make_one(
        folder,
        cache,
        point_limit=650000,
        kernel_factors=(0.79, 0.16),
        part_importance={
            "Central bearing beam": 7.0,
            "Pipe coupling": 12.0,
            "Grey insulated cables": 9.0,
        },
    )
    for role, name in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("surface", "surface.npz"),
        ("metadata", "metadata.json"),
    ]:
        row[role] = "missing/M576/" + name
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


def add_guard(parts, cam, photo):
    mesh = parts.mesh
    # In the best-resolved wide panel, the openings reveal rails of the panels behind.
    # Model those openings through the plastic, including their inner edge surfaces.
    a, b = cam.guard[:2]
    H = cam.guard_height
    xx, zz = np.meshgrid((np.arange(200) + 0.5) / 200, 1 - (np.arange(240) + 0.5) / 240)
    ref = xyz(a + (b - a) * xx[:, :, None], zz * H)
    atlas = sample(photo, cam.project(ref, wide=True))
    # Low-resolution wide pixels mix the front panel with the rails behind.
    # Remove background colour from the plastic and place reflectors only on the outside.
    clean_red = (atlas[:, :, 0].astype(float) > 1.7 * atlas[:, :, 1]) & (
        atlas[:, :, 0].astype(float) > 1.6 * atlas[:, :, 2]
    )
    red = np.median(atlas[clean_red], axis=0)
    observed = photograph("5710912")[36:67, 64:99].astype("f4")
    variation = observed - cv2.GaussianBlur(observed, (0, 0), 2.3)
    yy, xx = np.indices(atlas.shape[:2])
    grain = sample(variation, np.stack([xx, yy], -1), cv2.BORDER_REFLECT_101) * 0.22
    inner = np.clip(red + grain, 0, 255).astype("u1")
    yfrac = (yy + 0.5) / atlas.shape[0]
    xfrac = (xx + 0.5) / atlas.shape[1]
    red_weight = clean_red.astype("f4")
    # Smooth only pixels belonging to red plastic, so the rear panels and vegetation
    # do not leave silhouettes on the front rails. The neutral term fills deep slots.
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), 5.0)
    low = (blur(atlas.astype("f4") * red_weight[:, :, None]) + 0.12 * red) / (
        blur(red_weight)[:, :, None] + 0.12
    )
    outer = np.uint8(np.clip(0.78 * low + 0.22 * red + grain, 0, 255))
    for lo, hi in [(0.025, 0.205), (0.779, 0.963)]:
        reflect = (xfrac > 0.313) & (xfrac < 0.644) & (yfrac > lo) & (yfrac < hi)
        outer[reflect] = atlas[reflect]
    face = mesh.material("Red plastic and two reflective strips", outer, 0.61, 0.03, 0.14)
    back = mesh.material("Unstriped inside of guard", inner, 0.68, 0.02, 0.14)
    side = mesh.material("Red moulded edges", np.uint8(inner * 0.74), 0.65, 0.02, 0.2)
    # Coordinates are fractions of the panel width and occupied height.
    rings = [rounded(0, 0.015, 1, 0.995, 0.026)]
    for bottom, top in [(0.222, 0.458), (0.536, 0.769)]:
        for x0, x1, r in [(0.012, 0.059, 0.020), (0.106, 0.858, 0.038), (0.906, 0.972, 0.023)]:
            rings.append(rounded(x0, bottom, x1, top, r))
    rings.append(rounded(0.107, 0.959, 0.291, 0.989, 0.010))
    name = "Folding manhole guard panel "
    for i, (a, b) in enumerate(zip(cam.guard, np.roll(cam.guard, -1, axis=0))):
        edge = b - a
        out = unit(np.array([edge[1], -edge[0]]))
        normal = np.r_[out, 0.0]

        def place(p, off=0):
            return xyz(a + p[:, 0, None] * edge + off * out, p[:, 1] * H)

        for off, n in [(0.011, normal), (-0.011, -normal)]:
            parts.face(
                name + str(i + 1) + (" outside" if off > 0 else " inside"),
                rings,
                lambda p: place(p, off),
                face if off > 0 else back,
                lambda p, v: p,
                n,
                pitch=0.15,
            )
        # Give the folding corners cylindrical knuckles. The far exposure resolves
        # their location but not individual pin lengths.
        parts.tube(
            name + str(i + 1) + " hinge pin",
            xyz(np.vstack([a, a]), [0.028, 0.961]),
            0.010,
            side,
            16,
        )
        for lo, hi in [(0.068, 0.198), (0.452, 0.538), (0.784, 0.926)]:
            zs = np.array([lo, lo + 0.007, hi - 0.007, hi])
            parts.tube(
                name + str(i + 1) + " hinge",
                xyz(np.tile(a, (4, 1)), zs),
                [0.014, 0.019, 0.019, 0.014],
                side,
                20,
            )
        # Moulded feet at the paired end stiles contact the road.
        for fraction in (0.035, 0.965):
            mid = a + fraction * edge
            e = unit(edge)
            foot = mid + np.array(
                [
                    -e * 0.028 - out * 0.027,
                    e * 0.028 - out * 0.027,
                    e * 0.028 + out * 0.027,
                    -e * 0.028 + out * 0.027,
                ]
            )
            if cv2.contourArea(foot.astype("f4"), oriented=True) < 0:
                foot = foot[::-1]
            strip(parts, name + str(i + 1) + " feet", foot, 0.0, 0.027, side)
            parts.face(
                name + str(i + 1) + " feet", [foot], xy(0.027), side, lambda q, v: q * 0 + 0.5, UP
            )
        for j, ring in enumerate(rings):
            for v, w in zip(ring, np.roll(ring, -1, axis=0)):
                q = np.array([v, w, w, v])
                vertices = place(q)
                vertices[:2, :2] += out * 0.011
                vertices[2:, :2] -= out * 0.011
                faces = np.array([[0, 1, 2], [0, 2, 3]])
                if j == 0:
                    faces = faces[:, ::-1]
                parts.add(name + str(i + 1) + " edge", vertices, faces, rect(0, 0, 1, 1), side)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    make(p.parse_args().out)
