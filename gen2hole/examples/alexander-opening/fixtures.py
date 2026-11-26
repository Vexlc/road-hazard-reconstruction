"""Place the interior step from both photographs and the slotted delineator from the street view."""

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.ndimage import distance_transform_edt
from geometry import SIZE, unit


def preview(p):
    return (np.asarray(p) + 0.5) * SIZE / [613, 460] - 0.5


def wall_pixel(p, i):
    # Coordinate windows used to trace the inside of the opening in the two sources.
    return preview(
        np.asarray(p) / 520 * ([222, 223] if i == 0 else [145, 131])
        + ([338, 97] if i == 0 else [235, 244])
    )


def marker_pixel(p):
    return preview(np.asarray(p) / [560, 565] * [259, 263] + [334, 69])


def circle(n=192):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def step_path(cams):
    traces = [
        [[143, 78], [155, 80], [168, 81], [181, 81], [190, 79], [196, 75], [200, 69], [201, 61]],
        [
            [56, 178],
            [66, 179],
            [74, 177],
            [81, 174],
            [85, 169],
            [87, 162],
            [86, 152],
            [86, 140],
            [86, 131],
        ],
    ]
    curves = [wall_pixel(p, i) for i, p in enumerate(traces)]

    def error(z):
        # Part of each endpoint is hidden by the lip. Compare only the common bend.
        world = [c.plane(p, z) for c, p in zip(cams, curves)]
        distances = []
        for i, (lo, hi) in enumerate([(3, 8), (2, 8)]):
            samples = cams[1 - i].project(world[i][lo:hi])
            target = curves[1 - i]
            a, b = target[:-1], target[1:]
            ab = b - a
            t = np.clip(np.sum((samples[:, None] - a) * ab, axis=2) / np.sum(ab**2, axis=1), 0, 1)
            ds = np.linalg.norm(samples[:, None] - a - t[:, :, None] * ab, axis=2)
            distances.extend(ds.min(axis=1))
        return np.mean(np.square(distances))

    fit = minimize_scalar(error, bounds=(-0.85, -0.38), method="bounded")
    z = fit.x
    w0, w1 = [c.plane(p, z) for c, p in zip(cams, curves)]
    # Far photo extends the wall-facing leg beyond the near lip. Merge in order.
    axis = unit(w0[-1] - w0[-3])
    extra = w1[(w1 - w0[-1]) @ axis > 0.005]
    path = np.vstack([w0, extra])
    return path, z, np.sqrt(fit.fun)


def make_step(mesh, cams, photo, grid_faces, orient):
    path, z, err = step_path(cams)
    axis = unit(np.gradient(path, axis=0))
    side = unit(np.cross(axis, [0, 0, 1]))
    angle = np.linspace(0, 2 * np.pi, 33)
    r = 0.011
    v = path[:, None] + r * (
        np.cos(angle)[None, :, None] * side[:, None] + np.sin(angle)[None, :, None] * [0, 0, 1]
    )
    # Rust, grayscale wear and pits from the top of the photographed bar.
    color = photo[0]
    px = wall_pixel(
        np.array([158, 81])
        + np.stack(np.meshgrid(np.linspace(-9, 9, 96), np.linspace(-2, 2, 40)), axis=-1),
        0,
    )
    tex = color(px)
    mat = mesh.material("Oxidised rung", tex, 0.84, 0.23, 0.8)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    texcoord = np.stack(np.broadcast_arrays(s[:, None] / s[-1], angle / (2 * np.pi)), axis=-1)
    f = grid_faces(len(path), len(angle))
    verts = v.reshape(-1, 3)
    target = (v - path[:, None]).reshape(-1, 3)[f].mean(1)
    mesh.add("Alexander step", verts, orient(verts, f, target), texcoord, mat)
    return dict(
        centerline_z_m=z, bend_reprojection_rms_px=err, radius_assumption_m=r, visible_path_m=s[-1]
    )


def delineator(mesh, cam, sample, xyz, orient, grid_faces, face_up, rings_faces):
    base = np.array(
        [
            [15, 354],
            [57, 332],
            [107, 345],
            [229, 454],
            [247, 494],
            [249, 516],
            [221, 547],
            [172, 536],
            [40, 417],
            [18, 393],
        ]
    )
    holes = [
        [[38, 353], [58, 342], [69, 348], [48, 365], [38, 361]],
        [[213, 510], [229, 493], [239, 501], [226, 523], [217, 530], [210, 525]],
    ]
    top = 0.052
    foot = cam.plane(marker_pixel(base), top)
    center = cam.plane(marker_pixel([144, 439]), top)
    slots = [cam.plane(marker_pixel(h), top) for h in holes]
    low = xyz(foot[:, :2], 0)
    rubber = mesh.material(
        "Scuffed black ballast",
        sample(
            marker_pixel(
                np.stack(np.meshgrid(np.linspace(52, 77, 94), np.linspace(421, 424, 24)), axis=-1)
            )
        )
        * 0.55,
        0.92,
        0,
        0.9,
    )
    verts = np.vstack([foot, *slots])
    face = face_up(verts, rings_faces([v[:, :2] for v in [foot, *slots]]))
    # Top atlas includes local scratches. Mask the post footprint before texturing the rubber.
    yy, xx = np.indices((470, 520))
    base_px = np.stack((xx / 2, yy / 2 + 323), axis=-1)
    tex = sample(marker_pixel(base_px))
    x, y = base_px.transpose(2, 0, 1)
    body = ((x - 149) * 0.78 + (y - 433) * 0.62) ** 2 / 43**2 + (
        (y - 433) * 0.78 - (x - 149) * 0.62
    ) ** 2 / 56**2 < 1.18
    tex[body] = sample(marker_pixel(np.stack((60 + x % 29, 376 + y % 18), axis=-1)))[body]
    mat = mesh.material("Ballast upper face", tex, 0.9, 0, 0.6)

    def base_uv(v):
        p = ((cam.project(v) + 0.5) / SIZE * [613, 460] - 0.5 - [334, 69]) / [259, 263] * [560, 565]
        return (p - [0, 323]) / [260, -235] + [0, 1]

    mesh.add("Alexander marker foot", verts, face, base_uv(verts), mat)

    def wall(name, a, b, reverse=False):
        n = len(a)
        i = np.arange(n)
        v = np.vstack([a, b])
        f = np.vstack([np.c_[i, (i + 1) % n, i + n], np.c_[(i + 1) % n, (i + 1) % n + n, i + n]])
        d = v[f].mean(1) - a.mean(0)
        d[:, 2] = 0
        # A measured dark ballast swatch continues around closed and slotted side faces.
        uv = np.c_[np.tile(np.linspace(0, 1, n), 2), np.repeat([1, 0], n)]
        mesh.add(name, v, orient(v, f, -d if reverse else d), uv, rubber)

    wall("Alexander marker ballast", foot, low)
    for i, h in enumerate(slots):
        wall("Alexander marker slot " + str(i), h, xyz(h[:, :2], 0), True)
    # The cone is nearly vertical in world coordinates. Image transverse sections
    # constrain the slightly flexible body without assuming that its silhouette is symmetric.
    sections = np.array(
        [
            [[109, 415], [167, 459]],
            [[117, 397], [183, 444]],
            [[148, 367], [197, 411]],
            [[196, 319], [248, 368]],
            [[239, 274], [293, 321]],
            [[299, 224], [347, 268]],
            [[357, 169], [406, 207]],
            [[425, 100], [472, 131]],
        ]
    )
    sec = marker_pixel(sections)
    m = sec.mean(1)
    ray = cam.rays(m)
    d = np.sum((center[:2] - cam.eye[:2]) * ray[:, :2], axis=1) / np.sum(ray[:, :2] ** 2, axis=1)
    centres = cam.eye + d[:, None] * ray
    centres[0] = center
    radius = (
        np.linalg.norm(sec[:, 1] - sec[:, 0], axis=1)
        * np.linalg.norm(centres - cam.eye, axis=1)
        / (2 * cam.f)
    )
    # Extend the base slightly at the first two rings to form the moulded skirt.
    radius[:2] *= [1.16, 1.08]
    angle = np.linspace(-np.pi, np.pi, 97)
    front = unit(cam.eye[:2] - center[:2])
    side = np.array([-front[1], front[0]])
    radial = np.cos(angle)[:, None] * front + np.sin(angle)[:, None] * side
    v = np.array([xyz(c[:2] + r * radial, c[2]) for c, r in zip(centres, radius)])
    z = np.linspace(centres[-1, 2], centres[0, 2], 640)
    theta = np.linspace(-np.pi, np.pi, 384)
    rad = np.interp(z, centres[:, 2], radius)
    mid = np.stack([np.interp(z, centres[:, 2], centres[:, j]) for j in range(2)], -1)
    # Fold hidden paint around the body, inset samples at the silhouette to avoid asphalt bleed.
    sample_radial = abs(np.cos(theta))[:, None] * front + 0.91 * np.sin(theta)[:, None] * side
    q = xyz(mid[:, None, :] + rad[:, None, None] * sample_radial, z[:, None])
    pick = cam.project(q)
    poly = np.vstack([sec[:, 0], sec[::-1, 1]])
    lo = np.int32(np.floor(poly.min(0))) - 8
    hi = np.int32(np.ceil(poly.max(0))) + 9
    mask = np.zeros(tuple((hi - lo)[::-1]), "u1")
    cv2.fillPoly(mask, [np.int32(np.rint(poly - lo))], 1)
    mask = cv2.erode(mask, np.ones((11, 11), "u1"))
    _, nearest = distance_transform_edt(mask == 0, return_indices=True)
    ij = np.int32(np.rint(pick - lo)).clip([0, 0], hi - lo - 1)
    pick = np.where(
        (mask[ij[..., 1], ij[..., 0]] == 0)[..., None],
        nearest[:, ij[..., 1], ij[..., 0]][::-1].transpose(1, 2, 0) + lo,
        pick,
    )
    sleeve = sample(pick)
    # In each transverse ring extend orange or reflective paint over the grazing
    # silhouette. Retain the full texture on the portion facing the street camera.
    midcolor = np.median(sleeve[:, 165:220].astype(float), axis=1)
    sim = np.linalg.norm(sleeve.astype(float) - midcolor[:, None], axis=2)
    sideweight = np.clip((abs(np.sin(theta)) - 0.72) / 0.23, 0, 1)
    mix = sideweight[None] * np.clip((sim - 55) / 25, 0, 1)
    sleeve = sleeve * (1 - mix[..., None]) + midcolor[:, None] * mix[..., None]
    mat = mesh.material("Orange and white reflective sleeve", sleeve, 0.64, 0, 0.32)
    f = grid_faces(len(centres), len(angle))
    vv = v.reshape(-1, 3)
    uv = np.stack(
        np.broadcast_arrays(
            (angle + np.pi) / (2 * np.pi), ((centres[:, 2] - z[-1]) / np.ptp(z))[:, None]
        ),
        -1,
    )
    mesh.add(
        "Alexander marker striped sleeve",
        vv,
        orient(vv, f, (v - centres[:, None]).reshape(-1, 3)[f].mean(1)),
        uv,
        mat,
    )
    shoulder = v[-1, :-1]
    cap_rgb = sample(
        marker_pixel(
            np.stack(np.meshgrid(np.linspace(425, 439, 28), np.linspace(125, 134, 20)), axis=-1)
        )
    )
    orange = mesh.material("Orange moulding shoulder", cap_rgb, 0.74, 0, 0.32)
    mesh.add(
        "Alexander marker shoulder",
        shoulder,
        face_up(shoulder, rings_faces([shoulder[:, :2]])),
        0.5 + (shoulder[:, :2] - centres[-1, :2]) / (2 * radius[-1]),
        orange,
    )
    # Handhold with an actual through-slot. Both faces are traced in the plane of the handle.
    outer = [
        [428, 98],
        [445, 81],
        [478, 46],
        [505, 27],
        [523, 20],
        [536, 22],
        [544, 30],
        [542, 44],
        [526, 67],
        [469, 133],
    ]
    inner = [[484, 76], [495, 58], [512, 41], [525, 33], [530, 35], [526, 43], [510, 62], [493, 77]]
    anchor = centres[-1]
    norm = np.r_[front, 0.0]
    contours = [cam.plane(marker_pixel(p), anchor @ norm, norm) for p in [outer, inner]]
    flat = lambda v: np.c_[v[:, :2] @ side, v[:, 2]]
    v = np.vstack(contours)
    f = rings_faces([flat(p) for p in contours])
    primitive = mesh.material(
        "Worn handle polymer",
        sample(
            marker_pixel(
                np.stack(np.meshgrid(np.linspace(476, 482, 50), np.linspace(65, 72, 30)), axis=-1)
            )
        ),
        0.74,
        0,
        0.4,
    )
    xx, yy = np.meshgrid(np.linspace(417, 551, 270), np.linspace(15, 139, 250))
    grip = sample(marker_pixel(np.stack((xx, yy), -1)))
    hsv = cv2.cvtColor(grip, cv2.COLOR_RGB2HSV)
    red = ((hsv[..., 0] < 18) | (hsv[..., 0] > 167)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 56)
    _, nn = distance_transform_edt(~red, return_indices=True)
    cap = mesh.material("Photographed slotted grip", grip[nn[0], nn[1]], 0.73, 0, 0.35)
    local = np.vstack([outer, inner])
    face_uv = (local - [417, 15]) / [134, -124] + [0, 1]
    for sign, name in [(1, "front"), (-1, "back")]:
        vv = v + sign * 0.009 * norm
        mesh.add("Alexander marker handle " + name, vv, orient(vv, f, sign * norm), face_uv, cap)
    for i, p in enumerate(contours):
        n = len(p)
        j = np.arange(n)
        v = np.vstack([p + 0.009 * norm, p - 0.009 * norm])
        f = np.vstack([np.c_[j, (j + 1) % n, j + n], np.c_[(j + 1) % n, (j + 1) % n + n, j + n]])
        mesh.add(
            "Alexander marker handle edge " + str(i),
            v,
            orient(v, f, (v[f].mean(1) - p.mean(0)) * (1 if i == 0 else -1)),
            flat(v),
            primitive,
        )
    return dict(
        marker_height_m=float(contours[0][:, 2].max()),
        ballast_height_assumption_m=top,
        base_origin_m=xyz(center[:2], 0).tolist(),
    )
