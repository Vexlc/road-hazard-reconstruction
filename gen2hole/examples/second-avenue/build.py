"""Build a cut-out road, separate upright cones, and three singly observed cavity contents."""

import argparse
import hashlib
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from camera import Camera, HERE, S

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    face_up,
    rings_faces,
    flatten_scene,
    grid,
    grid_faces,
    make_one,
    rect,
    texture_uv,
    write_json,
    xyz,
)
from surfaces import refine

RINGS = S["surfaces"]


def sample(photo, pix):
    return cv2.remap(
        photo,
        *((pix + 0.5) * photo.shape[1] / 820 - 0.5).astype("f4").transpose(2, 0, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def orient(v, f, towards):
    a, b, c = v[f].transpose(1, 0, 2)
    flip = (np.cross(b - a, c - a) * towards).sum(1) < 0
    f = f.copy()
    f[flip] = f[flip, ::-1]
    return f


def strip(mesh, name, rows, mat, uv, target):
    v = np.vstack(rows)
    n = len(rows[0])
    j = np.arange(n)
    f = np.vstack((np.c_[j, (j + 1) % n, j + n], np.c_[(j + 1) % n, (j + 1) % n + n, j + n]))
    mesh.add(name, v, orient(v, f, target(v[f].mean(1))), uv(v), mat)


def photo_patch(mesh, name, contour, depth, material, uv):
    points = xyz(np.asarray(contour, float), 0)
    f = rings_faces([points[:, :2]])
    points, f = refine(points, f, 5.0)
    image = points[:, :2]
    v = cam.plane(image, -depth(image))
    mesh.add(name, v, orient(v, f, cam.eye - v[f].mean(1)), uv(v), material)


def depth_limit(p):
    # Exit of a source ray through the vertical extrusion of the traced rim.
    start = cam.plane(p)[:, :2]
    direction = (start - cam.eye[:2]) / cam.eye[2]
    limit = np.full(len(p), np.inf)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for a, b in zip(rim, np.roll(rim, -1, axis=0)):
        edge = b[:2] - a[:2]
        denom = cross(direction, edge)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross(a[:2] - start, edge) / denom
            u = cross(a[:2] - start, direction) / denom
        valid = (denom < 0) & (t >= 0) & (u >= 0) & (u <= 1)
        limit[valid] = np.minimum(limit[valid], t[valid])
    return np.where(np.isfinite(limit), np.minimum(limit, 0.65), 0.0)


def contents(mesh, photo):
    # One cropped atlas retains native sampling for all camera-facing cavity patches.
    off = np.array([280, 345])
    crop = photo[690:1230, 560:1520]
    mat = mesh.material("Photographic cavity faces", crop, 0.84, 0, 0.65)

    def uv(v):
        return texture_uv(cam.project(v) - off, [480, 270])

    def shape(p, kind):
        x, y = p.T
        if kind == "black_bag_visible":
            d = 0.23 - 0.12 * np.exp(-(((x - 423) / 88) ** 2) - ((y - 497) / 75) ** 2)
        elif kind == "cardboard_visible":
            fold = np.abs((x - 596) * 0.63 - (y - 511))
            d = 0.188 + 0.00042 * (575 - x) - 0.00095 * np.minimum(fold, 48)
        else:
            # Rounded visible portion of the tipped cone. The front road lip clips its wide end.
            w = np.clip((y - 524) / 70, 0, 1)
            axis = 473 + 57 * w
            radius = 8 + 31 * w
            d = 0.20 - 0.082 * w - 0.035 * np.sqrt(np.maximum(0, 1 - ((x - axis) / radius) ** 2))
        return np.minimum(d, depth_limit(p) * 0.92)

    for key, name in [
        ("black_bag_visible", "Black sack"),
        ("cardboard_visible", "Folded cardboard"),
        ("fallen_pylon_visible", "Trapped cone"),
    ]:
        photo_patch(mesh, name, RINGS[key], lambda p: shape(p, key), mat, uv)
        # Close the photographed fronts toward the back of each item. The rear loop
        # shrinks in the reference image, so it never protrudes past the traced front.
        pixels = np.array(RINGS[key], float)
        centre = pixels.mean(0)
        mid = centre + 0.88 * (pixels - centre)
        back = centre + 0.58 * (pixels - centre)
        front = cam.plane(pixels, -shape(pixels, key))
        shoulder = cam.plane(mid, -np.minimum(shape(mid, key) + 0.065, depth_limit(mid) * 0.96))
        rear = cam.plane(back, -np.minimum(shape(back, key) + 0.14, depth_limit(back) * 0.975))
        source_mid = {
            "Black sack": [408, 486],
            "Folded cardboard": [551, 539],
            "Trapped cone": [505, 562],
        }[name]

        # Sample colors inside the exposed face. Reserve the extrapolation for hidden surfaces.
        def rear_uv(v):
            q = source_mid + 0.23 * (cam.project(v) - source_mid)
            return texture_uv(q - off, [480, 270])

        for suffix, top, bottom in [(" side", front, shoulder), (" reverse", shoulder, rear)]:
            strip(
                mesh,
                name + suffix,
                [top, bottom],
                mat,
                rear_uv,
                lambda c: c - (front.mean(0) + rear.mean(0)) * 0.5,
            )
        f = rings_faces([back])
        mesh.add(
            name + " back",
            rear,
            orient(rear, f, rear[f].mean(1) - front.mean(0)),
            rear_uv(rear),
            mat,
        )

        # Continue the portions that run behind the near asphalt lip. Directly
        # undercut the road by 1 cm so the hidden ends stay buried on a reverse view.
        clip = {
            "Black sack": list(range(30, 37)),
            "Folded cardboard": list(range(16, 24)),
            "Trapped cone": list(range(7, 12)),
        }[name]
        edge = pixels[clip]
        edge_front = front[clip]
        buried = cam.plane(edge)
        buried[:, :2] += 0.01 * (cam.eye[:2] - buried[:, :2])
        buried[:, 2] = edge_front[:, 2] - 0.04
        v = np.vstack((edge_front, buried))
        f = grid_faces(2, len(clip))
        # Texture along the visible object's grain, away from the road-side pixels.
        donor = source_mid + 0.38 * (edge - source_mid)
        uv_edge = texture_uv(donor - off, [480, 270])
        uv_deep = texture_uv(0.78 * donor + 0.22 * np.array(source_mid) - off, [480, 270])
        mesh.add(name + " continuation", v, face_up(v, f), np.vstack((uv_edge, uv_deep)), mat)

    # In the source view the exposed far side ends against the contents or deep shadow.
    # Texture the upper wall directly, and extend below with same-site dark material.
    picks = np.array(S["wall_crest_indices"])
    z = cam.closest_height(S["wall_toes_px"], rim[picks, :2])
    low_z = np.interp(np.arange(len(rim)), picks, z)
    low = xyz(rim[:, :2], low_z)
    inward = lambda c: np.c_[-c[:, :2] + rim[:, :2].mean(0), np.zeros(len(c))]
    # Only the far arc is photographically supported. All other wall faces use continuation.
    deeper = xyz(rim[:, :2], -0.52)
    # Unwrap the walls in station/depth coordinates. Reuse original strata between
    # the observed cap and the first occluding item. On unseen wall, continue a dark
    # aggregate swatch, with exposure fading gradually below the last visible sample.
    count = len(rim)
    nx, ny = count * 30, 384
    u = (np.arange(nx) + 0.5) * count / nx
    i = u.astype(int)
    t = (u - i)[:, None]
    xy = rim[i, :2] * (1 - t) + rim[(i + 1) % count, :2] * t
    levels = -(np.arange(ny) + 0.5) * 0.52 / ny
    wall_points = xyz(np.broadcast_to(xy, (ny, nx, 2)), np.broadcast_to(levels[:, None], (ny, nx)))
    puv = cam.project(wall_points)
    color = sample(photo, puv).astype(float)
    field = np.zeros((615, 820), "u1")
    cv2.fillPoly(field, [np.int32(RINGS["outer_cut"])], 1)
    for key in list(RINGS)[1:]:
        cv2.fillPoly(field, [np.int32(RINGS[key])], 0)
    inside = cv2.remap(field, *puv.transpose(2, 0, 1).astype("f4"), cv2.INTER_NEAREST)
    far = (i >= 1) & (i <= 25)
    valid = (inside > 0) & far
    # Reject later reappearances of the silhouette below an occluding item.
    valid[:4, far] = True
    valid = np.logical_and.accumulate(valid, axis=0)
    last = np.maximum(valid.sum(0) - 1, 0)
    row, col = np.indices((ny, nx))
    # Warm, granular fill from under the lip at the left edge of the excavation.
    grain_px = np.stack(
        [
            313 + 11 * (1 + np.sin(col * 0.025 + np.sin(row * 0.05))),
            471 + 6 * (1 + np.sin(row * 0.13 + col * 0.011)),
        ],
        -1,
    )
    grain = sample(photo, grain_px).astype(float)
    shade = (0.54 * np.exp(levels / 0.18) + 0.18)[:, None, None]
    continuation = grain * shade
    fade = np.clip(1 - (row - last) / 21, 0, 1) * far
    tex = color[np.minimum(row, last), col] * fade[..., None] + continuation * (1 - fade[..., None])
    wallmat = mesh.material("Shaded cut continuation", tex, 0.96, 0, 0.8)
    for i in range(count):
        j = (i + 1) % count
        observed = i in picks[:-1]
        for k, (top, bottom) in enumerate([(rim, low), (low, deeper)]):
            v = np.array([top[i], top[j], bottom[i], bottom[j]])
            f = np.array([[0, 1, 2], [1, 3, 2]])
            supported = observed and k == 0
            coords = np.c_[
                [i / count, (i + 1) / count, i / count, (i + 1) / count], 1 + v[:, 2] / 0.52
            ]
            mesh.add(
                ("Exposed asphalt wall " if supported else "Inferred wall ") + f"{i}-{k}",
                v,
                orient(v, f, inward(v[f].mean(1))),
                uv(v) if supported else coords,
                mat if supported else wallmat,
            )
    floor = mesh.material("Optical pocket shadow", photo[942:954, 1220:1260] * 0.50, 0.98, 0, 0.4)
    mesh.add(
        "Optical pocket",
        deeper,
        face_up(deeper, rings_faces([deeper[:, :2]])),
        deeper[:, :2],
        floor,
    )
    return -z


def cones(mesh, photo):
    for idx, (foot, profiles) in enumerate(zip(cam.feet, S["cone_profiles_px"])):
        name = "Traffic cone " + str(idx + 1)
        center = foot.mean(0)
        sx = np.array(profiles)  # sampled cross sections, from tip to skirt
        x, y = sx[:, :2].mean(1), sx[:, 2]
        zs = cam.closest_height(np.c_[x, y], center[:2])
        zs = np.clip(zs, center[2] + 0.001, cam.h)
        zs[0] = cam.h
        zs[-1] = center[2] + 0.003
        rings = []
        angles = np.linspace(-np.pi, np.pi, 97)
        for left, right, row, z in zip(sx[:, 0], sx[:, 1], y, zs):
            mid = cam.plane([(left + right) / 2, row], z)
            # A horizontal image chord through the cone has a metric half-width at its centre depth.
            r = (right - left) / 2 * np.linalg.norm(mid - cam.eye) / cam.f
            rings.append(xyz(mid[:2] + r * np.stack([np.sin(angles), -np.cos(angles)], axis=1), z))
        rings = np.array(rings)
        # Map each ring to horizontal sections in the photo, reflecting the hidden half.
        h = 384
        rows = np.linspace(y[0] + 1, y[-1] - 2, h)
        left = np.interp(rows, y, sx[:, 0]) + 1
        right = np.interp(rows, y, sx[:, 1]) - 1
        a = np.linspace(-np.pi, np.pi, 512)
        pix = np.stack(
            np.broadcast_arrays(
                left[:, None] + (right - left)[:, None] * (0.5 + 0.485 * np.sin(a)), rows[:, None]
            ),
            -1,
        )
        mat = mesh.material(name + " striped sleeve", sample(photo, pix), 0.67, 0, 0.25)
        v = rings.reshape(-1, 3)
        f = grid_faces(len(y), len(angles))
        ref = np.repeat(rings[:, :-1, :].mean(1), len(angles), axis=0)
        uv = np.stack(
            np.broadcast_arrays(
                np.linspace(0, 1, len(angles)), 1 - ((y - y[0]) / (y[-1] - y[0]))[:, None]
            ),
            -1,
        )
        mesh.add(name + " body", v, orient(v, f, (v - ref)[f].mean(1)), uv, mat)
        # The actual tops show a small dark opening with a rolled orange rim.
        outer = rings[0, :-1]
        top = outer.mean(0)
        inner = top + 0.56 * (outer - top)
        orange = mesh.material(
            name + " edge",
            photo[int(2 * (y[-2] - 4)) : int(2 * y[-2]), int(2 * x[-2]) : int(2 * x[-2]) + 16],
            0.77,
            0,
            0.3,
        )
        strip(
            mesh,
            name + " crown",
            [outer, inner],
            orange,
            lambda v: v[:, :2],
            lambda c: np.tile([0, 0, 1], (len(c), 1)),
        )
        mesh.add(
            name + " top opening",
            inner - [0, 0, 0.003],
            face_up(inner, rings_faces([inner[:, :2]])),
            inner[:, :2],
            "Optical pocket shadow",
        )
        # Use the photographed exposed rubber around the footprint, fill only the body occlusion.
        b = np.array(S["upright_cones"][idx]["base"])
        lo = np.floor(b.min(0) - 3).astype(int)
        hi = b.max(0) + 3
        yy, xx = np.indices(tuple(((hi - lo) * 2)[::-1]))
        pp = np.stack((xx, yy), -1) / 2 + lo
        tex = sample(photo, pp)
        body = (
            (pp[..., 1] <= y[-1] + 3)
            & (pp[..., 0] > np.interp(pp[..., 1], y, sx[:, 0]) - 2)
            & (pp[..., 0] < np.interp(pp[..., 1], y, sx[:, 1]) + 2)
        )
        tex[body] = sample(photo, np.array([[[x[-1], y[-1] + 9]]]))[0, 0]
        rubber = mesh.material(name + " foot rubber", tex, 0.85, 0, 0.5)
        base = xyz(foot[:, :2], 0)
        strip(
            mesh,
            name + " base edge",
            [base, foot],
            orange,
            lambda v: v[:, :2],
            lambda c: c - center,
        )
        mesh.add(
            name + " base",
            foot,
            face_up(foot, rings_faces([foot[:, :2]])),
            texture_uv(cam.project(foot) - lo, hi - lo),
            rubber,
        )


def pavement(mesh, photo, span):
    xy = grid(span, 0.0025)
    pix = cam.project(xyz(xy, 0))
    terrain = np.ones((615, 820), "u1")
    for i, c in enumerate(S["upright_cones"]):
        s = np.array(S["cone_profiles_px"][i])
        body = np.vstack([s[:, [0, 2]], s[::-1][:, [1, 2]]])
        for poly in (body, c["base"]):
            cv2.fillPoly(terrain, [np.int32(poly)], 0)
    edge = cv2.distanceTransform(terrain, cv2.DIST_L2, 5)
    x, y = pix.transpose(2, 0, 1)
    w = np.clip(
        np.minimum.reduce(
            [
                x,
                y,
                819 - x,
                614 - y,
                cv2.remap(edge, x.astype("f4"), y.astype("f4"), cv2.INTER_LINEAR) - 3,
            ]
        )
        / 12,
        0,
        1,
    )
    # Fold an undamaged field swatch into areas hidden by upright cones or outside the frame.
    tile = 1 - abs((xy / np.array([0.30, 0.27]) % 2) - 1)
    fill = sample(photo, tile * [113, 83] + [174, 171])
    tex = sample(photo, pix) * w[..., None] + fill * (1 - w[..., None])
    mat = mesh.material("Second Avenue road", tex, 0.95, 0, 0.75)
    outline = rect(span)
    v = np.vstack((outline, rim))
    mesh.add(
        "Road around the aperture",
        v,
        face_up(v, rings_faces([outline[:, :2], rim[:, :2]])),
        v[:, :2] / span + 0.5,
        mat,
        road=True,
    )


def receiver_sky(folder, span):
    import open3d as o3d

    scene = Surface(folder / "asset.glb", exclude={1, 180, 181, 182, 183, 184}).scene
    origins = xyz(grid(span, 0.01), 0.001)
    loss = np.zeros(origins.shape[:2], "f4")
    for i in range(32):
        z = (i + 0.5) / 32
        angle = i * np.pi * (3 - np.sqrt(5))
        direction = np.array(
            [np.cos(angle) * np.sqrt(1 - z * z), np.sin(angle) * np.sqrt(1 - z * z), z]
        )
        rays = np.concatenate((origins, np.broadcast_to(direction, origins.shape)), -1)
        loss += (
            z
            / 16
            * scene.test_occlusions(
                o3d.core.Tensor(rays.astype("f4")), tfar=2.0, nthreads=2
            ).numpy()
        )
    return loss


def build(out):
    global cam, rim
    source = HERE / "references/site.jpg"
    if hashlib.sha256(source.read_bytes()).hexdigest() != S["source_sha256"]:
        raise ValueError("Source JPEG checksum mismatch")
    cam = Camera()
    rim = cam.plane(RINGS["outer_cut"])
    photo = np.array(
        Image.open(source).convert("RGB").resize((1640, 1230), Image.Resampling.LANCZOS)
    )
    mesh = Mesh()
    wall = contents(mesh, photo)
    cones(mesh, photo)
    span = (
        np.ceil(
            (np.maximum(abs(rim[:, :2]).max(0), abs(cam.feet[:, :, :2]).max((0, 1))) + 0.13) * 100
        )
        * 0.02
    )
    pavement(mesh, photo, span)
    xy = grid(span, 0.005)
    folder = out / "series/assets/pothole/P584"
    cache = out / "series/mesh_cache"
    dims = sorted(cv2.minAreaRect(rim[:, :2].astype("f4"))[1], reverse=True)
    meta = dict(
        id="P584",
        category="pothole",
        family="second_avenue",
        state="debris_obscured_pothole",
        seed=29584,
        label="第二大道大坑、警示锥与坑内杂物",
        source=S["source"],
        source_author=S["author"],
        source_license=S["license"],
        source_sha256=S["source_sha256"],
        source_eye_m=cam.eye.tolist(),
        assumed_cone_height_m=cam.h,
        aperture_box_m=dims,
        wall_visible_drop_m=wall.tolist(),
        floor_depth_observed=False,
        optical_closure_depth_m=0.52,
        surface_dimensions_m=span.tolist(),
    )
    mesh.save(folder, cache, meta, dict(width=span[0], length=span[1]))
    flatten_scene(folder / "asset.glb")
    mesh.parts = [
        p
        for p in mesh.parts
        if not p[0].startswith(
            ("Inferred wall", "Optical pocket", "Black sack", "Folded cardboard", "Trapped cone")
        )
    ]
    (folder / "bounded.glb").write_bytes(mesh.glb_bytes())
    flatten_scene(folder / "bounded.glb")
    origins = xyz(xy, 2)
    rays = np.broadcast_to([0.0, 0.0, -1.0], origins.shape)
    dist, part, _ = Surface(folder / "asset.glb").cast(origins, rays)
    lit = np.isfinite(dist)
    np.savez_compressed(
        folder / "surface.npz",
        width=span[0],
        length=span[1],
        height=np.where(lit, 2 - dist, 0).astype("f4"),
        alpha=lit,
        first_part=part,
        defect_mask=part > 1,
        installation_mask=part > 1,
        water_mask=np.zeros_like(lit),
        photographic_rgb=True,
        registered_relief=True,
        cast_receiver_sunlight=True,
        receiver_sky_blocked=receiver_sky(folder, span),
        hypothesized_depth=np.isin(part, [181, 182, 183, 184]),
    )
    row = make_one(folder, cache, point_limit=280000, kernel_factors=(0.70, 0.16))
    for k, f in (
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("metadata", "metadata.json"),
        ("surface", "surface.npz"),
    ):
        row[k] = "pothole/P584/" + f
    write_json(out / "series/assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
