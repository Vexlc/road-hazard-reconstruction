"""Photo-registered road ends, gravel banks, broken slabs and a pipe crossing a flooded washout."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import least_squares
from stereo import load

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    xyz,
    grid,
    face_up,
    grid_faces,
    refine,
    rings_faces,
    normalize,
    make_one,
    flatten_scene,
    write_json,
)


def distance_to_trace(p, trace):
    a = trace[:-1]
    e = np.diff(trace, axis=0)
    t = np.clip(((p[:, None] - a) * e).sum(2) / (e * e).sum(1), 0, 1)
    delta = p[:, None] - a - t[:, :, None] * e
    d = (delta * delta).sum(2)
    return np.sqrt(d.min(1))


def crossing(pair):
    s = pair.s["blue_tube"]
    a = np.asarray(s["west_trace"], float)
    b = np.asarray(s["northwest_trace"], float)
    # Both photographs cover the same length up to the right image boundary in the close view.
    # Keep its continuation on the west-view bearing with a smooth unobserved height.
    overlap = a[:, 1] < 342

    def residual(z):
        w = pair.plane(a, z)
        return np.r_[
            distance_to_trace(pair.project(w[overlap], 1), b),
            np.diff(z, 2) * 28,
            np.diff(z) * 6,
            (z[[0, 4]] + 1.16) * 12,
            (z[~overlap] + 1.16) * 11,
        ]

    z = least_squares(residual, np.full(len(a), -1.16), bounds=(-1.40, -0.63), loss="soft_l1").x
    path = pair.plane(a, z)
    # Estimate radius from apparent half-widths in the northwest picture, in a
    # direction normal to the projection of the local cylinder axis.
    measurements = []
    for x, y, half in s["source_radius_samples_northwest"]:
        fine = np.vstack([np.linspace(l, r, 40, endpoint=False) for l, r in zip(path, path[1:])])
        uv = pair.project(fine, 1)
        j = np.argmin(np.linalg.norm(uv - [x, y], axis=1))
        point = fine[j]
        axis = normalize(fine[min(j + 1, len(fine) - 1)] - fine[max(0, j - 1)])
        normal = normalize(np.cross(axis, point - pair.eye(1)))
        px_per_m = (
            np.linalg.norm(
                pair.project(point + 0.01 * normal, 1) - pair.project(point - 0.01 * normal, 1)
            )
            / 0.02
        )
        measurements.append(half / px_per_m)
    return path, float(np.median(measurements)), measurements


class Scene:
    def __init__(self, pair):
        self.pair = pair
        self.s = pair.s
        self.mesh = Mesh()
        self.groups = defaultdict(list)
        self.images = []
        for key in ("west", "northwest"):
            p = HERE / "references" / (key + ".jpg")
            assert hashlib.sha256(p.read_bytes()).hexdigest() == self.s["views"][key]["sha256"]
            self.images.append(np.asarray(Image.open(p).convert("RGB")))
        # Remove the crossing from the bank's albedo before adding its independent geometry.
        bare = []
        for im, key in zip(self.images, ("west_trace", "northwest_trace")):
            mask = np.zeros(im.shape[:2], "u1")
            pts = self.s["blue_tube"][key]
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                width = (
                    6 + int(13 * i / (len(pts) - 2))
                    if key == "west_trace"
                    else 11 + int(34 * i / (len(pts) - 2))
                )
                cv2.line(mask, tuple(a), tuple(b), 255, width)
            if key == "west_trace":
                for side_pipe in self.s["right_conduits_west"]:
                    points = side_pipe["trace"]
                    for j, (a, b) in enumerate(zip(points, points[1:])):
                        cv2.line(mask, tuple(a), tuple(b), 255, 8 + 2 * j)
            else:
                cv2.line(mask, (133, 471), (135, 580), 255, 31)
            bare.append(cv2.inpaint(im, mask, 5, cv2.INPAINT_TELEA))
        self.atlas = self.mesh.material(
            "Two field exposures: asphalt, gravel and cut faces",
            np.hstack(bare),
            0.94,
            0,
            0.54,
            lossless_color=True,
        )
        self.gravel = self.mesh.material(
            "Inferred submerged gravel", self.images[1][298:321, 201:283], 0.97, 0, 0.5
        )
        self.water = self.mesh.material(
            "Standing water under overcast sky",
            np.full((16, 16, 3), [100, 95, 73], "u1"),
            0.065,
            0,
            0,
        )
        self.mesh.ior[self.water] = 1.333

    def uv(self, pixel, view):
        return np.c_[((pixel[:, 0] + 0.5) / 1024 + view) / 2, 1.0 - (pixel[:, 1] + 0.5) / 683]

    def add(self, name, v, f, uv, material):
        self.groups[(name, material)].append((np.asarray(v), np.asarray(f), np.asarray(uv)))

    def surface(self, name, pixels, z, view, anchors=None, rough=0.0):
        poly = np.asarray(pixels, float)
        z = np.broadcast_to(z, len(poly))
        # Subdivide in the recording plane before ray lifting. A strongly oblique cut can
        # fold back in a plan view, while the photographed boundary remains unambiguous.
        v, f = refine(xyz(poly, 0.0), rings_faces([poly]), 22.0)
        coords = v[:, :2]
        known = xyz(poly, z) if anchors is None else np.vstack([xyz(poly, z), anchors])
        interp = LinearNDInterpolator(known[:, :2], known[:, 2])
        height = interp(coords)
        invalid = ~np.isfinite(height)
        if invalid.any():
            height[invalid] = NearestNDInterpolator(known[:, :2], known[:, 2])(coords[invalid])
        if rough:
            # Suppress roughness at the shared shore and pavement boundaries.
            edge = distance_to_trace(coords, np.vstack([poly, poly[:1]]))
            weight = np.clip(edge / 9, 0, 1)
            height += (
                rough
                * weight
                * np.sin(0.13 * coords[:, 0] + 0.032 * coords[:, 1])
                * np.cos(0.15 * coords[:, 1])
            )
        world = self.pair.plane(coords, height, view)
        f = face_up(world, f)
        if view == 0 and name in ("Fracture lip and exposed bed", "Photographed verge"):
            alternate = self.pair.project(world, 1)
            shore = np.asarray(self.s["near_shore_northwest"])
            boundary = np.float32(np.vstack([shore, [[730, 682], [0, 682]]]))
            centre = alternate[f].mean(1)
            inside = np.array(
                [cv2.pointPolygonTest(boundary, tuple(pt), False) > 0 for pt in centre]
            )
            areas = lambda tri: abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
            use = inside & (areas(alternate[f]) > 1.85 * areas(coords[f]))
            if use.any():
                selected = f[use].ravel()
                self.add(
                    name,
                    world[selected],
                    np.arange(len(selected)).reshape(-1, 3),
                    self.uv(alternate[selected], 1),
                    self.atlas,
                )
            f = f[~use]
        if len(f):
            self.add(name, world, f, self.uv(coords, view), self.atlas)
        return interp

    def cut(self, pixels, view, thickness):
        top = self.pair.plane(pixels, 0.0, view)
        low = top.copy()
        low[:, 2] -= thickness
        base = self.pair.project(low, view)
        # Fine along-edge interpolation preserves projective texture from the front view.
        v = np.stack([top, low])
        f = grid_faces(2, len(top))
        v = v.reshape(-1, 3)
        a, b, c = v[f].transpose(1, 0, 2)
        flip = np.sum(np.cross(b - a, c - a) * (self.pair.eye(view) - a), axis=1) < 0
        f[flip] = f[flip, ::-1]
        self.add(
            "Detached pavement edges", v, f, self.uv(self.pair.project(v, view), view), self.atlas
        )
        return base

    def terrain(self, view):
        s = self.s
        key = ("west", "northwest")[view]
        end = ("near", "far")[view]
        lip = np.array(s[end + "_lip_" + key])
        water = np.asarray(s[end + "_shore_" + key])
        if view:
            grade = lip[3:8]
            outer = s["far_road_outer_northwest"]
        else:
            grade = lip
            outer = s["near_road_outer_west"]
        self.surface("Detached pavement top", np.vstack([grade, outer]), 0.0, view)
        thickness = 0.10 if view == 0 else 0.12
        edge = self.cut(lip, view, thickness)
        side = np.asarray(s[end + "_terrain_outer_" + key])
        if view:
            poly = np.vstack([edge, side[:, :2], water[::-1]])
            z = np.r_[
                np.full(len(lip), -thickness),
                side[:, 2],
                np.full(len(water), s["nominal_water_z_m"]),
            ]
        else:
            poly = np.vstack([edge, side[::-1, :2], water[::-1]])
            z = np.r_[
                np.full(len(lip), -thickness),
                side[::-1, 2],
                np.full(len(water), s["nominal_water_z_m"]),
            ]
        anchors = np.array(
            s["terrain_depth_anchors_west"] if not view else s["far_terrain_anchors_northwest"]
        )
        interp = self.surface("Fracture lip and exposed bed", poly, z, view, anchors, rough=0.016)
        for slab in s["near_slabs_west"] if not view else [s["far_slab_northwest"]]:
            outline = np.array(slab["outline"])
            height = interp(outline)
            if not np.isfinite(height).all():
                raise ValueError("Slab extends beyond the registered terrain")
            height += slab["rise_m"]
            self.surface("Photographed bed fragment", outline, height, view)
            top = self.pair.plane(outline, height, view)
            base = top.copy()
            base[:, 2] -= slab["rise_m"] + 0.035
            n = len(top)
            i = np.arange(n)
            j = (i + 1) % n
            v = np.vstack([top, base])
            f = np.vstack([np.c_[i, j, n + i], np.c_[j, n + j, n + i]])
            a, b, c = v[f].transpose(1, 0, 2)
            flip = np.sum(np.cross(b - a, c - a) * (a - v.mean(0)), axis=1) < 0
            f[flip] = f[flip, ::-1]
            self.add(
                "Detached pavement edges",
                v,
                f,
                self.uv(self.pair.project(v, view), view),
                self.atlas,
            )
        return self.pair.plane(water, s["nominal_water_z_m"], view)

    def pool(self, near, far):
        z = self.s["nominal_water_z_m"]
        poly = np.vstack([near, far[::-1]])
        vertices, f = refine(poly, rings_faces([poly[:, :2]]), 0.36)
        f = face_up(vertices, f)
        self.add("Water surface", vertices, f, np.zeros((len(vertices), 2)), self.water)
        # The depth below opaque water cannot be recovered from the views.
        bed = vertices.copy()
        bed[:, 2] = z - 0.38 + 0.035 * np.sin(bed[:, 0] * 2.1) * np.cos(bed[:, 1] * 1.9)
        uv = (bed[:, :2] - bed[:, :2].min(0)) / np.ptp(bed[:, :2], axis=0)
        self.add("Subgrade bearing", bed, f, uv, self.gravel)

    def tube(self, name, path, r, mat, color):
        centers = np.vstack(
            [
                np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.07)), endpoint=False)
                for a, b in zip(path, path[1:])
            ]
            + [path[-1:]]
        )
        tangent = normalize(np.gradient(centers, axis=0))
        side = normalize(np.cross(tangent, [0.0, 0.0, 1.0]))
        up = np.cross(side, tangent)
        angle = np.linspace(0, 2 * np.pi, 49)
        axis = (
            np.cos(angle)[None, :, None] * up[:, None, :]
            + np.sin(angle)[None, :, None] * side[:, None, :]
        )
        v = centers[:, None, :] + r * axis
        f = grid_faces(*v.shape[:2])
        v = v.reshape(-1, 3)
        axis = axis.reshape(-1, 3)
        a, b, c = v[f].transpose(1, 0, 2)
        flip = (np.cross(b - a, c - a) * axis[f[:, 0]]).sum(1) < 0
        f[flip] = f[flip, ::-1]
        t = np.broadcast_to(
            np.linspace(0.01, 0.99, len(centers))[:, None], (len(centers), len(angle))
        )
        uv = np.stack([t, np.broadcast_to(angle / (2 * np.pi), t.shape)], axis=-1).reshape(-1, 2)
        self.add(name, v, f, uv, mat)
        # Leave visible mouths open with a bore, inset dark termination and annular wall.
        for k, sign in ((0, 1), (-1, -1)):
            ring = axis.reshape(len(centers), len(angle), 3)[k, :-1]
            edge = centers[k] + r * ring
            inside = centers[k] + 0.78 * r * ring
            back = inside + sign * 0.23 * tangent[k]
            n = len(ring)
            i = np.arange(n)
            j = (i + 1) % n
            v = np.vstack([edge, inside, back, back.mean(0)[None]])
            f = np.vstack(
                [
                    np.c_[i, j, i + n],
                    np.c_[j, j + n, i + n],
                    np.c_[i + n, j + n, i + 2 * n],
                    np.c_[j + n, j + 2 * n, i + 2 * n],
                    np.c_[i + 2 * n, j + 2 * n, np.full(n, 3 * n)],
                ]
            )
            # Orient ends outward and use the inward-facing bore wall.
            a, b, c = v[f].transpose(1, 0, 2)
            orient = np.tile(-sign * tangent[k], (len(f), 1))
            orient[2 * n : 4 * n] = -(a[2 * n : 4 * n] - centers[k])
            flip = np.sum(np.cross(b - a, c - a) * orient, axis=1) < 0
            f[flip] = f[flip, ::-1]
            self.add(name + " bore", v, f, np.zeros((len(v), 2)), color)

    def finalize(self):
        bounds = np.vstack([v for group in self.groups.values() for v, _, _ in group])
        shift = np.r_[(bounds[:, :2].min(0) + bounds[:, :2].max(0)) / 2, 0.0]
        for (name, material), pieces in self.groups.items():
            verts = []
            faces = []
            uv = []
            offset = 0
            for v, f, u in pieces:
                verts.append(v - shift)
                faces.append(f + offset)
                uv.append(u)
                offset += len(v)
            self.mesh.add(name, np.vstack(verts), np.vstack(faces), np.vstack(uv), material)
        return shift, np.ptp(bounds[:, :2], axis=0) + 0.15


def build(out):
    pair = load()
    s = pair.s
    scene = Scene(pair)
    mesh = scene.mesh
    a = scene.terrain(0)
    b = scene.terrain(1)
    near_side = pair.plane(s["right_near_shore_west"], s["nominal_water_z_m"])
    far_side = pair.plane(s["right_far_shore_west"], s["nominal_water_z_m"])
    scene.pool(np.vstack([a, near_side]), np.vstack([b, far_side]))
    outboard = np.asarray(s["right_earth_outer_west"])
    outline = np.vstack([s["near_shore_west"][-1:], s["right_near_shore_west"], outboard[:, :2]])
    scene.surface(
        "Fracture lip and exposed bed",
        outline,
        np.r_[np.full(1 + len(near_side), s["nominal_water_z_m"]), outboard[:, 2]],
        0,
        rough=0.012,
    )
    lip = np.array(s["near_lip_west"][:1])
    edge = pair.project(pair.plane(lip, 0.0) - [0, 0, 0.10], 0)
    grass = np.asarray(s["near_verge_outer_west"])
    outer = np.asarray(s["near_road_outer_west"])[1:]
    outline = np.vstack([lip, edge, s["near_shore_west"][:1], grass[:, :2], outer])
    scene.surface(
        "Photographed verge",
        outline,
        np.r_[0, -0.10, s["nominal_water_z_m"], grass[:, 2], np.zeros(len(outer))],
        0,
    )
    blue, r, rad_samples = crossing(pair)
    # Sample real colours along the pipe axis and across its visible diameter.
    q = np.array(s["blue_tube"]["northwest_trace"], float)
    t = np.linspace(0, len(q) - 1, 768)
    uv = np.stack([np.interp(t, np.arange(len(q)), q[:, i]) for i in (0, 1)], axis=1)
    tangent = normalize(np.gradient(uv, axis=0))
    normal = tangent[:, [1, 0]] * [-1, 1]
    width = np.linspace(3.0, 17.0, len(t))
    rows = np.cos(np.linspace(-np.pi, np.pi, 64))
    pix = uv[None] + (rows[:, None] * width)[..., None] * normal[None] * 0.76
    color = cv2.remap(
        scene.images[1],
        pix[:, :, 0].astype("f4"),
        pix[:, :, 1].astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    tube = mesh.material("Blue weathered crossing from second exposure", color, 0.74, 0.05, 0.45)
    gray = mesh.material(
        "Grey open tube from the far bank", scene.images[1][278:283, 105:110], 0.9, 0.16, 0.25
    )
    dark = mesh.material("Unlit bore", np.full((8, 8, 3), [38, 34, 27], "u1"), 0.97, 0, 0)
    # Its near end disappears into the exposed earth in the west photograph.
    blue = np.vstack(
        [blue, pair.plane([[550, 397], [543, 423], [531, 455]], [-1.17, -1.20, -1.23])]
    )
    scene.tube("Exposed field conduit 01 crossing", blue, r, tube, dark)
    for i, pipe in enumerate(s["right_conduits_west"], 4):
        path = pair.plane(pipe["trace"], pipe["model_z_m"])
        start, end = np.asarray(pipe["texture_segment"])
        line = np.linspace(start, end, 180)
        normal = normalize((end - start)[[1, 0]] * [-1, 1])
        half = np.asarray(pipe["half_width_px"])
        pixels = (
            line[None]
            + normal[None, None]
            * np.linspace(-0.72, 0.72, 32)[:, None, None]
            * np.linspace(*half, 180)[None, :, None]
        )
        patch = cv2.remap(
            scene.images[0],
            pixels[:, :, 0].astype("f4"),
            pixels[:, :, 1].astype("f4"),
            cv2.INTER_LINEAR,
        )
        material = mesh.material(
            "Photographed " + pipe["name"] + " conduit", patch, 0.82, 0.04, 0.4
        )
        point = pair.plane([line[len(line) // 2]], np.mean(pipe["model_z_m"]), 0)[0]
        side = normalize(np.cross(path[-1] - path[0], point - pair.eye(0)))
        width = (
            np.linalg.norm(
                pair.project(point + 0.01 * side, 0) - pair.project(point - 0.01 * side, 0)
            )
            / 0.02
        )
        scene.tube(
            f"Exposed field conduit {i:02} " + pipe["name"],
            path,
            float(half.mean() / width),
            material,
            dark,
        )
    controls = pair.report()["landmarks"]
    for name, landmark, other, z in [
        ("Exposed field conduit 02 far stub", "open end on far bank", [108, 270], -0.88),
        ("Exposed field conduit 03 near stub", "near bank stub end", [135, 625], -1.65),
    ]:
        tip = np.asarray(controls[landmark]["world_m"])
        other = pair.plane([other], z, 1)[0]
        path = np.stack([tip, other])
        if landmark == "near bank stub end":
            path = np.vstack([tip, pair.plane([[135, 564]], -1.10, 1), other])
        scene.tube(name, path, r * 0.66, gray, dark)
    shift, span = scene.finalize()
    series = out / "series"
    cache = series / "mesh_cache"
    folder = series / "assets/pothole/P564"
    xy = grid(span, max(span) / 600)
    ext = np.ptp(xy, axis=(0, 1))
    meta = dict(
        id="P564",
        label=s["label"],
        seed=29564,
        category="pothole",
        state="two_photo_reconstruction",
        family="trent_washout",
        source=[x["photo"] for x in s["views"].values()],
        source_author=s["source_author"],
        source_license=s["license"],
        surface_dimensions_m=ext.tolist(),
        asset_world_origin_m=shift.tolist(),
        stereo=pair.report(),
        water_plane_m=s["nominal_water_z_m"],
        tube_radius_model_m=r,
        tube_radius_samples_m=rad_samples,
        blue_centerline_m=(blue - shift).tolist(),
        inferred="Metric scale, opaque bed, masked surfaces, bank heights between tie points and occluded tube continuation.",
    )
    mesh.save(folder, cache, meta, dict(width=ext[0], length=ext[1], height=np.zeros(xy.shape[:2])))
    flatten_scene(folder / "asset.glb")
    start = xyz(xy, 3.0)
    ray = np.broadcast_to([0.0, 0.0, -1.0], start.shape)
    surface = Surface(folder / "asset.glb")
    dist, part, _ = surface.cast(start, ray)
    below, _, _ = Surface(folder / "asset.glb", exclude={4, 59}).cast(start, ray)
    np.savez_compressed(
        folder / "surface.npz",
        width=ext[0],
        length=ext[1],
        height=np.where(np.isfinite(dist), 3.0 - dist, 0.0).astype("f4"),
        alpha=np.isfinite(dist),
        defect_mask=(part > 1),
        installation_mask=(part > 0),
        water_mask=(part == 4),
        opening_mask=np.zeros_like(part, bool),
        substrate_height=np.where(np.isfinite(below), 3.0 - below, 0.0).astype("f4"),
        photographic_rgb=True,
    )
    row = make_one(
        folder,
        cache,
        point_limit=780000,
        kernel_factors=(0.78, 0.16),
        part_importance={
            "Water surface": 2.6,
            "Subgrade bearing": 0.4,
            "Exposed field conduit 01 crossing": 8.0,
        },
    )
    from water import fit_water_radiance

    row["water_appearance"] = fit_water_radiance(folder, pair, shift, scene.images)
    write_json(folder / "metadata.json", row)
    for role, file in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("surface", "surface.npz"),
        ("metadata", "metadata.json"),
    ]:
        row[role] = "pothole/P564/" + file
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
