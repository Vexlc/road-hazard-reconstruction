"""Build three interchangeable road installations and the isolated photographed casting."""

import argparse
import hashlib
from pathlib import Path
import sys
import cv2
import mapbox_earcut
import numpy as np
from casting import Casting, HERE, S, rgb

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import Mesh, Surface, flatten_scene, grid, make_one, poly_mask, write_json, xyz

SPAN = np.array([1.64, 0.96])
KEY = "M589"


def triangles(rings):
    ends = np.cumsum([len(r) for r in rings], dtype="u4")
    return mapbox_earcut.triangulate_float64(np.vstack(rings), ends).reshape(-1, 3)


def box(half):
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half


def face(mesh, name, rings, z, mat):
    p = np.vstack(rings)
    f = triangles(rings)
    mesh.add(name, xyz(p, z), f, 0.5 + p / SPAN, mat, name.startswith("Road "))


def band(mesh, name, a, b, mat, invert=False):
    n = len(a)
    i = np.arange(n)
    j = np.roll(i, -1)
    v = np.vstack([a, b])
    f = np.vstack([np.c_[i, j, i + n], np.c_[j, j + n, i + n]])
    if invert:
        f = f[:, ::-1]
    uv = np.vstack([np.c_[np.arange(n) % 2, np.ones(n)], np.c_[np.arange(n) % 2, np.zeros(n)]])
    mesh.add(name, v, f, uv, mat)


def material(mesh, name, color, rough, metal):
    return mesh.material(name, np.full((32, 32, 3), color, dtype="u1"), rough, metal, 0.1)


def iron_material(mesh, name, tint):
    path = HERE / "references/oxidation.png"
    if hashlib.sha256(path.read_bytes()).hexdigest() != S["rust_patch_sha256"]:
        raise ValueError("Rust sample digest mismatch")
    sample = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(float)
    # Use photographed grain as an analogue for otherwise unseen iron. Mirroring
    # across tile boundaries keeps its intensity continuous on large panels.
    contrast = np.clip((sample + 0.5) / np.percentile(sample, 50, axis=(0, 1)), 0.42, 1.9)
    tile = np.uint8(np.clip(tint * (0.45 + 0.55 * contrast), 0, 255))
    tile = np.vstack([tile, tile[::-1]])
    tile = np.hstack([tile, tile[:, ::-1]])
    return mesh.material(name, np.tile(tile, (2, 2, 1)), 0.92, 0.15, 0.35)


def grate(mesh, cam, pose, lift):
    rings = cam.rings()
    xy = np.vstack(rings)
    lo = xy.min(0) - 0.003
    hi = xy.max(0) + 0.003
    cols, rows = np.ceil((hi - lo) / 0.00045).astype(int)
    x = lo[0] + (np.arange(cols) + 0.5) * (hi[0] - lo[0]) / cols
    y = hi[1] - (np.arange(rows) + 0.5) * (hi[1] - lo[1]) / rows
    xx, yy = np.meshgrid(x, y)
    px = (
        cam.project(np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        .reshape(rows, cols, 2)
        .astype("f4")
    )
    tex = cv2.remap(rgb(), px, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    iron = mesh.material("Brastad painted casting", tex, 0.9, 0.21, 0.45, lossless_color=True)
    hidden = iron_material(mesh, "Dark oxidised interior iron", np.array([57, 47, 42]))
    underside = iron_material(mesh, "Unseen cast underside", np.array([71, 57, 48]))
    theta = np.deg2rad(pose[2])
    c, s = np.cos(theta), np.sin(theta)

    def move(p, z):
        return xyz(p @ np.array([[c, s], [-s, c]]) + pose[:2], z + lift)

    faces = triangles(rings)
    mesh.add("Lid face", move(xy, 0), faces, (xy - lo) / (hi - lo), iron)
    mesh.add(
        "Lid underside", move(xy, -cam.thickness), faces[:, ::-1], (xy - lo) / (hi - lo), underside
    )
    # Use disjoint index ranges for every loop so all eight holes are real bores.
    v = []
    f = []
    uv = []
    cursor = 0
    for q in rings:
        q = np.vstack([q, q[0]])
        n = len(q)
        i = np.arange(n - 1)
        j = i + 1
        v.extend([move(q, 0), move(q, -cam.thickness)])
        f.extend([np.c_[i, i + n, j] + cursor, np.c_[j, i + n, j + n] + cursor])
        along = np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1).cumsum()]
        uv.extend(
            [np.c_[along / along[-1], np.full(n, 0.53)], np.c_[along / along[-1], np.full(n, 0.27)]]
        )
        cursor += 2 * n
    mesh.add("Lid aperture walls and perimeter", np.vstack(v), np.vstack(f), np.vstack(uv), hidden)


def socket(mesh, cam):
    # These dimensions make a compatible virtual seat. The original photograph
    # shows discarded ironwork and does not constrain the chamber underground.
    top = cam.rings()[0] * 1.019
    clear = box([0.171, 0.169])
    frame = box(np.max(abs(top), axis=0) + 0.022)
    iron = iron_material(mesh, "Scenario cast frame", np.array([84, 73, 65]))
    road = material(mesh, "Host pavement receiver", [95, 96, 94], 0.97, 0)
    wall = iron_material(mesh, "Scenario oxidised inlet lining", np.array([51, 47, 42]))
    floor = material(mesh, "Optical closure at assumed depth", [25, 25, 25], 1.0, 0)
    face(mesh, "Road around the aperture", [box(SPAN / 2), frame], 0, road)
    face(mesh, "Cast ring and cover seat", [frame, top], 0, iron)
    # JOIN ledge top and riser into one semantic part (no duplicate GLB node names).
    seat = Mesh()
    face(seat, "ledge", [top, clear], -cam.thickness, iron)
    band(seat, "riser", xyz(top, 0), xyz(top, -cam.thickness), iron, invert=False)
    vertices = []
    indices = []
    uv = []
    i = 0
    for _, v, f, t, *_ in seat.parts:
        vertices.append(v)
        indices.append(i + f)
        uv.append(t)
        i += len(v)
    mesh.add(
        "Cast seat riser and ledge", np.vstack(vertices), np.vstack(indices), np.vstack(uv), iron
    )
    depth = S["trial_socket_depth_m"]
    band(mesh, "Inspection shaft", xyz(clear, -cam.thickness), xyz(clear, -depth), wall)
    face(mesh, "Shaft bottom", [clear], -depth, floor)


def build(out):
    cam = Casting()
    for state, pose in S["relative_grate_poses"].items():
        model = out / state / "model"
        series = model / "series"
        folder = series / "assets/missing" / KEY
        mesh = Mesh()
        socket(mesh, cam)
        # Seated face is flush. Displaced and partly overlapping castings rest on grade.
        grate(mesh, cam, np.array(pose), 0 if state == "seated" else cam.thickness)
        meta = dict(
            id=KEY,
            label="Brastad 白漆八槽排水篦",
            family="brastad_grate",
            category="missing",
            state=state,
            seed=29589,
            source=S["source"],
            source_author=S["author"],
            source_license=S["licence"],
            source_sha256=S["crop_sha256"],
            surface_dimensions_m=SPAN.tolist(),
            grate_pose_xy_yaw_deg=pose,
            plate_thickness_m=float(cam.thickness),
            nominal_slot_pitch_m=S["nominal_slot_pitch_m"],
            socket_depth_scenario_m=S["trial_socket_depth_m"],
            unmeasured_geometry_parts=[5, 6, 7, 10],
        )
        xy = grid(SPAN, 0.002)
        mesh.save(folder, series / "mesh_cache", meta, dict(height=np.zeros(xy.shape[:2])))
        flatten_scene(folder / "asset.glb")
        surf = Surface(folder / "asset.glb")
        orig = xyz(xy, 1.0)
        t, part, _ = surf.cast(orig, np.broadcast_to([0.0, 0.0, -1.0], orig.shape))
        valid = np.isfinite(t)
        cut = poly_mask(xy, cam.outer * 1.019)
        np.savez_compressed(
            folder / "surface.npz",
            width=SPAN[0],
            length=SPAN[1],
            height=np.where(valid, 1 - t, 0),
            alpha=(part > 1).astype("f4"),
            registered_relief=True,
            cast_receiver_sunlight=True,
            first_part=part,
            defect_mask=part > 1,
            installation_mask=cut,
            opening_mask=cut,
            unresolved_chamber=np.isin(part, [6, 7]),
            water_mask=np.zeros_like(valid),
            photographic_rgb=True,
        )
        row = make_one(
            folder, series / "mesh_cache", point_limit=420000, kernel_factors=(0.87, 0.14)
        )
        for k, f in [
            ("glb", "asset.glb"),
            ("ply", "asset.ply"),
            ("metadata", "metadata.json"),
            ("surface", "surface.npz"),
        ]:
            row[k] = "missing/" + KEY + "/" + f
        write_json(series / "assets/asset_manifest.json", [row])
        write_json(model / "run.json", dict(editions=["series"]))
    movable = Mesh()
    grate(movable, cam, np.array([0, 0, 0]), 0)
    (out / "grate.glb").write_bytes(movable.glb_bytes())
    flatten_scene(out / "grate.glb")
    empty = Mesh()
    socket(empty, cam)
    (out / "empty.glb").write_bytes(empty.glb_bytes())
    flatten_scene(out / "empty.glb")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
