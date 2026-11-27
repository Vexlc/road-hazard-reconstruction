"""Build the observed pair and three portable states of the raised foreground utility cover."""

import argparse
from pathlib import Path
import sys
import cv2
import mapbox_earcut
import open3d as o3d
import numpy as np
from PIL import Image
from camera import HERE, S, calibrate, photo
from casting import project, rectified, visible_face

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import Mesh, Surface, grid, make_one, flatten_scene, texture_uv, write_json, xyz

KEY = "A591"
SPAN = np.array([1.62, 1.2])
R = S["nominal_near_diameter_m"] / 2
H = S["model_raise_m"]
THICK = S["trial_cover_thickness_m"]
OPEN = S["trial_shaft_radius_m"]
SEAT = R + 0.0015
FRAME = R + 0.020
N = 192
ANGLE = np.arange(N) * 2 * np.pi / N
UNIT = np.c_[np.cos(ANGLE), np.sin(ANGLE)]


def add_band(mesh, name, lower, upper, mat, inward=False):
    i = np.arange(len(lower))
    j = np.roll(i, -1)
    n = len(i)
    f = np.vstack([np.c_[i, j, j + n], np.c_[i, j + n, i + n]])
    if inward:
        f = f[:, ::-1]
    mesh.add(
        name, np.vstack([lower, upper]), f, np.vstack([0.5 + UNIT * 0.40, 0.5 + UNIT * 0.48]), mat
    )


def sheet(mesh, name, rings, z, mat, road=False, reverse=False):
    coords = np.vstack(rings)
    faces = mapbox_earcut.triangulate_float64(
        coords, np.cumsum(list(map(len, rings)), dtype="u4")
    ).reshape(-1, 3)
    mesh.add(
        name, xyz(coords, z), faces[:, ::-1] if reverse else faces, 0.5 + coords / SPAN, mat, road
    )


def materials(mesh, atlas):
    img = photo()
    mesh.material("Pepco tread", atlas, 0.77, 0.34, 0.42)
    rim = atlas[260:355, 12:22].copy()
    mid = np.median(rim, axis=(0, 1))
    rim = np.uint8(np.clip(mid + 0.40 * (rim - mid), 0, 255))
    mesh.material("Edge iron analogue", rim, 0.82, 0.26, 0.25)
    mesh.material("Receiver asphalt", img[1770:1910, 240:600], 0.98, 0, 0.25)
    mesh.material(
        "Chamber lining assumption", np.full((8, 8, 3), [43, 41, 39], "u1"), 0.97, 0, 0.12
    )
    mesh.material("Dark optical termination", np.full((8, 8, 3), [22, 22, 21], "u1"), 1, 0, 0)


def receiver_shade(surface, field):
    # Cosine-weighted sky visibility on near-grade road surfaces. Cast only near
    # the assembly; the renderer adds the host's measured directional sunlight.
    xy = field[::2, ::2]
    keep = np.linalg.norm(xy, axis=-1) < 0.77
    eye = xyz(xy[keep], 0.0015)
    total = np.zeros(len(eye))
    for k in range(48):
        t = (k + 0.5) / 48
        phi = k * np.pi * (3 - np.sqrt(5.0))
        ray = np.array([np.sqrt(t) * np.cos(phi), np.sqrt(t) * np.sin(phi), np.sqrt(1 - t)])
        batch = np.column_stack([eye, np.broadcast_to(ray, eye.shape)]).astype("f4")
        total += surface.scene.test_occlusions(o3d.core.Tensor(batch), nthreads=2).numpy()
    sky = np.zeros(keep.shape, "f4")
    sky[keep] = total / 48
    return cv2.resize(sky, (field.shape[1], field.shape[0]), interpolation=cv2.INTER_LINEAR)


def cover(mesh, shift, z):
    ring = R * UNIT
    centre = np.vstack([[0, 0], ring])
    i = np.arange(N) + 1
    mesh.add(
        "Lid face",
        xyz(centre + shift, z),
        np.c_[np.zeros(N, int), i, np.roll(i, -1)],
        0.5 + centre / (2 * R),
        "Pepco tread",
    )
    add_band(
        mesh,
        "Lid rolled edge",
        xyz(ring + shift, z - THICK),
        xyz(ring + shift, z),
        "Edge iron analogue",
    )
    sheet(mesh, "Lid underside", [ring + shift], z - THICK, "Edge iron analogue", reverse=True)


def surround(mesh):
    rect = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * SPAN / 2
    sheet(mesh, "Road around the aperture", [rect, UNIT * FRAME], 0, "Receiver asphalt", road=True)
    sheet(mesh, "Cast ring and cover seat", [UNIT * FRAME, UNIT * SEAT], H, "Edge iron analogue")
    add_band(
        mesh,
        "Pepco outer frame wall",
        xyz(UNIT * FRAME, 0),
        xyz(UNIT * FRAME, H),
        "Edge iron analogue",
    )
    sheet(
        mesh,
        "Cast seat riser and ledge",
        [UNIT * SEAT, UNIT * OPEN],
        H - THICK,
        "Edge iron analogue",
    )
    add_band(
        mesh,
        "Inspection shaft",
        xyz(UNIT * OPEN, -S["trial_closure_depth_m"]),
        xyz(UNIT * OPEN, H - THICK),
        "Chamber lining assumption",
        True,
    )
    add_band(
        mesh,
        "Pepco seat riser",
        xyz(UNIT * SEAT, H - THICK),
        xyz(UNIT * SEAT, H),
        "Edge iron analogue",
        True,
    )
    sheet(
        mesh, "Shaft bottom", [UNIT * OPEN], -S["trial_closure_depth_m"], "Dark optical termination"
    )


def observed(cal, out):
    mesh = Mesh()
    img = photo()
    mat = mesh.material("Payton Chung, CC BY 2.0", img, 0.85, 0.19, 0.25)
    stats = []
    for i, (pos, r) in enumerate(
        [(cal["near_xz_m"], R), (cal["far_xz_m"], cal["far_diameter_m"] / 2)]
    ):
        axis = np.arange(-r, r + 0.005, 0.005)
        x, y = np.meshgrid(axis, axis)
        xy = np.stack([x, y], -1).reshape(-1, 2)
        world = xy + pos
        uv = project(np.array(cal["homography"]), world)
        valid = (
            (np.linalg.norm(xy, axis=1) <= r)
            & (uv >= 1).all(1)
            & (uv < np.array(img.shape[1::-1]) - 2).all(1)
        )
        a = np.arange(x.size).reshape(x.shape)[:-1, :-1].ravel()
        b = a + 1
        c = a + len(axis)
        e = c + 1
        f = np.vstack([np.c_[a, b, c], np.c_[b, e, c]])
        f = f[valid[f].all(1)]
        mesh.add(
            ["Pepco near photographed face", "Pepco far photographed face"][i],
            xyz(world - cal["near_xz_m"], H),
            f,
            texture_uv(uv, img.shape[1::-1]),
            mat,
        )
        stats.append(float(len(f) * 0.005**2 / 2 / (np.pi * r * r)))
    output = out / "observed-pair.glb"
    output.write_bytes(mesh.glb_bytes())
    flatten_scene(output)
    return stats


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    cal, obs, rings = calibrate()
    atlas, mask, stats = rectified(cal)
    write_json(
        out / "registration.json",
        dict(
            calibration=cal,
            rings=rings,
            appearance=stats,
            observed_mesh_area_fraction=observed(cal, out),
        ),
    )
    np.savez_compressed(out / "observations.npz", **obs, observed_top=mask)
    image = Image.fromarray(atlas)
    image.save(out / "casting.png")
    loose = Mesh()
    materials(loose, atlas)
    cover(loose, np.zeros(2), 0)
    path = out / "loose-cover.glb"
    path.write_bytes(loose.glb_bytes())
    flatten_scene(path)
    for state in S["states"]:
        series = out / state / "model/series"
        dest = series / "assets/raised" / KEY
        mesh = Mesh()
        materials(mesh, atlas)
        surround(mesh)
        shift = np.array(S["offset_xy_m"]) if state == "offset" else np.zeros(2)
        if state != "open":
            cover(mesh, shift, H + (THICK if state == "offset" else 0))
        field = grid(SPAN, 0.003)
        meta = dict(
            id=KEY,
            category="raised",
            family="pepco_trench",
            label="华盛顿公用管线凸起井盖",
            state=state,
            seed=29591,
            source=S["source"],
            source_author=S["author"],
            source_license=S["license"],
            surface_dimensions_m=SPAN.tolist(),
            nominal_diameter_m=2 * R,
            frame_raise_m=H,
            trial_thickness_m=THICK,
            trial_optical_closure_m=S["trial_closure_depth_m"],
            cover_translation_xy_m=shift.tolist(),
            photo_mask="surface.npz:source_face",
        )
        cache = series / "mesh_cache"
        mesh.save(dest, cache, meta, dict(height=np.zeros(field.shape[:2])))
        flatten_scene(dest / "asset.glb")
        surf = Surface(dest / "asset.glb")
        eye = xyz(field, 1)
        t, parts, _ = surf.cast(eye, np.broadcast_to([0.0, 0.0, -1.0], eye.shape))
        cut = np.linalg.norm(field, axis=-1) < FRAME
        np.savez_compressed(
            dest / "surface.npz",
            width=SPAN[0],
            length=SPAN[1],
            height=np.where(np.isfinite(t), 1 - t, 0).astype("f4"),
            first_part=parts,
            alpha=(parts > 1).astype("f4"),
            defect_mask=parts > 1,
            installation_mask=cut,
            opening_mask=cut,
            registered_relief=True,
            photographic_rgb=True,
            receiver_sky_blocked=receiver_shade(surf, field),
            cast_receiver_sunlight=True,
            source_face=(parts == 9) & visible_face(cal, field - shift),
            water_mask=np.zeros_like(cut),
        )
        row = make_one(dest, cache, point_limit=270000, kernel_factors=(0.87, 0.16))
        for k, f in [
            ("glb", "asset.glb"),
            ("ply", "asset.ply"),
            ("metadata", "metadata.json"),
            ("surface", "surface.npz"),
        ]:
            row[k] = "raised/" + KEY + "/" + f
        write_json(series / "assets/asset_manifest.json", [row])
        write_json(series.parent / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
