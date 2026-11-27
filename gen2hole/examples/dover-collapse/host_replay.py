"""Drape a receiving exposure onto the displaced deck using its calibrated road plane."""

import argparse
from importlib import import_module
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from cameras import HERE

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import LIB

sys.path.insert(0, str(LIB.parent / "scripts"))
driver = import_module("render-road-distress")
from camera_geometry import CAMERAS
from traffic_safe_drive import read, write


def replay(raw, masks, models, out, side):
    camera = side + "_camera"
    original = driver.drive.renderer.frame_crop
    records = {}

    def frame(mesh, anchor, model, pose, path, points, window, ego):
        result = original(mesh, anchor, model, pose, path, points, window, ego)
        if not result or result["clipped_sides"] or result["touches_mask"]:
            return result
        part = result["part"]
        before = result["before"]
        after = result["after"]
        lo = result["xy"]
        h, w = part.shape
        # The adjoining grade in the replay carries the host exposure.
        after[part > 0] = before[part > 0]
        yy, xx = np.where(part >= 2)
        if not len(xx):
            return result
        rays = model[2](*lo, w, h)[0][yy, xx]
        basis = np.array(anchor["basis_columns"])
        centre = np.array(anchor["center_world"])
        eye = (pose[:3, 3] - centre) @ basis
        dirs = rays @ (pose[:3, :3].T @ basis)
        shot = np.column_stack([np.broadcast_to(eye, dirs.shape), dirs]).astype("f4")
        hits = mesh.scene.cast_rays(o3d.core.Tensor(shot), nthreads=2)
        gid = hits["geometry_ids"].numpy()
        t = hits["t_hit"].numpy()
        defect = np.isfinite(t) & np.isin(
            gid, [g for g, s in mesh.surfaces.items() if s[0] in (2, 3)]
        )
        yy, xx, t, dirs = yy[defect], xx[defect], t[defect], dirs[defect]
        normal = hits["primitive_normals"].numpy()[defect]
        normal = np.where(normal[:, 2:] < 0, -normal, normal)
        local = eye + t[:, None] * dirs
        local[:, 2] = 0
        uv = model[1]((local @ basis.T + centre - pose[:3, 3]) @ pose[:3, :3])
        photo = np.asarray(Image.open(path).convert("RGB"))
        frame_id = int(path.stem)
        traffic = np.asarray(
            Image.open(masks / f"{frame_id:03}_{CAMERAS.index(camera)}.png").convert("L")
        )
        # Protect host material donors independently of destination visibility.
        # The original flat-road projection can lie beyond the visible depressed deck.
        above = (points - centre) @ basis[:, 2] > 0.10
        optical = (points[above] - pose[:3, 3]) @ pose[:3, :3]
        optical = optical[(optical[:, 2] > 0.05) & (np.linalg.norm(optical, axis=1) < 70)]
        pixels = np.rint(model[1](optical)).astype(int)
        shape = np.array(photo.shape[1::-1])
        valid = (pixels >= 0).all(1) & (pixels < shape).all(1)
        depth = np.full(photo.shape[:2], np.inf, "f4")
        np.minimum.at(
            depth,
            (pixels[valid, 1], pixels[valid, 0]),
            np.linalg.norm(optical[valid], axis=1).astype("f4"),
        )
        depth = cv2.erode(depth, np.ones((7, 7), "u1"))
        donor_range = np.linalg.norm(local - eye, axis=1)
        ij = np.floor(uv).astype(int)
        inside = (ij >= 0).all(1) & (ij + 1 < shape).all(1)
        ego_edge = np.array(ego["edge"]) * shape / np.array(ego["grid"])
        for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            x = (ij[:, 0] + dx).clip(0, shape[0] - 1)
            y = (ij[:, 1] + dy).clip(0, shape[1] - 1)
            inside &= (
                (traffic[y, x] == 0)
                & (depth[y, x] >= donor_range - 0.04)
                & (y < np.interp(x, ego_edge[:, 0], ego_edge[:, 1]))
            )
        # Fall back to the same destination pixel if the donor pixel fails a visibility check.
        donor = before[yy, xx].astype(float)
        if inside.any():
            base = ij[inside]
            f = uv[inside] - base
            donor[inside] = sum(
                photo[base[:, 1] + dy, base[:, 0] + dx]
                * ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]))[:, None]
                for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]
            )
        lighting = anchor["relief_daylight"]
        sun = np.array(lighting["sun_world"]) @ basis
        direct = lighting["direct_fraction"]
        dot = np.clip(normal @ sun, 0, 1)
        # Trace the local sun ray, retaining ambient skylight in geometric shadow.
        surface = eye + t[:, None] * dirs
        shadowrays = np.column_stack(
            [surface + 0.003 * sun, np.broadcast_to(sun, surface.shape)]
        ).astype("f4")
        blocked = mesh.scene.test_occlusions(o3d.core.Tensor(shadowrays), nthreads=2).numpy()
        ratio = (1 - direct) + direct * dot * (~blocked) / sun[2]
        face = part[yy, xx] == 3
        ratio[face] = np.minimum(ratio[face], 0.17)
        # Use the host diffuse colour and modulate irradiance in linear light.
        v = donor / 255
        linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4) * ratio[:, None]
        srgb = np.where(
            linear <= 0.0031308, 12.92 * linear, 1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055
        )
        after[yy, xx] = np.rint(np.clip(srgb, 0, 1) * 255).astype("u1")
        shift = np.linalg.norm(uv[inside] - np.c_[xx[inside], yy[inside]] - lo, axis=1)
        records[str(frame_id)] = dict(
            registered=int(inside.sum()),
            same_pixel_fallback=int((~inside).sum()),
            median_shift_px=float(np.median(shift)) if len(shift) else None,
        )
        result["changed"] = int(np.any(before != after, axis=2).sum())
        return result

    with patch.object(driver.drive.renderer, "frame_crop", frame):
        driver.render(
            raw,
            masks,
            models,
            out,
            22,
            [920, 730],
            ["P590"],
            camera,
            semantic_occlusion=True,
            relief_daylight=read(HERE.parent / "gervis-rupture/daylight.json"),
            focus_part=2,
        )
    path = out / "P590"
    frames = read(path / "placement.json")["exposures"]
    write(path / "registration.json", {str(r["frame"]): records[str(r["frame"])] for r in frames})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "masks", "models", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("side", choices=("front", "back"))
    replay(**vars(p.parse_args()))
