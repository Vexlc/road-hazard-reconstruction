"""Compare exported geometry with the field evidence, optical water and solid-only ray queries."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cameras import Cameras, S, photograph
from render import Renderer
from geometry import unit
from inspect_assets import inspect
from photo_mesh import Surface, write_json


def iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def road_visibility(eye, rays, part, t, size):
    d = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    plane = np.divide(-eye[2], d[:, :, 2], out=np.full(t.shape, np.inf), where=d[:, :, 2] != 0)
    cross = eye[:2] + plane[:, :, None] * d[:, :, :2]
    through = (abs(cross[:, :, 0]) <= size[0] / 2) & (abs(cross[:, :, 1]) <= size[1] / 2)
    return np.isfinite(t) & ((part == 1) | (plane <= 0) | (t < plane + 0.002) | through)


def compare(models, out):
    folder = models / "model/series/assets/missing/M576"
    cam = Cameras(S["working_focal_px"])
    out.mkdir(exist_ok=True, parents=True)
    delta = np.load(folder / "surface.npz")
    aperture = delta["opening_mask"]
    shaft = Surface(folder / "asset.glb")
    solids = Surface(folder / "asset.glb", exclude={4})
    xy = np.array([[-0.23, -0.11], [-0.11, 0.05], [-0.16, 0.18]])
    origins = np.c_[xy, np.ones(len(xy))]
    down = np.tile([0.0, 0.0, -1.0], (len(xy), 1))
    optical, first, _ = shaft.cast(origins, down)
    bare, _, _ = solids.cast(origins, down)
    assert np.all(first == 4) and np.all(np.isinf(bare))
    meta = json.loads((folder / "metadata.json").read_text())
    assert np.allclose(1 - optical, meta["water_plane_m"], atol=1.0e-5)
    info = dict(
        focal_sensitivity=[Cameras(f).report() for f in S["focal_sweep_px"]],
        underwater_solid_misses=len(xy),
        vertical_optical_depth_below_road_m=(optical - 1).tolist(),
        aperture_m2=float(
            aperture.sum()
            * (delta["width"] / (aperture.shape[1] - 1))
            * (delta["length"] / (aperture.shape[0] - 1))
        ),
    )
    fields = {}
    for key, wide in [("5710912", False), ("5710915", True)]:
        source = photograph(key)
        uv = np.indices(source.shape[:2])[::-1].transpose(1, 2, 0)
        rays = cam.rays(uv, wide).reshape(*source.shape[:2], 3)
        eye = cam.wide_eye if wide else cam.eye
        file = folder / ("asset.glb" if wide else "inspection.glb")
        rgb, part, t = Renderer(file).render(eye, rays)
        good = road_visibility(eye, rays, part, t, cam.size)
        rgb[~good] = 34
        part[~good] = 0
        np.savez_compressed(out / (key + "-rays.npz"), part=part, range_m=np.where(good, t, np.nan))
        Image.fromarray(rgb).save(out / (key + "-render.jpg"), quality=95)
        fields[key] = (source, rgb)
        if wide:
            y, x = np.indices(part.shape)
            roi = (x >= 180) & (x < 238) & (y >= 194) & (y < 254)
            red = (source[:, :, 0].astype(float) > 1.6 * source[:, :, 1]) & (
                source[:, :, 0].astype(float) > 1.4 * source[:, :, 2]
            )
            white = (
                roi
                & (x >= 196)
                & (x <= 208)
                & (((y >= 207) & (y <= 212)) | ((y >= 240) & (y <= 245)))
            )
            truth = roi & (red | white)
            info["approach_guard_iou"] = iou(truth, part == 174)
            info["approach_slab_visible_px"] = int(np.isin(part, [9, 11]).sum())
            assert info["approach_guard_iou"] > 0.7 and info["approach_slab_visible_px"] > 120
        else:
            samples = {
                "water": ([4], [[220, 298], [226, 335], [247, 388], [245, 438]]),
                "wall": ([6], [[390, 198], [417, 241], [459, 209]]),
                "yellow rail": ([20], [[280, 204], [313, 255], [343, 300]]),
                "coupling mouth": ([120, 121, 169], [[172, 216], [176, 219]]),
                "wall bracket": ([118], [[332, 383]]),
            }
            probes = {k: [int(part[y, x]) for x, y in v[1]] for k, v in samples.items()}
            print(probes, flush=True)
            assert all(all(p in samples[k][0] for p in v) for k, v in probes.items())
            info["near_pixel_probes"] = probes
    # Count submerged-surface visibility with the same closed-guard geometry
    # and camera, changing only the presence of the guard. Extend the host road
    # for visibility so side-on rays cannot see through the edge of the finite patch.
    bareguard = Surface(folder / "asset.glb", exclude={174})
    records = {}
    for y, h in [(-2.2, 1.65), (-3.6, 1.65), (-7.5, 1.45)]:
        eye = np.array([0.15, y, h])
        forward = unit(-eye)
        right = unit(np.cross(forward, [0, 0, 1.0]))
        up = np.cross(right, forward)
        v, u = np.indices((360, 460))
        rays = forward + (u[:, :, None] - 230) / 560 * right - (v[:, :, None] - 180) / 560 * up
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        ray_o = np.broadcast_to(eye, rays.shape)
        views = []
        for scene in [bareguard, shaft]:
            t, p, _ = scene.cast(ray_o, rays)
            views.append((p == 4) & road_visibility(eye, rays, p, t, cam.size))
        total = int(views[0].sum())
        remaining = int(views[1].sum())
        records[str(y)] = dict(
            camera_z_m=h,
            water_px_without_guard=total,
            water_px_with_guard=remaining,
            blocked_fraction=1 - remaining / total if total else None,
        )
    info["driver_water_views"] = records
    write_json(out / "field.json", info)
    pic = Image.new("RGB", (1040, 804), (31, 36, 37))
    draw = ImageDraw.Draw(pic)
    a, b = fields["5710912"]
    for x, im, label in [
        (0, a, "Inside | Peter Facey, 1 Feb 2018"),
        (323, b, "Inspection mesh | source view"),
    ]:
        pic.paste(Image.fromarray(im).resize((315, 420)), (x, 29))
        draw.text((x + 6, 9), label)
    a, b = fields["5710915"]
    pic.paste(Image.fromarray(a).resize((384, 288)), (648, 29))
    draw.text((650, 9), "Approach | same chamber, 62 seconds later")
    for x, im in [(648, a), (842, b)]:
        pic.paste(Image.fromarray(im[193:266, 157:242]).resize((190, 161)), (x, 320))
    draw.text((650, 482), "Photograph                  Guard and moved slab")
    return info, pic


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    info, fig = compare(args.models, args.out)
    inspect(args.models, args.out, lambda *unused: info, fit_tall=True, ids={"M576"})
    draw = ImageDraw.Draw(fig)
    for x, name in [(0, "near"), (520, "opposite")]:
        im = Image.open(args.out / ("M576-" + name + ".jpg"))
        im.thumbnail((510, 277))
        fig.paste(im, (x + (510 - im.width) // 2, 513))
        draw.text((x + 9, 494), "Native Gaussian view | " + name)
    fig.save(args.out / "whites-hill.jpg", quality=94)
