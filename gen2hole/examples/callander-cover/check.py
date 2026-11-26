"""Compare source-camera faces and the exposed front edge in two height hypotheses."""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from camera import Camera, HERE, load

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from inspect_assets import inspect

sys.path.insert(0, str(HERE.parent / "whites-hill"))
from render import Renderer


def check(models, out):
    s, cam = load()

    def field(folder, surf, meta, spec, run):
        yy, xx = np.indices((s["size_px"][1], s["size_px"][0]))
        uv = np.stack((xx, yy), axis=-1)
        rays = cam.rays(uv)
        rgb, parts, t = Renderer(folder / "asset.glb").render(cam.eye, rays)
        _, flat, tf = Renderer(folder / "flush.glb").render(cam.eye, rays)
        expected = np.zeros(parts.shape, "u1")
        cv2.fillPoly(
            expected, [np.int32(np.rint(np.asarray(s["face_corners_px"]) * 16))], 1, shift=4
        )
        mask = expected > 0
        iou = float(((parts == 9) & mask).sum() / ((parts == 9) | mask).sum())
        probes = [
            dict(uv=[u, v], expected=p, hit=int(parts[v, u])) for u, v, p in s["survey_part_probes"]
        ]
        if any(q["hit"] != q["expected"] for q in probes):
            raise ValueError(str(probes))
        if iou < 0.96:
            raise ValueError("Lid footprint differs from photo")
        source = np.asarray(Image.open(HERE / "references/1673294.jpg").convert("RGB"))
        Image.fromarray(rgb).save(out / "source-render.png")
        np.savez_compressed(
            out / "source-rays.npz",
            range_m=t,
            first_part=parts,
            flush_range_m=tf,
            flush_first_part=flat,
        )
        trace = []
        for (u, v), toe in s["heldout_front_edge_top_and_toe_px"]:
            ys = np.arange(v - 3, v + 14, 0.1)
            ray = cam.rays(np.stack([np.full_like(ys, u), ys], axis=-1))
            dist, part, norm = surf.cast(np.broadcast_to(cam.eye, ray.shape), ray)
            face = (part == 5) & (abs(norm[:, 2]) < 0.15)
            if not face.any():
                raise ValueError("Missing front face")
            trace.append(
                dict(
                    x_px=u,
                    toe_v_observed_px=toe[1],
                    toe_v_rendered_px=float(ys[face].max()),
                    visible_vertical_band_px=float(face.sum() * 0.1),
                )
            )
        same = (parts == 9) & (flat == 9)
        both = (parts > 1) & (flat > 1)
        depth = (t[both] - tf[both]) * 1000
        with np.load(folder / "surface.npz") as f:
            h = f["height"]
            ids = f["first_part"]
            levels = {
                str(i): [float(h[ids == i].min()), float(h[ids == i].max())]
                for i in np.unique(ids)
                if i
            }
            if abs(h[ids == 9]).max() > 2e-5 or h[ids == 2].min() >= -0.015:
                raise ValueError(levels)
        sweep = []
        for focal in s["focal_sweep_px"]:
            trial = Camera(s, focal)
            sweep.append(
                dict(
                    focal_px=focal,
                    corner_rms_px=trial.error,
                    length_m=trial.length,
                    toe_drop_m=[trial.drop(q) for q in s["front_edge_top_and_toe_px"]],
                )
            )
        err = max(abs(q["toe_v_observed_px"] - q["toe_v_rendered_px"]) for q in trace)
        if err > 2:
            raise ValueError(trace)
        return dict(
            corner_rms_px=cam.error,
            nominal_face_width_m=s["nominal_face_width_m"],
            fitted_length_m=cam.length,
            source_lid_mask_iou=iou,
            sampled_probes=probes,
            heldout_toes=trace,
            model_height_extrema_m=levels,
            focal_sweep=sweep,
            lid_depth_delta_max_mm=float(abs(t[same] - tf[same]).max() * 1000),
            paired_abs_range_change_p99_mm=float(np.percentile(abs(depth), 99)),
            source_appearance_mae_srgb=float(np.abs(rgb.astype(float) - source)[parts > 1].mean()),
        )

    out.mkdir(parents=True, exist_ok=True)
    inspect(models, out, field, ids={"A580"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("out", type=Path)
    check(**vars(p.parse_args()))
