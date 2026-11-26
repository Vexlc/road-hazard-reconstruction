"""Recast the source camera, check held-out silhouette rows and inspect two splat viewpoints."""

from importlib import import_module
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image
from camera import Camera, HERE, S
from build import Surface, RINGS
from inspect_assets import inspect
from mesh_transfer import RoadMesh

view = import_module("check-verge-chamber")


def check(models, out):
    def evidence(folder, surface, meta, spec, run):
        c = Camera()
        yy, xx = np.indices((615, 820))
        rays = c.rays(np.stack([xx, yy], -1))
        rgb, first, dist = view.render(RoadMesh(folder), c.eye, rays)
        Image.fromarray(rgb).save(out / "field-render.jpg", quality=95)
        bounded, _, _ = Surface(folder / "bounded.glb").cast(
            np.broadcast_to(c.eye, rays.shape), rays
        )
        cavity = np.zeros(first.shape, "u1")
        cv2.fillPoly(cavity, [np.int32(RINGS["outer_cut"])], 1)
        observed = {}
        for key, code in zip(list(RINGS)[1:], [182, 183, 184]):
            traced = np.zeros_like(cavity)
            cv2.fillPoly(traced, [np.int32(RINGS[key])], 1)
            match = first == code
            iou = (traced.astype(bool) & match).sum() / (traced.astype(bool) | match).sum()
            observed[key] = float(iou)
            assert iou > 0.90, (key, iou)
        errors = []
        for k, rows in enumerate(S["check_silhouettes_px"]):
            for left, right, y in rows:
                xs = np.flatnonzero((first[y] == 25) & ((xx[y] < 300) if k == 0 else (xx[y] > 300)))
                errors.extend([int(xs.min() - left), int(xs.max() - right)])
        source = np.asarray(
            Image.open(HERE / "references/site.jpg").resize((820, 615), Image.Resampling.LANCZOS)
        )
        orig = source.copy()
        for key, code, color in zip(
            list(RINGS)[1:], [182, 183, 184], [(26, 211, 228), (241, 168, 244), (252, 177, 35)]
        ):
            curves, _ = cv2.findContours(
                (first == code).astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(orig, curves, -1, color, 1, cv2.LINE_AA)
        cv2.polylines(orig, [np.int32(RINGS["outer_cut"])], True, (238, 218, 60), 1, cv2.LINE_AA)
        Image.fromarray(orig).save(out / "field-overlay.jpg", quality=94)
        np.savez_compressed(
            out / "field-rays.npz", first_part=first, optical_range_m=dist, bounded_range_m=bounded
        )
        np.testing.assert_array_equal(
            first[[398, 479, 537, 549, 275], [378, 411, 568, 491, 498]], [180, 182, 183, 184, 25]
        )
        cavity_iou = (np.isin(first, [180, 181, 182, 183, 184]) & cavity.astype(bool)).sum() / (
            np.isin(first, [180, 181, 182, 183, 184]) | cavity.astype(bool)
        ).sum()
        assert cavity_iou > 0.98
        # Lens variation changes pitch as well as distance. Refit pose at each setting.
        sensitivity = []
        for height, f in [(h, f) for h in [0.65, 0.71, 0.76] for f in [0.95, 1.0, 1.05]]:
            test = Camera(height, c.f * f)
            rim = test.plane(RINGS["outer_cut"])
            box = cv2.minAreaRect(rim[:, :2].astype("f4"))[1]
            sensitivity.append(
                dict(cone_height_m=height, focal_factor=f, aperture_box_m=sorted(box, reverse=True))
            )
        return dict(
            plane_pose_pitch_roll_height=c.pose.tolist(),
            aperture_iou=float(cavity_iou),
            component_iou=observed,
            held_out_cone_edge_errors_px=errors,
            source_rays_without_bounded_hit_in_aperture=int(
                np.sum((cavity == 1) & ~np.isfinite(bounded))
            ),
            silhouette_depth_hypotheses=True,
            sensitivity=sensitivity,
        )

    inspect(models, out, evidence, fit_tall=True, ids={"P584"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("models", "out"):
        p.add_argument(k, type=Path)
    check(**vars(p.parse_args()))
