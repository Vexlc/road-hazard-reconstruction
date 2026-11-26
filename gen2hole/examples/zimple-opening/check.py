"""Compare both exported meshes to native source rays, held-out post edges and open probes."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from camera import Camera, HERE, S, photograph

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from inspect_assets import inspect
from mesh_transfer import RoadMesh
from photo_mesh import Surface

view = import_module("check-verge-chamber")


def check(models, out):
    cam = Camera()
    src = photograph()

    def evidence(folder, visual, meta, spec, run):
        observed = Surface(folder / "observed.glb")
        road = RoadMesh(folder)
        yy, xx = np.indices((955, 665))
        pixels = np.stack((xx + 2760, yy + 2070), axis=-1)
        ray = cam.ray(pixels)
        rgb, parts, _ = view.render(road, cam.eye, ray)
        field = src[20:975, 160:825]
        Image.fromarray(field).save(out / "field.jpg", quality=95)
        Image.fromarray(rgb).save(out / "same-pose.jpg", quality=95)
        # Compare against the photograph's green run on intermediate rows. These
        # samples omit buildings, reflective grey strips, tape and the tapered shoulder.
        hsv = cv2.cvtColor(field, cv2.COLOR_RGB2HSV)
        errors = []
        picked = set(np.array(S["marker_body_rows_y_left_right"])[:, 0])
        for y in range(2460, 2780, 7):
            if y in picked:
                continue
            j = y - 2070
            mask = (hsv[j, :, 0] > 28) & (hsv[j, :, 0] < 55) & (hsv[j, :, 1] > 70)
            roi = (pixels[j, :, 0] > 3018) & (pixels[j, :, 0] < 3140)
            image_x = np.flatnonzero(mask & roi)
            model_x = np.flatnonzero((parts[j] == 176) & roi)
            if len(image_x) < 30 or len(model_x) < 30:
                raise ValueError(("marker row", y))
            errors.extend(model_x[[0, -1]] - image_x[[0, -1]])
        probes = {
            "shaft_center": ([3020, 2979], 0),
            "far_lip": ([2898, 2948], 5),
            "post": ([3073, 2180], 176),
            "reflector": ([3080, 2375], 176),
            "ballast": ([2970, 2844], 176),
            "entering_line": ([3176, 2964], 177),
            "ground_line": ([3302, 2921], 177),
            "gravel": ([2890, 3014], 1),
        }
        answers = {}
        for name, (p, wanted) in probes.items():
            r = cam.ray([p])
            d, k, _ = observed.cast(cam.eye[None, :], r)
            answers[name] = dict(pixel=p, first_part=int(k[0]), expected=wanted)
        print("source probes", answers, flush=True)
        assert all(v["expected"] == v["first_part"] for v in answers.values())
        # Inspect through-handle bores from above. Rays may reach lower feet, but
        # they must pass below the 116-mm top plate before the first return.
        xy = np.array([cam.plane(a, 0.116)[:, :2].mean(0) for a in S["base_handles_px"]])
        o = np.column_stack((xy, np.full(2, 2.0)))
        dist, _, _ = observed.cast(o, np.broadcast_to([0.0, 0.0, -1.0], o.shape))
        handle_height = np.where(np.isfinite(dist), 2 - dist, 0)
        assert np.all(handle_height < 0.073)
        origin = np.array([[0.0, 0.0, 2.0]])
        down = np.array([[0.0, 0.0, -1.0]])
        unknown = observed.cast(origin, down)[0][0]
        closure = visual.cast(origin, down)[0][0]
        assert np.isinf(unknown) and abs(2 - closure + S["visual_only_closure_m"]) < 1e-5
        eye = np.array([1.10, -2.65, 1.24])
        r, _ = view.view_rays(eye, np.array([0.22, 0.16, 0.47]), 710, 850, 630)
        oblique, _, _ = view.render(road, eye, r)
        Image.fromarray(oblique).save(out / "approach.jpg", quality=94)
        with np.load(folder / "surface.npz") as f:
            cell = f["width"] * f["length"] / np.prod(np.array(f["height"].shape) - 1)
            area = float(f["opening_mask"].sum() * cell)
            assert area > 0.20 and area < 0.27
            assert np.array_equal(f["opening_mask"], f["alpha"] & ~f["observed_depth_valid"])
            assert np.all(
                np.asarray(meta["surface_dimensions_m"]) / 2 + 1e-6
                >= np.maximum(abs(observed.low[:2]), abs(observed.high[:2]))
            )
        sensitivity = []
        for dia in [0.55, 0.60, 0.65]:
            c = Camera(dia)
            xy = c.plane(S["marker_contact_center_px"])[:2]
            sensitivity.append(
                dict(
                    diameter_m=dia,
                    camera_height_m=float(c.eye[2]),
                    post_height_m=float(c.upright_height(xy, 2124)),
                )
            )
        rms = float(np.sqrt(np.mean(np.square(errors))))
        if rms > 4 or max(cam.hold_errors) > 3:
            raise ValueError((rms, max(cam.hold_errors)))
        return dict(
            source_probes=answers,
            rim_fit_max_px=float(cam.fit_errors.max()),
            rim_heldout_count=len(cam.hold_errors),
            rim_heldout_mean_px=float(cam.hold_errors.mean()),
            rim_heldout_max_px=float(cam.hold_errors.max()),
            post_heldout_edges=len(errors),
            post_outline_rms_px=rms,
            handle_first_height_m=handle_height.tolist(),
            scale_sensitivity=sensitivity,
            open_plan_m2=area,
            center_vertical_visual_z_m=float(2 - closure),
            center_vertical_observed="no return",
        )

    inspect(models, out, evidence, fit_tall=True, ids={"M582"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("models", "out"):
        p.add_argument(key, type=Path)
    check(**vars(p.parse_args()))
