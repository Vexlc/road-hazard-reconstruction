"""Compare the exported surfaces against the source exposure, then measure cone occlusion by viewpoint."""

import argparse
import json
import sys
from pathlib import Path
from importlib import import_module
import cv2
import numpy as np
from PIL import Image
from camera import HERE, S, SIZE, SOURCE, RADIUS, fit, rays

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from inspect_assets import inspect
from mesh_transfer import RoadMesh
from photo_mesh import Surface, xyz

view = import_module("check-verge-chamber")


def check(models, out):
    k, cones, registration = fit()
    eye = np.r_[-k[3:5], k[2]]
    out.mkdir(parents=True, exist_ok=True)

    def evidence(folder, surface, meta, spec, run):
        source = cv2.resize(
            np.asarray(Image.open(SOURCE).convert("RGB")), tuple(SIZE), interpolation=cv2.INTER_AREA
        )
        yy, xx = np.indices(tuple(SIZE[::-1]))
        r = rays(np.stack([xx, yy], -1), k)
        rgb, parts, _ = view.render(RoadMesh(folder), eye, r)
        Image.fromarray(rgb[370:705, 242:920]).save(out / "original-pose.jpg", quality=96)
        original_aperture = Surface(folder / "observed.glb")
        _, ob, _ = original_aperture.cast(np.broadcast_to(eye, r.shape), r)
        h, s, v = cv2.split(cv2.cvtColor(source, cv2.COLOR_RGB2HSV))
        red = ((h < 18) | (h > 169)) & (s > 110) & (v > 85)
        red &= ~((xx < 350) & (yy < 614 - 0.34 * (xx - 317)))
        regions = [(313, 426, 469, 653), (590, 383, 704, 573), (728, 446, 903, 694)]
        iou = []
        for x1, y1, x2, y2 in regions:
            mask = red[y1:y2, x1:x2]
            model = parts[y1:y2, x1:x2] == 25
            iou.append(float((mask & model).sum() / (mask | model).sum()))
        probes = {
            "road": (1, [[475, 649], [901, 641]]),
            "open": (0, [[620, 632], [657, 639], [570, 628]]),
            "iron_arc": (18, [[281, 611], [309, 609]]),
            "cones": (25, [[381, 498], [651, 477], [821, 526]]),
            "lining": (6, [[635, 602], [680, 603]]),
        }
        observed = {}
        for name, (wanted, p) in probes.items():
            p = np.asarray(p)
            observed[name] = ob[p[:, 1], p[:, 0]].tolist()
        # Compare the aperture in an otherwise identical copy of the mesh with the cones excluded.
        isolated = Surface(folder / "asset.glb", exclude={25})

        def projected_occlusion(camera, directions):
            origins = np.broadcast_to(camera, directions.shape)
            without = isolated.cast(origins, directions)[1]
            with_cones = surface.cast(origins, directions)[1]
            # Only rays that would hit the bore below the bearing count; exclude the metal lip.
            # Rays that enter the shaft under the mesh's distant cut boundary
            # would have already hit real pavement before reaching the opening.
            zdir = directions[..., 2]
            t = -camera[2] / np.where(abs(zdir) > 1e-9, zdir, np.nan)
            cross = camera[:2] + t[..., None] * directions[..., :2]
            target = (
                np.isin(without, [3, 6, 7]) & (t > 0) & (np.linalg.norm(cross, axis=-1) < RADIUS)
            )
            blocked = target & (with_cones == 25)
            return dict(
                bore_pixels=int(target.sum()),
                covered_by_cone_px=int(blocked.sum()),
                fraction=float(blocked.sum() / max(target.sum(), 1)),
            )

        fractions = {"source": projected_occlusion(eye, r)}
        for name, c in zip(("left", "far", "right"), cones):
            center = np.array(c["center_xy_m"])
            camera = np.r_[center / np.linalg.norm(center) * 3.4, 1.28]
            direction, _ = view.view_rays(camera, np.array([0.0, 0.0, 0.0]), 726, 640, 400)
            fractions[name] = projected_occlusion(camera, direction)
        with np.load(folder / "surface.npz") as field:
            area = (
                field["opening_mask"].sum()
                * field["width"]
                * field["length"]
                / np.prod(np.array(field["height"].shape) - 1)
            )
            assert np.array_equal(
                field["opening_mask"], ~field["observed_depth_valid"] & field["alpha"]
            )
        # Compare vertical rays through the notch to the undamaged rim on both sides.
        bounds = meta["ring"]["notch_angle_rad"]
        th = np.array([bounds[0] - 0.04, np.mean(bounds), bounds[1] + 0.04])
        p = xyz((RADIUS + 0.0015) * np.stack([np.cos(th), np.sin(th)], -1), 1.0)
        dist, _, _ = original_aperture.cast(p, np.tile([0.0, 0.0, -1.0], (3, 1)))
        notch = (1 - dist) * 1000
        report = dict(
            source_cone_mask_iou=iou,
            recorded_parts=observed,
            cone_occlusion=fractions,
            opening_plan_m2=float(area),
            rim_test_z_mm=notch.tolist(),
            registration=registration,
        )
        print(json.dumps(report), flush=True)
        assert all(all(v == probes[name][0] for v in group) for name, group in observed.items())
        assert min(iou) > 0.93 and abs(area - np.pi * S["bore_radius_assumed_m"] ** 2) < 0.001
        assert notch[1] < -5 and max(abs(notch[[0, 2]])) < 1
        assert fractions["source"]["covered_by_cone_px"] > 0 and all(
            q["bore_pixels"] > 500 for q in fractions.values()
        )
        return report

    inspect(models, out, evidence, fit_tall=True, ids={"M571"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("out", type=Path)
    check(**vars(p.parse_args()))
