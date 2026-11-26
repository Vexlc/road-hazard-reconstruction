"""Test source-camera visibility, physical overlap and the distinction between measured and visual depth."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from calibration import Calibration, HERE, SIZE, SPEC, THICKNESS, circle
from casting import model_xy, perimeter

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from inspect_assets import inspect
from mesh_transfer import RoadMesh
from photo_mesh import Surface, xyz

render = import_module("check-verge-chamber").render


def check(models, out):
    cam = Calibration()

    def field(folder, surf, meta, spec, run):
        observed = Surface(folder / "observed.glb")
        yy, xx = np.indices(tuple(SIZE[::-1]))
        r = cam.rays(np.stack([xx, yy], axis=-1))
        rgb, parts, dist = render(RoadMesh(folder), cam.eye, r)
        Image.fromarray(rgb).save(out / "source-view.jpg", quality=96)
        _, obs, _ = observed.cast(np.broadcast_to(cam.eye, r.shape), r)
        checks = {}
        for label, points in SPEC["image_checks"].items():
            p = np.array(points)
            vals = obs[p[:, 1], p[:, 0]]
            checks[label] = vals.tolist()
        bore = Polygon(circle([0, 0], cam.ledge_radius, 0, 4096)[:, :2])
        lid = Polygon(perimeter(cam))
        covered = bore.intersection(lid).area / bore.area
        with np.load(folder / "surface.npz") as field:
            confidence = field["observed_depth_valid"]
            opened = field["opening_mask"]
            pitch = field["width"] / (opened.shape[1] - 1)
            pixel_area = pitch * field["length"] / (opened.shape[0] - 1)
            exposed_grid = opened.sum() * pixel_area
            assert np.array_equal(~confidence & field["alpha"], opened)
        glyph_px = [
            [158, 291],
            [186, 319],
            [303, 310],
            [321, 309],
            [455, 290],
            [475, 307],
            [476, 328],
        ]
        starts = xyz(model_xy(glyph_px, cam), 1.0)
        d, part, _ = observed.cast(starts, np.broadcast_to([0.0, 0.0, -1.0], starts.shape))
        h = 1 - d
        report = dict(
            edge_test=cam.edge_errors(),
            source_probes=checks,
            assumed_radius_m=0.30,
            lid_center_displacement_m=float(np.linalg.norm(cam.lid_xy)),
            bore_covered_plan_fraction=covered,
            open_plan_area_m2=bore.difference(lid).area,
            exposed_grid_area_m2=float(exposed_grid),
            sensor_shaft_no_return_px=int((obs == 0).sum()),
            source_aperture_no_depth_probes=SPEC["image_checks"]["exposed_shaft"],
            observed_iron_lower_edge_range_m=[
                float(cam.wall_samples[:, 2].min()),
                float(cam.wall_samples[:, 2].max()),
            ],
            glyph_heights_mm=np.round(h * 1000, 3).tolist(),
            focal_fit_px=cam.f,
            sensitivity={},
        )
        # Count only no-returns inside the observed aperture, not outside the mesh.
        projected = cam.project(circle([0, 0], cam.ledge_radius, cam.ledge_z))
        roi = np.zeros(obs.shape, "u1")
        cv2.fillPoly(roi, [np.int32(np.rint(projected))], 1)
        report["sensor_shaft_no_return_px"] = int(((obs == 0) & (roi > 0)).sum())
        for fixed in (660, 990):
            alt = Calibration(fixed)
            report["sensitivity"][str(fixed)] = dict(
                lid_offset_m=float(np.linalg.norm(alt.lid_xy)), edge_test=alt.edge_errors()
            )
        print(report, flush=True)
        assert all(
            all(x == expected for x in checks[name])
            for name, expected in [
                ("lid", 9),
                ("concrete", 1),
                ("back_wall", 6),
                ("exposed_shaft", 0),
            ]
        )
        assert 0.02 < covered < 0.15 and abs(bore.difference(lid).area - exposed_grid) < 0.001
        assert np.all(part == 9) and np.allclose(
            h, THICKNESS + 0.0035 * np.array([1, 0, 1, 0, 1, 1, 0]), atol=0.0001
        )
        assert max(e["rms_px"] for e in cam.edge_errors().values()) < 2
        return report

    out.mkdir(parents=True, exist_ok=True)
    inspect(models, out, field, ids={"M570"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("out", type=Path)
    check(**vars(p.parse_args()))
