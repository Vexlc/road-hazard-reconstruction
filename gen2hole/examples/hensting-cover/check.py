"""Test the recovered cover against the photo, sections of the exposed webs and two splat views."""

import argparse
import json
from importlib import import_module
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from camera import HERE, load

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from inspect_assets import inspect
from mesh_transfer import RoadMesh
from photo_mesh import Surface

render = import_module("check-verge-chamber").render


def check(models, out):
    s, cam = load()

    def field(folder, surf, meta, spec, run):
        h, w = s["size"][::-1]
        yy, xx = np.indices((h, w))
        ray = cam.rays(np.stack([xx, yy], axis=-1))
        rgb, parts, dist = render(RoadMesh(folder), cam.eye, ray)
        Image.fromarray(rgb).save(out / "source-view.png")
        with np.errstate(invalid="ignore"):
            cap = np.isin(parts, [9, 5]) & (
                abs(cam.eye[2] + ray[:, :, 2] * dist - cam.grade) < 0.00005
            )
        edge = cap & ~cv2.erode(cap.astype("u1"), np.ones((3, 3), "u1")).astype(bool)
        marks = np.array(s["held_out_edge_pixels"])
        delta = distance_transform_edt(~edge)[marks[:, 1], marks[:, 0]]
        probes = [
            dict(pixel=[x, y], expected=c, first=int(parts[y, x]))
            for c, x, y in s["registration_checks"]
        ]
        levels = {}
        for name, xy in [
            ("lid", [0, 0]),
            ("excavated shoulder", [0.49, -0.15]),
            ("milled road", [0.72, -0.15]),
        ]:
            origin = np.array([[*xy, 1.0]])
            d, part, _ = surf.cast(origin, np.array([[0.0, 0.0, -1.0]]))
            levels[name] = dict(part=int(part[0]), z_m=float(1.0 - d[0]))
        sections = []
        metal = Surface(folder / "asset.glb", exclude={1, 2, 3})
        for x, y, foot_z in cam.ribs():
            z = (foot_z + cam.grade - 0.016) / 2
            x_offsets = np.array([-0.031, 0.0, 0.031])
            origins = np.column_stack([x + x_offsets, np.full(3, -0.9), np.full(3, z)])
            # Ignore road and cut earth to isolate the component section.
            d, p, _ = metal.cast(origins, np.broadcast_to([0.0, 1.0, 0.0], origins.shape))
            sections.append(
                dict(
                    first=p.tolist(),
                    web_projects_forward_mm=float((np.mean(d[[0, 2]]) - d[1]) * 1000),
                )
            )
        report = dict(
            component_probes=probes,
            held_out_edge_distances_px=delta.tolist(),
            edge_rms_px=float(np.sqrt(np.mean(delta**2))),
            levels=levels,
            rib_sections=sections,
            frame_foot_depth_below_lid_m=(cam.grade - cam.ribs()[:, 2]).tolist(),
            pose_angles_degree=np.degrees(cam.angles).tolist(),
        )
        print(json.dumps(report), flush=True)
        assert all(p["expected"] == p["first"] for p in probes)
        assert delta.max() <= 3 and report["edge_rms_px"] <= 1.6
        assert levels["lid"]["part"] == 9 and abs(levels["lid"]["z_m"] - cam.grade) < 1e-5
        assert levels["excavated shoulder"]["z_m"] < -0.08 and levels["milled road"]["part"] == 1
        assert all(
            q["first"] == [5, 147, 5] and 4 < q["web_projects_forward_mm"] < 30 for q in sections
        )
        return report

    out.mkdir(parents=True, exist_ok=True)
    inspect(models, out, field, ids={"A565"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("out", type=Path)
    check(**vars(parser.parse_args()))
