"""Verify through-slots, camera calibration, source-view appearance and installation states."""

import argparse
import json
from io import BytesIO
from importlib import import_module
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from casting import Casting, HERE, S, rgb
from build import KEY, xyz
from photo_mesh import LIB, Surface
from inspect_assets import inspect
from road_geometry import glb_triangles

sys.path.insert(0, str(LIB.parent / "scripts"))
topology = import_module("check-perforated-covers").topology


def write(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=lambda n: n.item()) + "\n"
    )


def check(models, out):
    out.mkdir(parents=True, exist_ok=True)
    cam = Casting()
    file = models / "grate.glb"
    plate = Surface(file)
    top = topology(glb_triangles(file))
    assert top["through_holes"] == 8

    def normal_cast(surf, xy):
        o = xyz(xy, 1.0)
        return surf.cast(o, np.broadcast_to([0.0, 0.0, -1.0], o.shape))[:2]

    for x, code in [(cam.centres, 0), ((cam.centres[:-1] + cam.centres[1:]) / 2, 9)]:
        xx, yy = np.meshgrid(x, np.linspace(-0.14, 0.14, 31))
        _, part = normal_cast(plate, np.stack([xx, yy], -1))
        assert np.all(part == code)
    # Reproject the delivered GLB through the fitted original camera and compare
    # its sampled photographic albedo only on interior top-face pixels.
    im = rgb()
    h, w = im.shape[:2]
    y, x = np.indices((h, w))
    dirs = (
        np.stack(
            [(x - cam.K[0, 2]) / cam.K[0, 0], (y - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(x)], -1
        )
        @ cam.R
    )
    eye = -cam.R.T @ cam.t
    t, part, _ = plate.cast(np.broadcast_to(eye, dirs.shape), dirs)
    core = cv2.erode((part == 9).astype("u1"), np.ones((9, 9), "u1")).astype(bool)
    pos = eye + t[core, None] * dirs[core]
    _, v, _, uv, tex = next(row for row in glb_triangles(file, appearance=True) if row[0] == 9)
    co = np.linalg.lstsq(np.c_[v[:, :2], np.ones(len(v))], uv, rcond=None)[0]
    uv = np.c_[pos[:, :2], np.ones(len(pos))] @ co
    tex = np.asarray(Image.open(BytesIO(tex)).convert("RGB"))
    grid = (uv * tex.shape[1::-1] - 0.5).astype("f4")
    from scipy.ndimage import map_coordinates

    sampled = np.column_stack(
        [
            map_coordinates(ch, grid[:, ::-1].T, order=1, mode="nearest")
            for ch in tex.transpose(2, 0, 1)
        ]
    )
    diff = np.abs(sampled.astype(float) - im[core])
    assert diff.mean() < 4 and cam.tip_errors[~cam.train].max() < 4
    proj = im.copy()
    contours, _ = cv2.findContours((part == 9).astype("u1"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(proj, contours, -1, (244, 175, 47), 1, cv2.LINE_AA)
    for iy, (y, xt, xb) in enumerate(S["front_edge_traces_y_topx_toex"]):
        color = (51, 218, 195) if y in S["edge_holdout_rows"] else (207, 235, 119)
        cv2.line(proj, (xt, y), (xb, y), color, 2)
    cv2.imwrite(str(out / "reprojection.jpg"), cv2.cvtColor(proj, cv2.COLOR_RGB2BGR))
    report = dict(
        topology=top,
        normal_rays_through_slots=248,
        normal_rays_on_ribs=217,
        fitted_slot_span_m=cam.span,
        slot_widths_m=np.sum(cam.half_width, axis=1).tolist(),
        inferred_thickness_m=cam.thickness,
        thickness_residual_px=cam.edge_residual.tolist(),
        heldout_tip_errors_px=cam.tip_errors[~cam.train].tolist(),
        source_face_core_pixels=int(core.sum()),
        top_face_srgb_mae=diff.mean(),
        top_face_srgb_p90=np.percentile(diff, 90),
        camera_sensitivity=[],
    )
    for focal, offset in [(0.9, (0, 0)), (1.1, (0, 0)), (1.0, (-220, -160)), (1.0, (220, 160))]:
        test = Casting(focal, offset)
        report["camera_sensitivity"].append(
            dict(
                focal_scale=focal,
                principal_shift_px=offset,
                thickness_m=test.thickness,
                slot_span_m=test.span,
                heldout_tip_max_px=test.tip_errors[~test.train].max(),
            )
        )
    states = {}

    empty = Surface(models / "empty.glb")

    def evidence(folder, surf, meta, spec, run):
        xx, yy = np.meshgrid(np.linspace(-0.168, 0.168, 169), np.linspace(-0.166, 0.166, 167))
        _, parts = normal_cast(surf, np.stack([xx, yy], -1))
        inside = parts == 7
        assert set(np.unique(parts)).issubset({5, 7, 9})
        with np.load(folder / "surface.npz") as field:
            y, x = np.nonzero(field["first_part"] == 9)
            shape = np.array(field["first_part"].shape[::-1]) - 1
            sample = (np.c_[x, y] / shape - 0.5) * [field["width"], -field["length"]]
        # Uniform extruded iron has the same plan centroid as its painted face.
        # Compare its line of gravity with the convex hull of actual underside contacts.
        dist, _ = normal_cast(empty, sample)
        base = -cam.thickness if meta["state"] == "seated" else 0.0
        contact = np.isclose(1 - dist, base, atol=1.0e-4)
        hull = cv2.convexHull(np.float32(sample[contact]))
        margin = cv2.pointPolygonTest(hull, tuple(sample.mean(0)), True)
        if margin <= 0:
            raise ValueError("Unsupported centre of mass")
        return dict(
            static_contact_area_fraction=float(contact.mean()),
            centroid_inside_support_hull_m=margin,
            normal_rays=len(xx.ravel()),
            clear_to_optical_closure=int(inside.sum()),
            throat_open_fraction=float(inside.mean()),
            plate_first=int((parts == 9).sum()),
        )

    for state in S["relative_grate_poses"]:
        r = inspect(models / state, out / state, evidence, ids={KEY})
        states[state] = r[KEY]
    frac = [
        states[k]["evidence"]["throat_open_fraction"] for k in ("seated", "partial", "displaced")
    ]
    assert 0.2 < frac[0] < frac[1] < frac[2] and frac[2] == 1.0
    z, part = normal_cast(empty, np.array([[0.0, 0.0]]))
    assert part[0] == 7 and abs(z[0] - 1.0 - S["trial_socket_depth_m"]) < 1e-5
    report["states"] = states
    write(out / "quality.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("models", "out"):
        p.add_argument(k, type=Path)
    check(**vars(p.parse_args()))
