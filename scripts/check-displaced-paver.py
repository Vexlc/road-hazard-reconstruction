"""Measure the observed paver on reserved camera rays and export an image-coordinate height probe."""

import argparse
import base64
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gen2hole"), str(ROOT / "gen2hole/downstream/sensors")]
from pit_captures import read_views
from road_geometry import Surface, read_json, sha256
from single_depth_surfaces import paver_labels, split_depth


def fit_top(points):
    xy = points[:, :2]
    design = np.column_stack((xy, np.ones(len(xy))))
    keep = np.ones(len(xy), bool)
    for _ in range(4):
        coef = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)[0]
        diff = points[:, 2] - design @ coef
        keep = abs(diff) < max(0.003, 3 * 1.4826 * np.median(abs(diff[keep])))
    return coef


def quantize(h):
    valid = np.isfinite(h)
    code = np.zeros(h.shape, "<u2")
    q = np.rint(h[valid] * 10000) + 32768
    if np.any((q < 1) | (q > 65535)):
        raise ValueError("Heights exceed the signed 0.1 mm probe range")
    code[valid] = q.astype("u2")
    return base64.b64encode(code.tobytes()).decode("ascii")


def assess(run, key):
    bundle = run / "series"
    cal = bundle / "calibration"
    spec = read_json(cal / (key + "_capture.json"))["spec"]
    view = read_views(
        spec, ROOT / "work/reconstruction-inputs", ROOT / "catalog/capture-inputs.json", cal
    )[spec["anchor"]]
    geometry = read_json(cal / (key + ".json"))
    center = np.asarray(geometry["local_origin_in_anchor_m"])
    basis = np.asarray(geometry["local_axes_in_anchor_camera"])
    labels = paver_labels(spec["surface_regions"], view)
    target = view.mask | (labels > 0)
    source_valid = view.depth > 0
    train, test = split_depth(view)
    xyz = np.zeros((*view.mask.shape, 3))
    y, x = np.where(target)
    rays = view.xyz(np.column_stack((x, y)), np.ones(len(x))) @ basis
    eye = -center @ basis
    xyz[y, x] = eye + rays * view.depth[y, x, None]
    actual = np.where(target & source_valid, xyz[:, :, 2], np.nan)
    path = bundle / "assets/pothole" / key / "asset.glb"
    distance, material, _ = Surface(path).cast(np.broadcast_to(eye, rays.shape), rays)
    computed = np.full(target.shape, np.nan)
    computed[y, x] = np.where(np.isfinite(distance), eye[2] + rays[:, 2] * distance, np.nan)
    mask = np.zeros(target.shape, "u1")
    mask[y, x] = material
    directory = run / "evaluation" / key
    valid = target & test & np.isfinite(computed)
    if not np.any((~source_valid) & target):
        raise ValueError("Expected raw range gaps in this wet capture")
    inner = ndimage.binary_erosion(labels == 1, iterations=2)
    coef = fit_top(xyz[inner & (train.depth > 0)])
    independent = inner & valid
    points = xyz[independent]
    est = points.copy()
    yy, xx = np.where(independent)
    directions = view.xyz(np.column_stack((xx, yy)), np.ones(len(xx))) @ basis
    t = (computed[independent] - eye[2]) / directions[:, 2]
    est[:] = eye + directions * t[:, None]
    real_coef = fit_top(points)
    mesh_coef = fit_top(est)
    axis = coef[:2] / np.linalg.norm(coef[:2])
    origin = xyz[inner & (train.depth > 0), :2].mean(0)
    along = (points[:, :2] - origin) @ axis
    low, high = np.percentile(along, [1, 99])
    bins = []
    for left, right in zip(np.linspace(low, high, 13)[:-1], np.linspace(low, high, 13)[1:]):
        keep = (along >= left) & (along < right)
        if keep.sum() < 6:
            raise ValueError("Insufficient reserved samples along the paver face")
        bins.append(
            dict(
                position_m=(left + right) / 2,
                samples=int(keep.sum()),
                measured_m=float(np.median(points[keep, 2])),
                mesh_m=float(np.median(est[keep, 2])),
            )
        )
    sign_margin = 2 * geometry["road_fit_rms_m"]
    sides = {}
    for name, selected in [("above", points[:, 2] > sign_margin), ("below", points[:, 2] < -sign_margin)]:
        if selected.sum() < 15:
            raise ValueError("The reserved observations do not resolve both sides of the road plane")
        sides[name] = dict(
            samples=int(selected.sum()),
            median_signed_height_mm=float(1000 * np.median(points[selected, 2])),
            mesh_same_sign=int((np.sign(est[selected, 2]) == np.sign(points[selected, 2])).sum()),
        )
    # Quantized to 0.1 mm. Zero denotes a missing source range or missed mesh ray.
    flags = target.astype("u1") | (view.mask.astype("u1") << 1) | (labels << 2) | (test.astype("u1") << 4)
    examples = []
    for title, fraction in [("砖的高端", 95), ("砖的低端", 5)]:
        i = np.argmin(abs(points[:, 2] - np.percentile(points[:, 2], fraction)))
        examples.append(dict(label=title, xy=[int(xx[i]), int(yy[i])]))
    for title, subset in [
        ("外露侧面", valid & (labels == 2) & (mask == 14)),
        ("缺少量程处", target & ~source_valid & np.isfinite(computed)),
    ]:
        if subset.any():
            y0, x0 = np.unravel_index(ndimage.distance_transform_edt(subset).argmax(), subset.shape)
            examples.append(dict(label=title, xy=[int(x0), int(y0)]))
    payload = dict(
        width=target.shape[1],
        height=target.shape[0],
        examples=examples,
        raw=quantize(actual),
        mesh=quantize(computed),
        flags=base64.b64encode(flags.tobytes()).decode("ascii"),
        first=base64.b64encode(mask.tobytes()).decode("ascii"),
    )
    (directory / "source-probe.js").write_text(
        "window.sourceHeightProbe("
        + json.dumps(key)
        + ","
        + json.dumps(payload, separators=(",", ":"))
        + ");\n"
    )
    levels = [-0.11, -0.06, -0.018, 0, 0.018, 0.040]
    colors = np.array(
        [[44, 69, 112], [67, 135, 172], [136, 200, 204], [229, 227, 212], [239, 172, 88], [168, 71, 52]]
    )
    for name, heights in [("measured", actual), ("mesh", computed)]:
        image = (view.rgb * 0.38 + 80).astype("u1")
        finite = np.isfinite(heights)
        image[target & ~finite] = [45, 46, 51]
        image[finite] = np.column_stack([np.interp(heights[finite], levels, c) for c in colors.T]).astype(
            "u1"
        )
        for i in [1, 2]:
            contours, _ = cv2.findContours(
                np.uint8(labels == i), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, (46, 42, 36), 1)
        Image.fromarray(image).save(directory / (name + "-height.png"))
    source = view.rgb.copy()
    for region, color in [(view.mask, (230, 197, 69)), (labels > 0, (32, 207, 185))]:
        contours, _ = cv2.findContours(region.astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(source, contours, -1, color, 2, cv2.LINE_AA)
    Image.fromarray(source).save(directory / "source-outlines.png")
    native_check = read_json(run / "evaluation/measurements.json")[key]
    if native_check["glb_sha256"] != sha256(path):
        raise ValueError("The measured model and the height probe refer to different GLBs")
    angles = [float(np.degrees(np.arctan(np.linalg.norm(c[:2])))) for c in (coef, real_coef, mesh_coef)]
    return dict(
        key=key,
        glb_sha256=sha256(path),
        camera=spec["anchor"],
        withheld_optical_range=native_check["withheld_mesh"],
        visible_region_checks=native_check["region_results"],
        isolated_mesh=native_check["comparison_meshes"]["isolated"],
        source_size_px=list(view.rgb.shape[1::-1]),
        original_annotation_pixels=int(view.mask.sum()),
        paver_pixels=int((labels > 0).sum()),
        paver_pixels_outside_original=int(((labels > 0) & ~view.mask).sum()),
        target_pixels=int(target.sum()),
        absent_raw_returns_in_target=int((target & ~source_valid).sum()),
        source_road_fit_rms_mm=1000 * geometry["road_fit_rms_m"],
        reserve_count=int(valid.sum()),
        top_interior_reserved=int(independent.sum()),
        training_top_tilt_deg=angles[0],
        heldout_top_tilt_deg=angles[1],
        mesh_top_tilt_deg=angles[2],
        top_height_p05_p95_mm=(1000 * np.percentile(points[:, 2], [5, 95])).tolist(),
        signs_beyond_two_road_rms=sides,
        slope_section=bins,
        probe_height_quantization_mm=0.1,
        probe_sha256=sha256(directory / "source-probe.js"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(assess(args.run, args.id), indent=2, allow_nan=False) + "\n")
