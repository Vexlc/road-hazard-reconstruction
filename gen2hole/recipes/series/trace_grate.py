"""Register the cover plane and recover each through-opening from a marked RGB photograph."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def image_patch(image, box):
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("Invalid photograph patch")
    return image[y0:y1, x0:x1]


def opening_loops(mask):
    n = len(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or (hierarchy[0, :, 3] >= 0).any():
        raise ValueError("Each opening must have one boundary with no interior islands")
    loops = []
    for c in contours:
        polygon = cv2.approxPolyDP(c, 0.75, True)[:, 0].astype(float) / (n - 1)
        if len(polygon) < 3 or cv2.contourArea(c) < 8:
            raise ValueError("An opening has too few source pixels")
        loops.append(polygon.tolist())
    return loops


def measure_side(quad, unit, config):
    """Recover perspective length and side depth up to the supplied reference width."""
    width = float(config["reference_width_m"])
    center = np.asarray(config["principal_xy"], float)
    corner = config["bottom_corner_index"]
    bottom = np.asarray(config["bottom_corner_xy"], float)
    if not width > 0 or center.shape != (2,) or bottom.shape != (2,) or corner not in range(4):
        raise ValueError("Invalid reference width or corresponding lower corner")
    h = cv2.getPerspectiveTransform(unit, quad)
    centered = h.copy()
    centered[:2] -= center[:, None] * h[2]
    a, b = centered[:, 0], centered[:, 1]
    denominator = a[2] * b[2]
    if abs(denominator) < 1e-8:
        raise ValueError("The photographed rectangle has insufficient perspective for focal estimation")
    focal_squared = -(a[:2] @ b[:2]) / denominator
    if not np.isfinite(focal_squared) or focal_squared <= 0:
        raise ValueError("Incompatible rectangle or optical centre")
    focal = np.sqrt(focal_squared)
    k = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]])
    axes = np.linalg.solve(k, h)
    axes *= width / np.linalg.norm(axes[:, 0])
    length = np.linalg.norm(axes[:, 1])
    rx = axes[:, 0] / width
    ry = axes[:, 1] / length
    down = np.cross(rx, ry)
    point = axes @ np.r_[unit[corner], 1]
    ray0 = k @ point
    dz = k @ down
    gradient = dz[:2] - bottom * dz[2]
    if np.linalg.norm(gradient) < 1e-7:
        raise ValueError("Side edge projects onto a point")
    thickness = float(gradient @ (bottom * ray0[2] - ray0[:2]) / (gradient @ gradient))
    if not 0 < thickness < min(width, length) / 2:
        raise ValueError("Implausible visible side thickness")
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([rx, -ry, -down])
    transform[:3, 3] = axes @ np.array([0.5, 0.5, 1])
    projected = k @ (point + thickness * down)
    return dict(
        size_m=[width, float(length)],
        thickness_m=thickness,
        reference_width_m=width,
        camera_K=k.tolist(),
        camera_from_cover=transform.tolist(),
        lower_corner_fit_error_px=float(np.linalg.norm(projected[:2] / projected[2] - bottom)),
        assumptions="Orthogonal rectangle, square pixels, supplied optical centre and reference width. Lower corner sets extrusion depth.",
    )


def trace(spec):
    if spec["perforations"]["shape"] == "polygon":
        from trace_polygon import trace as polygon
        return polygon(spec)
    refs = ROOT / "references"
    calibration = ROOT / "calibration"
    profile = spec["perforations"]
    frame = spec["frame"].get("original_reference", "frame.png")
    photo = np.array(Image.open(refs / frame).convert("RGB"))
    if frame != "frame.png":
        Image.fromarray(photo).save(refs / "frame.png")
    holes = np.array(Image.open(refs / profile["mask"]).convert("L")) > 127
    solid = profile.get("solid_face", False)
    if holes.shape != photo.shape[:2] or holes.any() == solid:
        raise ValueError("Mark the openings on an image aligned to the photograph")
    quad = np.asarray(profile["corners_xy"], dtype="f4")
    if (
        quad.shape != (4, 2)
        or not np.isfinite(quad).all()
        or not cv2.isContourConvex(quad)
        or cv2.contourArea(quad, oriented=True) <= 16
        or (quad < 0).any()
        or (quad >= np.array(photo.shape[1::-1])).any()
    ):
        raise ValueError("Mark four clockwise corners in the photograph starting at the top left")
    n = 768
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    calibration3d = None
    if "edge_calibration" in profile:
        if profile["shape"] != "rect":
            raise ValueError("Perspective side fitting requires a rectangle")
        calibration3d = measure_side(quad, unit, profile["edge_calibration"])
        size = np.array(calibration3d["size_m"])
        thickness = calibration3d["thickness_m"]
    else:
        size = np.asarray(profile["size_m"], float)
        thickness = float(profile["thickness_m"])
    if (
        size.shape != (2,)
        or not np.isfinite(size).all()
        or min(size) <= 0
        or not np.isfinite(thickness)
        or thickness <= 0
    ):
        raise ValueError("Expected positive finite cover dimensions in metres")
    mapping = cv2.getPerspectiveTransform(quad, unit * (n - 1))
    canonical = cv2.warpPerspective(photo, mapping, (n, n))
    mask = cv2.warpPerspective(np.uint8(holes), mapping, (n, n), flags=cv2.INTER_NEAREST)
    # A perforation cannot touch the outside of its own panel.
    if np.any(mask[[0, -1]]) or np.any(mask[:, [0, -1]]):
        raise ValueError("An opening touches the cover perimeter")
    loops = [] if solid else opening_loops(mask)
    variants = {}
    for state in spec["states"]:
        if "break_polygon_uv" not in state:
            continue
        cut = np.asarray(state["break_polygon_uv"], dtype=float)
        if (
            state.get("absent")
            or cut.ndim != 2
            or cut.shape[1] != 2
            or len(cut) < 3
            or not np.isfinite(cut).all()
            or ((cut <= 0) | (cut >= 1)).any()
        ):
            raise ValueError("Trace an interior break on a present panel")
        damaged = mask.copy()
        cv2.fillPoly(damaged, [np.rint(cut * (n - 1)).astype("i4")], 1)
        repaired = opening_loops(damaged)
        if len(repaired) >= len(loops) or (damaged == mask).all():
            raise ValueError("A broken web must connect neighbouring openings")
        key = state["id"]
        variants[key] = repaired
        Image.fromarray(damaged * 255).save(calibration / (key + "_openings.png"))
    Image.fromarray(canonical).save(refs / "lid_ortho.png")
    Image.fromarray(np.uint8(mask) * 255).save(calibration / "openings.png")
    if "displacement" in profile:
        source = np.asarray(Image.open(refs / profile["displacement"]))
        if source.shape != holes.shape or not np.issubdtype(source.dtype, np.integer):
            raise ValueError("Displacement must be an aligned single-channel integer image")
        values = cv2.warpPerspective(source.astype("f4"), mapping, (n, n))
        np.save(calibration / "relief.npy", values)
    if 'pavement' in profile:
        road = np.array(Image.open(refs / profile['pavement']).convert('RGB'))
    else:
        road = image_patch(photo, profile["road_patch_xyxy"])
    Image.fromarray(road).save(refs / "pavement.jpg", quality=95)
    Image.fromarray(image_patch(photo, profile["edge_patch_xyxy"])).save(refs / "iron.png")
    band = float(profile["frame_band_m"])
    offsets = np.array([s.get("offset_m", [0, 0]) for s in spec["states"]])
    span = float(np.max(size + 2 * np.maximum(band, abs(offsets).max(axis=0))) + 0.32)
    y, x = np.mgrid[:n, :n] / (n - 1)
    for s in spec['states']:
        fragment = s.get('fragment', {})
        if fragment.get('location') == 'road':
            path = np.asarray(profile['fracture_path_uv'])
            metal = (mask == 0) & (x >= np.interp(y, path[:, 1], path[:, 0]))
            if profile['shape'] == 'round':
                metal &= (x - .5)**2 + (y - .5)**2 <= .25
            if not metal.any():
                raise ValueError('No fragment under the marked path')
            xy = np.column_stack([x[metal] - .5, .5 - y[metal]]) * size
            angle = np.deg2rad(fragment.get('yaw_deg', 0))
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            moved = (xy - xy.mean(0)) @ rot.T + fragment['center_xy_m']
            span = max(span, 2 * (np.abs(moved).max() + 0.10))
    uv = np.stack([0.5 + (x - 0.5) * span / size[0], 0.5 + (y - 0.5) * span / size[1]], -1).astype("f4")
    coords = cv2.perspectiveTransform(uv, cv2.getPerspectiveTransform(unit, quad))
    density = (cv2.contourArea(quad) / np.prod(size)) ** 0.5
    canvas = cv2.remap(
        road,
        np.float32((x - 0.5) * span * density),
        np.float32((y - 0.5) * span * density),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if profile["photographed_surround"]:
        src = cv2.remap(photo, coords, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        h, w = photo.shape[:2]
        margin = np.minimum.reduce(
            [coords[:, :, 0], w - 1 - coords[:, :, 0], coords[:, :, 1], h - 1 - coords[:, :, 1]]
        )
        radius = np.max(abs(uv - 0.5), axis=2)
        fade = np.clip(margin / 20, 0, 1) * np.clip((1.01 - radius) / 0.13, 0, 1)
        if 'surround_mask' in profile:
            opacity = np.asarray(Image.open(refs / profile['surround_mask']).convert('L'))
            if opacity.shape != photo.shape[:2]:
                raise ValueError('The pavement opacity must match the source photograph')
            fade *= cv2.remap(opacity, coords, None, cv2.INTER_LINEAR).astype('f4') / 255
        if 'pavement' in profile:
            pos = (uv - 0.5) * size
            rim = pos / (size / 2 + band)
            radius = np.linalg.norm(rim, axis=2) if profile['shape'] == 'round' else abs(rim).max(2)
            fade *= np.clip((1.037 - radius) / 0.037, 0, 1)
        canvas = np.uint8(np.rint(src * fade[:, :, None] + canvas * (1 - fade[:, :, None])))
    Image.fromarray(canvas).save(refs / "apron.png")
    (calibration / "grate.json").write_text(
        json.dumps(
            dict(
                pixel_to_canonical=mapping.tolist(),
                resolution=n,
                size_m=size.tolist(),
                thickness_m=thickness,
                source_camera=calibration3d,
                openings_uv=loops,
                broken_openings_uv=variants,
                road_span_m=span,
            ),
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    trace(json.loads((ROOT / "calibration/survey.json").read_text()))
