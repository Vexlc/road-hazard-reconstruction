"""Recover the plane of a cast tread lattice and trace an angular cover with perimeter recesses."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def mapped(points, matrix):
    p = np.asarray(points, dtype=float)
    uv = np.c_[p, np.ones(len(p))] @ matrix.T
    return uv[:, :2] / uv[:, 2:]


def lattice(photo, spec, corners):
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    u, v = spec["template_xy"]
    w, h = spec["template_half_size_xy"]
    if min(w, h) < 1 or not (0 <= u - w <= u + w < gray.shape[1] and 0 <= v - h <= v + h < gray.shape[0]):
        raise ValueError("The tread patch must be entirely inside the source photograph")
    template = gray[v - h : v + h + 1, u - w : u + w + 1]
    corr = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    peak = cv2.dilate(corr, np.ones((min(w, h) * 2 + 1,) * 2, "u1"))
    y, x = np.where((corr == peak) & (corr >= spec["correlation_min"]))
    px, py = x + w, y + h
    dx, dy = spec["rough_horizontal_period_px"], spec["rough_row_step_px"]
    if min(dx, dy) <= 0 or not np.isfinite([dx, dy]).all():
        raise ValueError("Tread spacing must be finite and positive")
    row = np.rint((py - v) / dy).astype(int)
    col = np.rint((px - u) / dx - 0.5 * (row % 2)).astype(int)
    s = col + 0.5 * (row % 2)
    valid = np.hypot(px - u - s * dx, py - v - row * dy) <= spec["search_radius_px"]
    boundary = np.int32(corners)
    valid &= np.array(
        [cv2.pointPolygonTest(boundary, (float(i), float(j)), False) > 0 for i, j in zip(px, py)],
        dtype=bool,
    )
    xy = np.c_[px[valid], py[valid]]
    # The neighbouring tread centres form equilateral triangles. Unit horizontal period
    # corresponds to two columns; one staggered row advances 1/(2 sqrt(3)).
    uv = np.c_[s[valid], row[valid] / (2 * np.sqrt(3))]
    col, row = col[valid], row[valid]
    fit = (col + row * 2) % 3 != 0
    if len(np.unique(uv, axis=0)) != len(uv) or fit.sum() < 8 or (~fit).sum() < 4:
        raise ValueError("Need disjoint, unambiguous repeated junctions spanning the cover")
    H, _ = cv2.findHomography(uv[fit], xy[fit], 0)
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
        raise ValueError("The tread junctions do not determine a plane")
    errors = np.linalg.norm(mapped(uv, H) - xy, axis=1)
    return np.linalg.inv(H), dict(
        image_xy=xy.tolist(),
        lattice_xy=uv.tolist(),
        fit=fit.tolist(),
        residual_px=errors.tolist(),
        heldout_p95_px=float(np.percentile(errors[~fit], 95)),
    )


def trace(spec):
    profile = spec["perforations"]
    refs, cal = ROOT / "references", ROOT / "calibration"
    source = spec["frame"].get("original_reference", "frame.png")
    photo = np.array(Image.open(refs / source).convert("RGB"))
    Image.fromarray(photo).save(refs / "frame.png")
    corners = np.asarray(profile["rim_corners_xy"], float)
    if (
        corners.ndim != 2
        or corners.shape[1] != 2
        or len(corners) < 3
        or not np.isfinite(corners).all()
        or not cv2.isContourConvex(corners.astype("f4"))
        or len(corners) != len(profile["recesses_xy"])
    ):
        raise ValueError("Supply a convex perimeter and one recess list per edge")
    matrix, landmarks = lattice(photo, profile["tread"], corners)
    rect = mapped(corners, matrix)
    lo, hi = rect.min(0), rect.max(0)
    size = (hi - lo) * profile["reference_width_m"] / (hi[0] - lo[0])
    n = 768
    unit = np.diag([1 / (hi[0] - lo[0]), 1 / (hi[1] - lo[1]), 1.0])
    unit[:2, 2] = -lo / (hi - lo)
    norm = unit @ matrix
    can = np.diag([n - 1, n - 1, 1]) @ norm
    face = []
    recesses = []
    for a, b, edge in zip(corners, np.roll(corners, -1, axis=0), profile["recesses_xy"]):
        face.append(a)
        if not edge:
            continue
        curve = np.asarray(edge, float)
        vec = b - a
        along = (curve - a) @ vec / (vec @ vec)
        if not (0 < along[0] < along[-1] < 1):
            raise ValueError("Trace each notch along its perimeter direction")
        endpoints = a + along[[0, -1], None] * vec
        inside = cv2.convexHull(corners.astype("f4"))
        if any(cv2.pointPolygonTest(inside, tuple(q), True) <= 0 for q in curve[1:-1]):
            raise ValueError("Recess leaves the panel")
        curve = np.vstack((endpoints[0], curve[1:-1], endpoints[1]))
        face.extend(curve)
        recesses.append(mapped(curve, norm).tolist())
    outline = mapped(face, norm)
    canonical = cv2.warpPerspective(photo, can, (n, n))
    Image.fromarray(canonical).save(refs / "lid_ortho.png")

    def patch(name, key):
        x, y, r, b = profile[key]
        if not (0 <= x < r <= photo.shape[1] and 0 <= y < b <= photo.shape[0]):
            raise ValueError(f"Texture patch outside the surveyed photograph: {key}")
        tile = photo[y:b, x:r]
        Image.fromarray(tile).save(refs / name)
        return tile

    road = patch("pavement.jpg", "road_patch_xyxy")
    patch("iron.png", "edge_patch_xyxy")
    span = float(size.max() + 2 * profile["frame_band_m"] + 0.32)
    y, x = np.mgrid[:n, :n] / (n - 1)
    tex = np.stack([0.5 + (x - 0.5) * span / size[0], 0.5 + (y - 0.5) * span / size[1]], axis=-1)
    coords = mapped(tex.reshape(-1, 2), np.linalg.inv(norm)).reshape(n, n, 2).astype("f4")
    actual = cv2.remap(photo, coords, None, cv2.INTER_LINEAR)
    density = (n - 1) * span / max(size)
    fill = cv2.remap(
        road,
        ((x - 0.5) * density).astype("f4"),
        ((y - 0.5) * density).astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    fade = np.clip(
        np.minimum.reduce(
            [
                coords[:, :, 0],
                coords[:, :, 1],
                photo.shape[1] - 1 - coords[:, :, 0],
                photo.shape[0] - 1 - coords[:, :, 1],
            ]
        )
        / 12,
        0,
        1,
    )[:, :, None]
    Image.fromarray(np.uint8(actual * fade + fill * (1 - fade))).save(refs / "apron.png")
    # The inner black edge recesses are open to the external perimeter, not enclosed holes.
    marked = np.zeros(photo.shape[:2], "u1")
    for curve in recesses:
        cv2.fillPoly(marked, [np.rint(mapped(curve, np.linalg.inv(norm))).astype("i4")], 255)
    Image.fromarray(marked).save(cal / "perimeter-recesses.png")
    result = dict(
        pixel_to_canonical=can.tolist(),
        resolution=n,
        size_m=size.tolist(),
        thickness_m=profile["thickness_m"],
        source_camera=None,
        openings_uv=[],
        broken_openings_uv={},
        road_span_m=span,
        landmarks=landmarks,
        rim_uv=mapped(corners, norm).tolist(),
        lid_boundary_uv=outline.tolist(),
        recesses_uv=recesses,
    )
    (cal / "grate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
