"""Driving clearance, calibrated stereo foreground and lighting observations for above-road objects."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image


def registered_geometry(folder):
    """Identify the actual geometry used for support and clearance at the time of survey."""
    from road_geometry import sha256
    record = {name: sha256(folder / name) for name in ("asset.glb", "surface.npz")}
    if (folder / "samples.npz").is_file():
        record["samples.npz"] = sha256(folder / "samples.npz")
    return record


def check_geometry(anchor, folder, *, native=False):
    records = anchor.get("registered_geometry")
    if records is None:  # Accept original anchors created before per-file survey records.
        return
    from road_geometry import sha256
    names = ("asset.glb", "surface.npz", "samples.npz") if native else ("asset.glb", "surface.npz")
    actual = {k: sha256(folder / k) for k in names}
    if not any(all(r.get(k) == v for k, v in actual.items()) for r in records):
        raise ValueError("This asset differs from the geometry used to survey road support and clearance")


def surveyed_road_pixels(origin, directions, plane, host, anchor, receivers, band):
    """Identify metric road returns supported by the survey on both ends of the registration error."""
    from prop_lighting import ground_confidence
    if not np.isfinite(band) or not 0 < band <= .10:
        raise ValueError("Use a positive surveyed road-normal uncertainty of at most 10 cm")
    if receivers is None and "ground_receivers" not in anchor:
        raise ValueError("Surveyed road reconciliation needs registered receiver support")
    # A height tolerance alone admits arbitrarily distant surfaces at the optical horizon.
    # Limit the displacement to three local survey cells, and reject upward-looking rays.
    pitch = anchor.get("ground_receivers", {}).get("pitch_m", .12)
    finite = np.isfinite(host) & np.isfinite(plane)
    error = np.abs(np.subtract(host, plane, out=np.zeros_like(plane), where=finite))
    candidate = (finite & (plane > 0) & (directions[..., 2] < 0) & (error <= 3 * pitch)
                 & (error * abs(directions[..., 2]) <= band))
    if receivers is not None:
        candidate &= receivers
    if "ground_receivers" in anchor and candidate.any():
        o = np.broadcast_to(origin, directions.shape)[candidate, :2]
        d = directions[candidate, :2]
        measured = o + d * host[candidate, None]
        footprint = o + d * plane[candidate, None]
        candidate[candidate] = (ground_confidence(anchor, measured) >= .8) & (ground_confidence(anchor, footprint) >= .8)
    return candidate


def driving_clearance(anchor, bounds_xy, positions, *, half_width_m, half_length_m, margin_m=0.12):
    """Sweep the measured vehicle-reference trajectory past the full asset envelope in its support plane."""
    if (
        not np.isfinite([half_width_m, half_length_m, margin_m]).all()
        or min(half_width_m, half_length_m) <= 0
        or margin_m < 0
    ):
        raise ValueError("Invalid vehicle dimensions or clearance")
    positions = np.asarray(positions, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or len(positions) < 2
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Expected two or more finite XYZ vehicle centres")
    xy = ((positions - anchor["center_world"]) @ np.asarray(anchor["basis_columns"]))[:, :2]
    travel = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    moving = np.r_[True, np.diff(travel) > 0]
    if travel[-1] < 0.2:
        raise ValueError("Supply a moving metric vehicle trajectory to check clearance")
    s = np.linspace(0, travel[-1], int(np.ceil(travel[-1] / 0.075)) + 1)
    track = np.column_stack([np.interp(s, travel[moving], axis[moving]) for axis in xy.T])
    # Use the measured path tangent for each arc-length sample, including reversals.
    segments = np.diff(xy[moving], axis=0)
    segment_index = np.searchsorted(travel[moving][1:], s, side="right").clip(max=len(segments) - 1)
    direction = np.gradient(track, axis=0)
    degenerate = np.linalg.norm(direction, axis=1) < 1e-8
    direction[degenerate] = segments[segment_index[degenerate]]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    bounds_xy = np.asarray(bounds_xy, dtype=float)
    if bounds_xy.shape != (2, 2) or not np.isfinite(bounds_xy).all() or np.any(bounds_xy[1] <= bounds_xy[0]):
        raise ValueError("Expected a finite nonempty XY object envelope")
    lo, hi = bounds_xy
    asset = np.float32((corners + 1) / 2 * (hi - lo) + lo)
    best = np.inf
    for center, axis in zip(track, direction):
        local = corners * [half_length_m, half_width_m]
        ego = np.float32(center + local[:, 0, None] * axis + local[:, 1, None] * [-axis[1], axis[0]])
        if cv2.intersectConvexConvex(ego, asset)[0] > 1e-8:
            raise ValueError("Object intersects the vehicle clearance envelope: measured overlap")
        dist = min(
            -cv2.pointPolygonTest(a, tuple(v), True) for a, b in [(ego, asset), (asset, ego)] for v in b
        )
        best = min(best, dist)
    if best < margin_m:
        raise ValueError(
            f"Object intersects the vehicle clearance envelope: {best:.3f} m, required {margin_m:.3f} m"
        )
    return dict(
        minimum_clearance_m=best,
        required_margin_m=margin_m,
        half_width_m=half_width_m,
        half_length_m=half_length_m,
        trajectory_samples=len(track),
        sample_spacing_m=float(s[1] - s[0]),
    )


@lru_cache(maxsize=2)
def stereo_rig(root):
    rows = dict(
        line.split(":", 1)
        for line in (root / "calibration/perspective.txt").read_text().splitlines()
        if ":" in line
    )
    a, b = [np.fromstring(rows["P_rect_0" + str(i)], sep=" ").reshape(3, 4) for i in (0, 1)]
    if not np.allclose(a[:, :3], b[:, :3]) or not a[0, 3] > b[0, 3]:
        raise ValueError("Expected the KITTI horizontal rectified stereo rig")
    times = [
        np.char.replace(np.loadtxt(root / c / "timestamps.txt", dtype=str, delimiter="\t"), " ", "T").astype(
            "datetime64[ns]"
        )
        for c in ["image_00", "image_01"]
    ]
    return a, b, times


@lru_cache(maxsize=3)
def stereo_ranges(left):
    """Left-right checked foreground range from the synchronized camera pair. Unknown depth stays unknown."""
    root = left.parents[1]
    a, b, times = stereo_rig(root)
    frame = int(left.stem)
    if abs((times[0][frame] - times[1][frame]) / np.timedelta64(1, "ms")) > 1:
        raise ValueError("Stereo exposures differ by over one millisecond")
    right = root / "image_01" / left.name
    imgs = [np.asarray(Image.open(f).convert("L")) for f in (left, right)]
    params = dict(
        numDisparities=160,
        blockSize=5,
        P1=8 * 25,
        P2=32 * 25,
        uniquenessRatio=9,
        speckleWindowSize=70,
        speckleRange=2,
        disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    dl = cv2.StereoSGBM_create(minDisparity=0, **params).compute(*imgs).astype("f4") / 16
    dr = cv2.StereoSGBM_create(minDisparity=-159, **params).compute(*imgs[::-1]).astype("f4") / 16
    yy, xx = np.indices(dl.shape)
    rx = xx - np.rint(dl).astype(int)
    check = dr[yy, np.clip(rx, 0, dl.shape[1] - 1)]
    valid = (dl > 1) & (rx >= 0) & (rx < dl.shape[1]) & (check < 0) & (abs(dl + check) < 1.0)
    factor = np.sqrt(1 + ((xx - a[0, 2]) / a[0, 0]) ** 2 + ((yy - a[1, 2]) / a[1, 1]) ** 2)
    # A one-pixel disparity allowance moves ambiguous near surfaces farther away.
    result = np.full(dl.shape, np.inf, "f4")
    result[valid] = factor[valid] * (a[0, 3] - b[0, 3]) / np.maximum(dl[valid] - 1.0, 0.5)
    return result
