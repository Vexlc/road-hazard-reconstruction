"""Triangulate registered pixel correspondences from two metric camera ray fields."""

import numpy as np

from host_depth import optical_samples, reference_samples, sampled
from host_position import register_positions


def optics(view, calibration):
    rays, protected, origins = optical_samples(view, calibration)
    pose, rays, origins = reference_samples(view, rays, origins, with_origins=True)
    return pose, rays, protected, origins


def pixel_map(value, shape, name):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} needs real pixel coordinates")
    if value.shape == (*shape[1:], 2) and shape[0] == 1:
        value = value[None]
    if value.shape != (*shape, 2):
        raise ValueError(f"{name} needs one (x,y) correspondence per optical sample")
    return value


def interpolate(xy, rays, protected, origins, reverse):
    """Transport a secondary bearing, centre and optional reverse match together."""
    h, w = rays.shape[:2]
    valid = np.isfinite(xy).all(-1) & (xy >= 0).all(-1) & (xy <= [w - 1, h - 1]).all(-1)
    xy = np.where(valid[..., None], xy, 0.0)
    x, y = np.floor(xy).astype(np.intp).T
    fx, fy = (xy - np.column_stack([x, y])).T
    direction = np.zeros((len(x), 3))
    centre = np.zeros_like(direction)
    back = np.zeros((len(x), 2))
    for dx, dy, weight in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        active = weight > 0
        iy, ix = np.minimum(y + dy, h - 1), np.minimum(x + dx, w - 1)
        usable = ~protected[iy, ix]
        if reverse is not None:
            r = reverse[iy, ix]
            usable &= np.isfinite(r).all(-1)
        valid &= ~active | usable
        use = active & usable
        weight = weight[use, None]
        direction[use] += weight * rays[iy[use], ix[use]]
        if origins is not None:
            centre[use] += weight * origins[iy[use], ix[use]]
        if reverse is not None:
            back[use] += weight * r[use]
    norm = np.linalg.norm(direction, axis=1)
    valid &= norm > 1e-6
    return direction / np.maximum(norm[:, None], 1e-6), centre, back, valid


def triangulate(
    left,
    right,
    matches,
    left_calibration=None,
    right_calibration=None,
    *,
    min_parallax_deg=0.5,
    max_ray_gap_m=0.03,
    max_cycle_px=1.0,
):
    """Return a host packet referenced to the primary exposure; unresolved rays stay protected."""
    limits = [min_parallax_deg, max_ray_gap_m, max_cycle_px]
    if (
        not np.isfinite(limits).all()
        or not 0 < min_parallax_deg < 90
        or min(limits[1:]) < 0
    ):
        raise ValueError(
            "Expected positive parallax below 90 degrees and nonnegative finite tolerances"
        )
    if any(
        k in left
        for k in (
            "host_range",
            "host_z",
            "surface_position",
            "position_valid",
            "occluder_range",
        )
    ):
        raise ValueError("The primary exposure must have a single source of geometry")
    lp, lr, lp_mask, lo = optics(left, left_calibration)
    rp, rr, rp_mask, ro = optics(right, right_calibration)
    shape = lr.shape[:-1]
    if rr.shape[0] not in (1, shape[0]):
        raise ValueError(
            "Use one secondary ray field or one per primary optical sample"
        )
    xy = pixel_map(matches["right_xy"], shape, "right_xy")
    known = sampled(matches["match_valid"], shape, "match_valid")
    if known.dtype != bool:
        raise ValueError("match_valid must be boolean")
    back = (
        pixel_map(matches["left_xy"], rr.shape[:-1], "left_xy")
        if "left_xy" in matches
        else None
    )
    transform = rp[:3, :3].T @ lp[:3, :3]
    baseline = (rp[:3, 3] - lp[:3, 3]) @ lp[:3, :3]
    point = np.zeros_like(lr, dtype=np.float64)
    valid = np.zeros(shape, dtype=bool)
    sin2 = np.sin(np.deg2rad(min_parallax_deg)) ** 2
    for i in range(shape[0]):
        j = min(i, len(rr) - 1)
        y, x = np.nonzero(known[i] & ~lp_mask[i])
        for start in range(0, len(y), 65536):
            row, col = y[start : start + 65536], x[start : start + 65536]
            v, b, reverse_xy, ok = interpolate(
                xy[i, row, col],
                rr[j],
                rp_mask[j],
                None if ro is None else ro[j],
                None if back is None else back[j],
            )
            if back is not None:
                ok &= (
                    np.linalg.norm(reverse_xy - np.column_stack([col, row]), axis=1)
                    <= max_cycle_px
                )
            u = lr[i, row, col].astype(float)
            u /= np.linalg.norm(u, axis=1)[:, None]
            a = 0.0 if lo is None else lo[i, row, col]
            v = v @ transform
            delta = b @ transform + baseline - a
            cos = np.einsum("ij,ij->i", u, v).clip(-1, 1)
            denom = 1 - cos * cos
            du = np.einsum("ij,ij->i", delta, u)
            dv = np.einsum("ij,ij->i", delta, v)
            t = (du - cos * dv) / np.maximum(denom, sin2)
            s = (cos * du - dv) / np.maximum(denom, sin2)
            disp = t[:, None] * u
            gap = np.linalg.norm(disp - delta - s[:, None] * v, axis=1)
            ok &= (denom >= sin2) & (t > 0) & (s > 0) & (gap <= max_ray_gap_m)
            point[i, row, col] = a + disp
            valid[i, row, col] = ok
    packet = dict(
        left, surface_position=point, position_valid=valid, world_from_position=lp
    )
    for name in ("receiver_mask", "occluder_mask"):
        if name in packet:
            mask = sampled(packet[name], shape, name)
            if mask.dtype != bool:
                raise ValueError(f"{name} must label optical samples using booleans")
            packet[name] = mask & valid
    return register_positions(packet, calibration=left_calibration)
