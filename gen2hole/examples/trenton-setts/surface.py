"""Two-view plane sweep on exposed setts, with an independent set of sparse checks."""

from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter
from cameras import HERE, S, photograph
from stereo import pose, triangulate


def survey_coords(p):
    return (p - S["annotation_origin_px"]) / S["native_px_per_annotation_px"]


def native(p):
    return np.asarray(p) * S["native_px_per_annotation_px"] + S["annotation_origin_px"]


def labels(photo_uv):
    low = np.array(S["crop_bounds"][:2])
    high = np.array(S["crop_bounds"][2:])
    q = survey_coords(photo_uv) - low
    shape = tuple((high - low + 1)[::-1])
    canvas = np.ones(shape, "u1")
    for label, ring in enumerate(S["regions"].values(), 2):
        cv2.fillPoly(canvas, [np.int32(np.array(ring) - low)], label)
    ix = np.rint(q[..., 0]).astype(int)
    iy = np.rint(q[..., 1]).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < shape[1]) & (iy < shape[0])
    return np.where(inside, canvas[iy.clip(0, shape[0] - 1), ix.clip(0, shape[1] - 1)], 0)


def points_on_grade(c, ray, grade, dz=0):
    intercept = grade[2] + grade[:2] @ c.eye[:2]
    divisor = ray[..., 2] - ray[..., :2] @ grade[:2]
    return c.eye + ray * ((intercept - c.eye[2] + dz) / divisor)[..., None]


def read_pairs():
    with np.load(HERE / "correspondences.npz") as f:
        return f["pairs"].astype(float)


def road_plane(pairs, c0, reg):
    points = reg["xyz"]
    uv = survey_coords(pairs[:, :2])
    # Only native asphalt ties in and immediately beside the mapped damage. Exclude the casting.
    local = (uv[:, 0] > -300) & (uv[:, 0] < 1460) & (uv[:, 1] > -210) & (uv[:, 1] < 435)
    asphalt = labels(pairs[:, :2]) <= 1
    train = reg["train"]
    use = local & asphalt & (reg["errors"] < 2.4) & (np.linalg.norm(points[:, :2], axis=1) > 0.59)
    use &= np.isin(np.arange(len(pairs)), train)
    A = np.column_stack([points[:, :2], np.ones(len(points))])
    if use.sum() < 12:
        raise ValueError("Insufficient local grade ties")
    # Refit at centred residual thresholds so one repair or isolated low block cannot set grade.
    for tolerance in [0.047, 0.024, 0.015]:
        g = np.linalg.lstsq(A[use], points[use, 2], rcond=None)[0]
        err = points[:, 2] - A @ g
        use &= abs(err - np.median(err[use])) < tolerance
    return np.linalg.lstsq(A[use], points[use, 2], rcond=None)[0], use


def sample(im, uv):
    return cv2.remap(
        im,
        uv[..., 0].astype("f4"),
        uv[..., 1].astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def zncc(ref, other, blur, mu, var):
    mean = blur(other)
    covariance = blur(ref * other) - mu * mean
    return np.clip(
        covariance / np.sqrt(np.maximum(var * (blur(other * other) - mean**2), 1.0)), -1, 1
    )


def reconstruct(pitch=0.004):
    pairs = read_pairs()
    c0, c1, reg = pose(pairs)
    grade, control = road_plane(pairs, c0, reg)
    u, l, r, b = S["crop_bounds"]
    bounds = points_on_grade(c0, c0.ray(native([[u, l], [r, l], [r, b], [u, b]])), grade)
    lo = np.floor((bounds[:, :2].min(0) - 0.015) / pitch) * pitch
    hi = np.ceil((bounds[:, :2].max(0) + 0.015) / pitch) * pitch
    n = np.rint((hi - lo) / pitch).astype(int) + 1
    yy, xx = np.indices(tuple(n[::-1]))
    xy = np.stack([lo[0] + xx * pitch, hi[1] - yy * pitch], axis=-1)
    world = np.concatenate([xy, (xy @ grade[:2] + grade[2])[..., None]], axis=-1)
    uv = c0.project(world)
    ray = c0.ray(uv)
    lab = labels(uv)
    ims = [photograph(n) for n in ("detail", "street")]
    grey = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype("f4") for im in ims]

    def sweep(first, second, first_grey, second_grey):
        pix = first.project(world)
        rays = first.ray(pix)
        ref = sample(first_grey, pix)

        def blur(a):
            return cv2.GaussianBlur(a, (0, 0), 0.016 / pitch)

        mu = blur(ref)
        var = np.maximum(blur(ref * ref) - mu**2, 1.0)
        levels = np.arange(-0.073, 0.054, 0.0015)
        costs = []
        for level in levels:
            q = second.project(points_on_grade(first, rays, grade, level))
            costs.append(zncc(ref, sample(second_grey, q), blur, mu, var))
        scores = np.stack(costs)
        best = scores.argmax(0)
        peak = np.take_along_axis(scores, best[None], axis=0)[0]
        rival = np.max(
            np.where(abs(np.arange(len(levels))[:, None, None] - best) > 7, scores, -1), axis=0
        )
        sharp = peak - rival
        z = levels[best]
        valid = (peak > 0.64) & (sharp > 0.050) & (var > 16) & (best > 1) & (best < len(levels) - 2)
        valid &= abs(z - median_filter(z, size=5)) < 0.0065
        return z, peak, sharp, valid

    z, peak, sharp, reliable = sweep(c0, c1, grey[0], grey[1])
    reverse, _, _, reverse_ok = sweep(c1, c0, grey[1], grey[0])
    candidate = points_on_grade(c0, ray, grade, z)
    at_grade = points_on_grade(c1, c1.ray(c1.project(candidate)), grade)
    reverse_grid = (at_grade[..., :2] - [lo[0], hi[1]]) / [pitch, -pitch]
    reverse_z = sample(reverse.astype("f4"), reverse_grid)
    cycle = abs(z - reverse_z)
    reliable &= (
        (lab > 0) & (cycle < 0.0045) & (sample(reverse_ok.astype("f4"), reverse_grid) > 0.66)
    )
    # Repetitive joints can agree in both directions at an incorrect neighbouring
    # stone. Require nearby sparse, independently triangulated training coordinates.
    # Reserved pairs never enter these support checks or the grade fit.
    train = reg["train"]
    train = train[(reg["errors"][train] < 2.5) & (pairs[train, 4] > 0.24)]
    anchors = reg["xyz"][train]
    for label in (1, 2, 3):
        target = lab == label
        anchor_labels = labels(pairs[train, :2])
        keep = anchor_labels == label if label != 1 else anchor_labels <= 1
        controls = anchors[keep]
        if len(controls) < 3:
            raise ValueError(("Insufficient stereo controls", label))
        ds, ii = cKDTree(controls[:, :2]).query(candidate[target, :2], k=min(4, len(controls)))
        az = controls[:, 2] - controls[:, :2] @ grade[:2] - grade[2]
        agrees = (ds < 0.24) & (abs(az[ii] - z[target, None]) < 0.012)
        reliable[target] &= np.any(agrees, axis=1)
    # Estimate the slowly varying height of adjoining intact pavement. This accounts
    # for crossfall and for changes in level from patch to patch along the intact edge.
    road = reliable & (lab == 1)
    wt = gaussian_filter(road.astype(float), 0.115 / pitch)
    sm = gaussian_filter(np.where(road, z, 0.0), 0.115 / pitch) / np.maximum(wt, 1e-8)
    valid = wt > 0.025
    if not valid.any():
        raise ValueError("Road height not supported")
    sm = sm[tuple(distance_transform_edt(~valid, return_distances=False, return_indices=True))]
    # Preserve absolute source-ray height separately from the flattened relief used on a host.
    relief = np.zeros_like(z)
    source_height = sm.copy()
    for label in (2, 3):
        inside = lab == label
        support = reliable & inside
        if support.sum() < 130:
            raise ValueError(("Weak stereo support", label, support.sum()))
        distance, idx = distance_transform_edt(~support, return_indices=True)
        fill = z[tuple(idx)]
        # Smooth nearby observations in metric space; taper uncertain relief
        # back to adjoining grade beyond the last supported joint or stone.
        num = gaussian_filter(np.where(support, z, 0.0), 0.018 / pitch)
        den = gaussian_filter(support.astype(float), 0.018 / pitch)
        fill = np.where(
            den > 0.025, num / np.maximum(den, 1e-8), gaussian_filter(fill, 0.030 / pitch)
        )
        t = np.clip((0.18 - distance * pitch) / 0.12, 0, 1)
        filled = sm + t * t * (3 - 2 * t) * (fill - sm)
        source_height[inside] = filled[inside]
        relief[inside] = (filled - sm)[inside]
    w = points_on_grade(c0, ray, grade, source_height)
    return dict(
        pitch=pitch,
        xy=xy,
        world=w,
        uv=uv,
        labels=lab,
        grade=grade,
        road_dz=sm,
        source_dz=source_height,
        relief=relief,
        raw_dz=z,
        stereo_valid=reliable,
        cycle_error_m=cycle,
        peak_score=peak,
        peak_margin=sharp,
        controls=control,
        cameras=(c0, c1),
        fit=reg,
        pairs=pairs,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    h = reconstruct()
    np.savez_compressed(
        a.output, **{k: v for k, v in h.items() if k not in ("cameras", "fit", "pairs")}
    )
