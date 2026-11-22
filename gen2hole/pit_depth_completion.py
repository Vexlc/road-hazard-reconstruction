"""Recover missing range samples from reciprocal RGB correspondences at externally registered cameras."""

from dataclasses import replace
from itertools import combinations

import cv2
import numpy as np

from photo_parallax import Photo, match_in_image


def crop_view(view):
    y, x = np.where(view.mask)
    pad = 20
    x0, y0 = max(0, x.min() - pad), max(0, y.min() - pad)
    region = np.s_[y0 : y.max() + pad + 1, x0 : x.max() + pad + 1]
    k = view.K.copy()
    k[:2, 2] -= [x0, y0]
    photo = Photo(view.key, view.rgb[region], view.mask[region], k, view.sources)
    return photo, view.depth[region], region


def match_pair(a, b, crops, poses, center, up, span):
    matches = {}
    for first, second in [(a, b), (b, a)]:
        photo, _, _ = crops[first]
        pose = np.linalg.solve(poses[second], poses[first])
        normal = -up @ poses[first][:3, :3]
        distance = float((center - poses[first][:3, 3]) @ -up)
        if distance <= 0:
            raise ValueError('The camera must lie above the fitted road')
        h, good, uv, rays = match_in_image(photo, crops[second][0], pose, normal, distance, span)
        z = (distance - h) / np.maximum(rays @ normal, 1e-6)
        paired_z = (rays @ pose[2, :3]) * z + pose[2, 3]
        matches[first] = (h, good, uv, z, paired_z)
    for first, second in [(a, b), (b, a)]:
        h, good, uv, z, paired_z = matches[first]
        back_h, back_good, back_uv, *_ = matches[second]
        photo, depth, _ = crops[first]
        other, depth2, _ = crops[second]

        def remap(values):
            return cv2.remap(values, uv, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        y, x = np.indices(h.shape)
        loop = remap(back_uv)
        valid = good & (depth == 0) & photo.mask & (remap((back_good & other.mask).astype('u1')) > 0)
        valid &= (np.hypot(loop[:, :, 0] - x, loop[:, :, 1] - y) <= 2.0) & (abs(remap(back_h) - h) < 0.008)
        observed = remap(depth2)
        valid &= ((observed == 0) | (abs(paired_z - observed) <= 0.010)) & (z > 0.15) & (z < 15)
        yield first, np.where(valid, z, np.nan)


def recover_depth(views, poses, center, up, span):
    if len(views) < 2:
        raise ValueError('Photographic depth completion needs at least two fitting views')
    crops = {v.key: crop_view(v) for v in views}
    candidates = {v.key: [] for v in views}
    for a, b in combinations(crops, 2):
        for key, result in match_pair(a, b, crops, poses, center, up, span):
            candidates[key].append(result)
    enhanced, records = [], {}
    for view in views:
        _, _, region = crops[view.key]
        values = np.stack(candidates[view.key])
        finite = np.isfinite(values)
        number = finite.sum(0)
        low = np.where(finite, values, np.inf).min(0)
        high = np.where(finite, values, -np.inf).max(0)
        use = (number > 0) & (high - low < 0.010)
        repaired = view.depth.copy()
        repaired[region][use] = np.nansum(values, axis=0)[use] / number[use]
        enhanced.append(replace(view, depth=repaired))
        records[view.key] = dict(
            missing_target_pixels=int(((view.depth == 0) & view.mask).sum()),
            recovered_target_pixels=int(use.sum()),
            target_pixels=int(view.mask.sum()),
        )
    return enhanced, records
