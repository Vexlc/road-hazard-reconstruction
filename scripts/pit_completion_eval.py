"""Audit natural range dropouts, reciprocal photographic samples and paired holdout errors."""

import numpy as np
from PIL import Image

from pit_captures import transform
from pit_depth_completion import recover_depth


def source_regions(views, poses, frame, settings, field, folder):
    center = np.array(frame['local_origin_in_anchor_m'])
    up = np.asarray(frame['local_axes_in_anchor_camera'])[:, 2]
    repaired, stats = recover_depth(views, poses, center, up, settings['search_depth_m'])
    if stats != frame['depth_completion']:
        raise ValueError('Source completion counts changed after build')
    for original, filled in zip(views, repaired):
        native = original.depth > 0
        if not np.array_equal(original.depth[native], filled.depth[native]):
            raise ValueError('A native depth sample was overwritten')
        absent = original.mask & ~native
        added = absent & (filled.depth > 0)
        if ((~original.mask & ~native) & (filled.depth > 0)).any():
            raise ValueError('Photographic samples outside the marked pit')
        y, x = np.nonzero(original.mask)
        lo = np.maximum([x.min() - 16, y.min() - 16], 0)
        hi = np.minimum([x.max() + 17, y.max() + 17], original.rgb.shape[1::-1])
        box = (*lo, *hi)
        im = original.rgb.copy()
        im[absent] = [230, 145, 68]
        Image.fromarray(im).crop(box).save(folder / (original.key + '-missing.png'))
        im[added] = [59, 193, 188]
        Image.fromarray(im).crop(box).save(folder / (original.key + '-recovered.png'))
    regions = field['support_kind'].astype(int)
    mask = field['defect_mask'] > 0.5
    native = field['view_count'] > 0
    if not np.array_equal(native, regions == 1) or np.any(regions > 2):
        raise ValueError('Raw and photograph-only support must be disjoint')
    palette = np.array([[230, 145, 68], [67, 91, 112], [59, 193, 188]], dtype='u1')
    panel = palette[regions]
    panel[~mask] = [220, 220, 210]
    y, x = np.nonzero(mask)
    Image.fromarray(panel).crop((x.min() - 5, y.min() - 5, x.max() + 6, y.max() + 6)).save(
        folder / 'range-coverage.png'
    )


def paired_errors(view, pose, frame, field, repaired, native, crop, folder):
    """Choose bins using the measured held-out XYZ, independent of either predicted depth."""
    y, x = np.nonzero(view.mask & (view.depth > 0))
    xyz = transform(view.xyz(np.c_[x, y], view.depth[y, x]), pose)
    center = np.asarray(frame['local_origin_in_anchor_m'])
    local = (xyz - center) @ frame['local_axes_in_anchor_camera']
    kinds = field['support_kind']
    xy = np.rint((0.5 + local[:, :2] / [field['width'], -field['width']]) * (len(kinds) - 1)).astype(int)
    inside = ((xy >= 0) & (xy < len(kinds))).all(1)
    i, j = xy[inside].T
    selection = np.zeros(len(x), bool)
    selection[inside] = field['defect_mask'][j, i] > 0.5
    category = np.full(len(x), -1)
    category[np.flatnonzero(inside)] = kinds[j, i]
    both = np.isfinite(repaired[y, x]) & np.isfinite(native[y, x])
    keys = {'photographic': 2, 'native': 1, 'interpolated': 0}
    report = {}
    overlay = (view.rgb.astype(float) * 0.66).astype('u1')
    for key, value in keys.items():
        hit = selection & (category == value)
        use = hit & both
        item = dict(measured_target_pixels=int(hit.sum()), paired_rays=int(use.sum()))
        if use.any():
            rows, cols = y[use], x[use]
            actual = view.depth[rows, cols]
            a, b = [abs(z[rows, cols] - actual) * 1000 for z in (native, repaired)]
            delta = a - b
            for name, error in [('range_only', a), ('completed', b)]:
                item[name] = dict(
                    median_mm=float(np.median(error)),
                    mean_mm=float(error.mean()),
                    p95_mm=float(np.percentile(error, 95)),
                )
            item.update(improved_over_1mm=int((delta > 1).sum()), worsened_over_1mm=int((delta < -1).sum()))
            if key == 'photographic':
                overlay[rows, cols] = [216, 206, 163]
                overlay[rows[delta > 1], cols[delta > 1]] = [61, 207, 177]
                overlay[rows[delta < -1], cols[delta < -1]] = [231, 102, 80]
        report[key] = item
    Image.fromarray(overlay[crop]).save(folder / 'completion-errors.png')
    return report
