"""Hide an entire road-space patch in every fitting depth image and test against the original ranges."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole'))
from fuse_pit import fuse
from pit_captures import read_views, transform, project

from pit_projection import SingleView, camera_image


def patch_box(field):
    mask = field['defect_mask'] > 0.5
    pitch = float(field['width']) / (len(mask) - 1)
    distance = distance_transform_edt(mask)
    i, j = np.unravel_index(distance.argmax(), mask.shape)
    center = (np.array([j, i]) / (len(mask) - 1) - 0.5) * [field['width'], -field['width']]
    half = distance[i, j] * pitch * 0.55
    return center, half


def hide(views, center, half, record, poses, span):
    basis = np.asarray(record['local_axes_in_anchor_camera'])
    origin = np.asarray(record['local_origin_in_anchor_m'])
    corners = [
        [center[0] + half * x, center[1] + half * y, z]
        for x in [-1, 1]
        for y in [-1, 1]
        for z in [-span, 0.04]
    ]
    world = np.asarray(corners) @ basis.T + origin
    masked, masks = [], {}
    for view in views:
        camera = transform(world, np.linalg.inv(poses[view.key]))
        if (camera[:, 2] <= 0).any():
            raise ValueError('Test volume crosses camera plane')
        contour = cv2.convexHull(project(camera, view.K).astype('f4'))
        mask = np.zeros_like(view.mask, 'u1')
        cv2.fillConvexPoly(mask, np.rint(contour[:, 0]).astype('i4'), 1)
        mask = cv2.dilate(mask, np.ones((5, 5), 'u1')).astype(bool)
        masked.append(replace(view, depth=np.where(mask, 0, view.depth)))
        masks[view.key] = mask
    return masked, masks


def benchmark(run, spec, data, catalog, files_base=None):
    bundle = run / 'series'
    key = spec['id']
    settings = spec['depth_completion']
    source = json.loads((bundle / 'calibration' / (key + '_capture.json')).read_text())
    observed = read_views(spec, data, catalog, files_base or bundle / 'calibration')
    fitting = [observed[k] for k in spec['fit']]
    poses = {k: np.array(m) for k, m in source['poses_view_to_anchor'].items() if k in spec['fit']}
    record = source['geometry']
    with np.load(bundle / 'assets/pothole' / key / 'surface.npz') as field:
        center, half = patch_box(field)
    masked, masks = hide(fitting, center, half, record, poses, settings['search_depth_m'])
    fine = dict(spacing=settings['grid_pitch_m'], reach_cells=settings['support_radius_cells'])
    outputs = [fuse(masked, poses, completion=mode, **fine) for mode in (None, settings)]
    models = [(SingleView(field), frame) for field, _, frame in outputs]
    if not np.array_equal(outputs[0][0]['view_count'], outputs[1][0]['view_count']):
        raise ValueError('Blind controls used different native ranges')
    result = dict(patch_center_xy_m=center.tolist(), patch_width_m=float(2 * half), frames={})
    folder = run / 'evaluation' / key / 'controlled'
    folder.mkdir(exist_ok=True)
    for view in fitting:
        y, x = np.where(view.mask & (view.depth > 0))
        original = view.depth[y, x]
        points = transform(view.xyz(np.c_[x, y], original), poses[view.key])
        local = (points - record['local_origin_in_anchor_m']) @ record['local_axes_in_anchor_camera']
        core = (abs(local[:, :2] - center) < half - fine['spacing'] * 2).all(1)
        core &= (local[:, 2] > -settings['search_depth_m']) & (local[:, 2] < 0.04)
        if not masks[view.key][y[core], x[core]].all():
            raise ValueError('A withheld target remained in the fitting ranges')
        z = [camera_image(view, poses[view.key], mesh, frame)[0][y, x] for mesh, frame in models]
        grid, _, frame = outputs[1]
        target = (points - frame['local_origin_in_anchor_m']) @ frame['local_axes_in_anchor_camera']
        xy = np.rint(
            (target[:, :2] / [grid['width'], -grid['width']] + 0.5) * (len(grid['height']) - 1)
        ).astype(int)
        inside = ((xy >= 0) & (xy < len(grid['height']))).all(1)
        i, j = xy[inside].T
        source_type = np.full(len(x), -1)
        source_type[inside] = grid['support_kind'][j, i]
        valid = core & np.isfinite(z[0]) & np.isfinite(z[1])
        sample = valid & (source_type == 2)
        measured = core & (source_type == 1)
        if measured.any():
            raise ValueError('Native range leaked into the held-out patch core')
        if not valid.any():
            result['frames'][view.key] = dict(hidden_direct_pixels=int(core.sum()), paired_rays=0)
            continue
        case = dict(
            hidden_direct_pixels=int(core.sum()),
            paired_rays=int(valid.sum()),
            photo_supported=int(sample.sum()),
        )
        im = (view.rgb.astype(float) * 0.6).astype('u1')
        im[y[valid], x[valid]] = [232, 159, 79]
        im[y[sample], x[sample]] = [56, 195, 175]
        ys, xs = np.where(view.mask)
        lo = np.maximum([xs.min() - 14, ys.min() - 14], 0)
        hi = np.minimum([xs.max() + 15, ys.max() + 15], view.rgb.shape[1::-1])
        Image.fromarray(im).crop((*lo, *hi)).save(folder / (view.key + '.png'))
        for title, region in [('whole_patch', valid), ('photographic', sample)]:
            if not region.any():
                continue
            case[title] = {}
            for name, depths in zip(('range_only', 'completed'), z):
                errors = abs(depths[region] - original[region]) * 1000
                case[title][name] = dict(
                    median_mm=float(np.median(errors)),
                    mean_mm=float(errors.mean()),
                    p95_mm=float(np.percentile(errors, 95)),
                )
        result['frames'][view.key] = case
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', type=Path, help='Defaults to evaluation/blind-patches.json in the run')
    parser.add_argument('--source', type=Path, default=ROOT / 'work/reconstruction-inputs')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'catalog/capture-inputs.json')
    parser.add_argument('--files-base', type=Path, help='Original survey folder for custom file paths')
    args = parser.parse_args()
    cv2.setNumThreads(3)
    specs = json.loads((args.run / 'series/calibration/survey.json').read_text())
    report = {
        s['id']: benchmark(args.run, s, args.source, args.catalog, args.files_base)
        for s in specs
        if 'depth_completion' in s
    }
    output = args.out or args.run / 'evaluation/blind-patches.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
