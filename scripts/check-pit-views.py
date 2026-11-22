"""Ray-test the released pothole GLBs against labelled interiors withheld from reconstruction."""

import argparse
import json
from itertools import zip_longest
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gen2hole'), str(ROOT / 'gen2hole/downstream/sensors')]
from fuse_pit import fuse
from pit_captures import read_views
from road_geometry import Surface, glb_triangles, sha256
from pit_projection import SingleView, camera_image
from pit_texture_eval import compare, read_texture
from pit_completion_eval import source_regions, paired_errors


def summary(view, depth, model):
    reference = view.mask
    intersection = model & reference
    valid = intersection & (view.depth > 0) & np.isfinite(depth)
    error = np.abs(depth[valid] - view.depth[valid]) * 1000
    blobs, n = ndimage.label(reference)
    rec = []
    for i in range(1, n + 1):
        region = blobs == i
        if region.sum() >= 64:
            rec.append(float(model[region].mean()))
    return dict(
        mask_iou=float(intersection.sum() / (model | reference).sum()),
        mask_recall=float(intersection.sum() / reference.sum()),
        mask_precision=float(intersection.sum() / max(1, model.sum())),
        reference_component_recall=sorted(rec),
        visible_pit_depth_samples=len(error),
        depth_abs_median_mm=float(np.median(error)) if len(error) else None,
        depth_abs_p95_mm=float(np.percentile(error, 95)) if len(error) else None,
        annotated_target_touches_image_border=bool(
            np.any(reference[[0, -1]]) or np.any(reference[:, [0, -1]])
        ),
    )


def components(mask, x, y):
    labels, n = ndimage.label(mask)
    result = []
    for i in range(1, n + 1):
        hit = labels == i
        if hit.sum() >= max(12, 0.004 * mask.sum()):
            result.append([float(x[hit].mean()), float(y[hit].mean())])
    return sorted(result)


def separation(points):
    return float(np.linalg.norm(np.subtract(*points))) if len(points) == 2 else None


def model_centers(product, field):
    height = field['height']
    w = float(field['width'])
    axis = np.linspace(-w / 2, w / 2, len(height))
    x, y = np.meshgrid(axis, -axis)
    origin = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
    _, part, _ = product.cast(origin, np.broadcast_to([0.0, 0.0, -1.0], origin.shape))
    return components(part.reshape(x.shape) == 2, x, y)


def measured_centers(view, pose, frame, pitch):
    labels, n = ndimage.label(view.mask)
    basis = np.array(frame['local_axes_in_anchor_camera'])
    center = np.array(frame['local_origin_in_anchor_m'])
    result = []
    for i in range(1, n + 1):
        mask = labels == i
        if mask.sum() < max(64, 0.004 * view.mask.sum()) or mask[[0, -1]].any() or mask[:, [0, -1]].any():
            continue
        y, x = np.where(mask & (view.depth > 0))
        if not len(x):
            continue
        xyz = view.xyz(np.c_[x, y], view.depth[y, x])
        points = (xyz @ pose[:3, :3].T + pose[:3, 3] - center) @ basis
        pixels = np.unique(np.rint(points[:, :2] / pitch).astype(int), axis=0)
        result.append((pixels.mean(0) * pitch).tolist())
    return sorted(result)


def inputs_panel(folder, views, field):
    folder.mkdir(parents=True, exist_ok=True)
    for view in views:
        im = Image.fromarray(view.rgb)
        im.thumbnail((960, 720))
        im.save(folder / (view.key + '.jpg'), quality=94)
    count = np.array(field['view_count'])
    palette = np.uint8([[52, 57, 56], [223, 181, 101], [65, 168, 155], [122, 122, 205]])
    image = palette[np.minimum(count.astype(int), 3)]
    contours, _ = cv2.findContours(
        field['defect_mask'].astype('u1'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (245, 245, 233), 1, cv2.LINE_AA)
    Image.fromarray(image).save(folder / 'support.png')


def overlays(view, depth, model, folder):
    masks = [view.mask, model]
    yy, xx = np.nonzero(view.mask | model)
    h, w = view.depth.shape
    pad = max(22, int(max(np.ptp(xx), np.ptp(yy)) * 0.1))
    crop = np.s_[
        max(0, yy.min() - pad) : min(h, yy.max() + pad + 1),
        max(0, xx.min() - pad) : min(w, xx.max() + pad + 1),
    ]
    image = view.rgb.copy()
    Image.fromarray(image[crop]).save(folder / 'photo.jpg', quality=95)
    for mask, color in zip(masks, [(255, 215, 67), (62, 229, 207)]):
        contours, _ = cv2.findContours(mask.astype('u1'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, max(1, round(w / 600)), cv2.LINE_AA)
    Image.fromarray(image[crop]).save(folder / 'contours.jpg', quality=95)
    valid = view.mask & model & (view.depth > 0)
    err = np.zeros(depth.shape)
    err[valid] = np.clip(abs(depth[valid] - view.depth[valid]) / 0.035, 0, 1)
    heat = cv2.applyColorMap(np.uint8(err * 255), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    background = np.rint(0.38 * view.rgb + 40).astype('u1')
    background[valid] = heat[valid]
    Image.fromarray(background[crop]).save(folder / 'errors.jpg', quality=95)
    return crop


def run(args):
    bundle = args.run / 'series'
    specs = json.loads((bundle / 'calibration/survey.json').read_text())
    reports = {}
    for spec in specs:
        if spec.get('mode')=='rgb_parallax':
            continue
        key = spec['id']
        capture = json.loads((bundle / 'calibration' / (key + '_capture.json')).read_text())
        views = read_views(spec, args.source, args.catalog, args.files_base or bundle / 'calibration')
        if capture['fit'] != spec['fit'] or set(capture['fit']) & set(spec['holdout']):
            raise ValueError('Incorrect split for ' + key)
        for k, v in views.items():
            if v.sources != capture['inputs'][k]['files']:
                raise ValueError('Input changed for ' + k)
        poses = {k: np.array(p) for k, p in capture['poses_view_to_anchor'].items()}
        glb = bundle / 'assets/pothole' / key / 'asset.glb'
        product = Surface(glb)
        with np.load(glb.parent / 'surface.npz') as field:
            width = float(field['width'])
            texture = read_texture(glb, width)
            centers = model_centers(product, field)
            inputs_panel(args.run / 'evaluation' / key / 'input', [views[k] for k in spec['fit']], field)
            if 'depth_completion' in spec:
                support = {k:field[k] for k in ['support_kind','defect_mask','width']}
                source_regions([views[k] for k in spec['fit']], poses, capture['geometry'],
                               spec['depth_completion'], field, args.run/'evaluation'/key/'input')
                comparison = bundle / 'comparison' / key
                baseline = comparison / 'range-only.glb'
                diagnostic = comparison / 'support.glb'
                range_only = Surface(baseline)
                if not np.array_equal(read_texture(baseline, width), texture):
                    raise ValueError('The range-only comparison must share the published texture')
                for left, right in zip_longest(glb_triangles(glb), glb_triangles(diagnostic)):
                    if left is None or right is None or any(not np.array_equal(a, b) for a, b in zip(left, right)):
                        raise ValueError('Height-source mesh differs from the published surface')
                colors = np.uint8([[230, 145, 68], [67, 91, 112], [59, 193, 188]])
                expected = colors[support['support_kind'].astype(int)]
                expected[support['defect_mask'] < 0.5] = [220, 220, 210]
                if not np.array_equal(read_texture(diagnostic, width), expected):
                    raise ValueError('Height-source mesh texture disagrees with the measured support')

        controls = {}
        for k in spec['fit']:
            field, _, frame = fuse([views[k]], poses)
            controls[k] = SingleView(field), frame
        rows = {}
        for k in spec['holdout']:
            v = views[k]
            depth, target = camera_image(v, poses[k], product, capture['geometry'])
            stats = summary(v, depth, target)
            if stats['visible_pit_depth_samples'] < 40:
                raise ValueError('No resolved overlap with the withheld target in ' + key + ' ' + k)
            folder = args.run / 'evaluation' / key / k
            folder.mkdir(parents=True, exist_ok=True)
            crop = overlays(v, depth, target, folder)
            photo_check = compare(
                v,
                poses[k],
                capture['geometry'],
                width,
                texture,
                depth,
                target,
                crop,
                folder / 'base-color.jpg',
            )
            # Validate the single-frame control renderer against the actual delivered GLB.
            with np.load(glb.parent / 'surface.npz') as field:
                control_depth, control_mask = camera_image(
                    v, poses[k], SingleView(field), capture['geometry']
                )
            same = (target & control_mask).sum() / max(1, (target | control_mask).sum())
            if (
                same < 0.999
                or np.median(abs(depth[target & control_mask] - control_depth[target & control_mask]))
                > 0.0001
            ):
                raise ValueError('Control rendering disagrees with exported geometry for ' + key)

            single = {}
            for src, (mesh, local) in controls.items():
                z, mask = camera_image(v, poses[k], mesh, local)
                single[src] = summary(v, z, mask)
                image = v.rgb.copy()
                for contour_mask, color in ((v.mask, (255, 215, 67)), (mask, (62, 229, 207))):
                    contour, _ = cv2.findContours(
                        contour_mask.astype('u1'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(
                        image, contour, -1, color, max(1, round(image.shape[1] / 600)), cv2.LINE_AA
                    )
                Image.fromarray(image[crop]).save(folder / ('single-' + src + '.jpg'), quality=95)
            observed_centers = measured_centers(
                v, poses[k], capture['geometry'], capture['geometry']['grid_pitch_m']
            )
            rows[k] = dict(
                exported_glb=stats,
                material_reprojection=photo_check,
                single_fit_frame_controls=single,
                control_renderer_glb_iou=float(same),
                heldout_centers_local_m=observed_centers,
                heldout_center_spacing_m=separation(observed_centers),
            )
            if 'depth_completion' in spec:
                base, _ = camera_image(v, poses[k], range_only, capture['geometry'])
                rows[k]['paired_completion'] = paired_errors(v, poses[k], capture['geometry'], support,
                                                             depth, base, crop, folder)
        reports[key] = dict(
            glb_sha256=sha256(glb),
            source=spec['source'],
            fit=spec['fit'],
            anchor=spec['anchor'],
            component_count=len(centers),
            glb_centers_local_m=centers,
            glb_center_spacing_m=separation(centers),
            heldout=rows,
        )
        if 'depth_completion' in spec:
            reports[key]['comparison_glb_sha256'] = {p.name: sha256(p) for p in (baseline, diagnostic)}
        print(key, k, stats, flush=True)
    args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--source', type=Path, default=ROOT / 'work/reconstruction-inputs')
    p.add_argument('--catalog', type=Path, default=ROOT / 'catalog/capture-inputs.json')
    p.add_argument('--files-base', type=Path, help='Original survey folder for custom file paths')
    run(p.parse_args())
