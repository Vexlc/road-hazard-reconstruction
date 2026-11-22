"""Ray-test the released pothole GLBs against labelled interiors withheld from reconstruction."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gen2hole'), str(ROOT / 'gen2hole/downstream/sensors')]
from fuse_pit import fuse
from pit_captures import read_views
from road_geometry import Surface, sha256
from pit_texture_eval import compare, read_texture


class SingleView:
    """One fitting frame rebuilt with the same surface estimator and mask-to-triangle rule."""

    def __init__(self, field):
        z = field['height']
        width = field['width']
        x = np.linspace(-width / 2, width / 2, len(z))
        xx, yy = np.meshgrid(x, -x)
        points = np.dstack((xx, yy, z)).reshape(-1, 3).astype('f4')
        ids = np.arange(z.size, dtype='u4').reshape(z.shape)
        a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
        tri = np.stack([np.stack([a, c, b], -1), np.stack([c, d, b], -1)], -2).reshape(-1, 3)
        is_pit = field['defect_mask'].ravel()[tri].mean(1) > 0.5
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.scene.add_triangles(o3d.core.Tensor(points), o3d.core.Tensor(tri))
        self.is_pit = is_pit

    def cast(self, eye, rays):
        result = self.scene.cast_rays(
            o3d.core.Tensor(np.ascontiguousarray(np.column_stack((eye, rays)), dtype='f4')), nthreads=2
        )
        depth = result['t_hit'].numpy()
        hits = np.isfinite(depth)
        parts = np.zeros(len(hits), 'u1')
        parts[hits] = 1 + self.is_pit[result['primitive_ids'].numpy()[hits]]
        return depth, parts, None


def camera_image(view, pose, surface, frame):
    basis = np.array(frame['local_axes_in_anchor_camera'])
    center = np.array(frame['local_origin_in_anchor_m'])
    eye = (pose[:3, 3] - center) @ basis
    rows, cols = view.mask.shape
    depth = np.empty((rows, cols), 'f4')
    label = np.zeros((rows, cols), bool)
    projection = np.linalg.inv(view.K).T @ pose[:3, :3].T @ basis
    for top in range(0, rows, 64):
        bottom = min(top + 64, rows)
        y, x = np.mgrid[top:bottom, 0:cols]
        directions = np.c_[x.ravel(), y.ravel(), np.ones(x.size)] @ projection
        dist, part, _ = surface.cast(np.broadcast_to(eye, directions.shape), directions)
        depth[top:bottom] = dist.reshape(x.shape)
        label[top:bottom] = part.reshape(x.shape) == 2
    return depth, label


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
