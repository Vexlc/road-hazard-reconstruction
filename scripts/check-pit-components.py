"""Reproject each connected opening of the released GLB into independently annotated RGB-D frames."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gen2hole'), str(ROOT / 'gen2hole/downstream/sensors')]
from pit_captures import read_views, transform
from pit_projection import camera_image
from road_geometry import Surface, glb_triangles, read_json, save_json, sha256


def outline(image, mask, rgb):
    lines, _ = cv2.findContours(mask.astype('u1'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, lines, -1, rgb, 2, cv2.LINE_AA)


def mark(rgb, areas, names, target=None):
    img = rgb.copy()
    if target is not None:
        outline(img, target, (255, 212, 80))
    outline(img, areas > 0, (74, 235, 213))
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    for i, name in enumerate(names, 1):
        y, x = np.where(areas == i)
        if not len(x):
            continue
        label = (max(2, min(img.width - 54, int(np.median(x)) - 23)), max(2, int(y.min()) - 21))
        draw.text(label, name, font=font, fill='white', stroke_width=2, stroke_fill='#172b28')
    return img


def audit(run, key, output):
    folder = run / 'evaluation' / key / 'components'
    record = read_json(folder / 'index.json')
    cal = run / 'series/calibration'
    capture = read_json(cal / (key + '_capture.json'))
    spec = capture['spec']
    views = read_views(spec, ROOT / 'work/reconstruction-inputs', ROOT / 'catalog/capture-inputs.json', cal)
    glb = run / 'series/assets/pothole' / key / 'asset.glb'
    if sha256(glb) != record['glb_sha256']:
        raise ValueError('Component index references a different parent mesh')
    parts = record['components']
    names = list(parts)
    centers = np.array([p['center_local_xy_m'] for p in parts.values()])
    tree = cKDTree(centers)
    # Verify that nearest-center IDs unambiguously name every vertex of every bed triangle.
    for part, xyz, faces in glb_triangles(glb):
        if part == 2:
            owner = tree.query(xyz[:, :2])[1][faces]
            counts = np.bincount(owner[:, 0], minlength=len(parts))
            if np.any(owner != owner[:, :1]) or counts.tolist() != [
                p['bed_triangles'] for p in parts.values()
            ]:
                raise ValueError('Voronoi ownership disagrees with connected triangle groups')
    for name, p in parts.items():
        if 'file' in p and sha256(folder / p['file']) != p['sha256']:
            raise ValueError('Component GLB changed: ' + name)
    (fit,) = spec['fit']
    label, _ = ndimage.label(views[fit].mask)
    mapping = np.zeros(label.max() + 1, int)
    for i, item in enumerate(parts.values(), 1):
        mapping[item['fit_polygon']] = i
    mark(views[fit].rgb, mapping[label], names).save(folder / 'source.jpg', quality=96)
    surface = Surface(glb)
    frame = capture['geometry']
    basis = np.array(frame['local_axes_in_anchor_camera'])
    origin = frame['local_origin_in_anchor_m']
    measured = read_json(run / 'evaluation/measurements.json')[key]
    scores = {}
    for k in spec['holdout']:
        view = views[k]
        pose = np.array(capture['poses_view_to_anchor'][k])
        depth, bed = camera_image(view, pose, surface, frame)
        y, x = np.where(bed)
        xyz = transform(view.xyz(np.c_[x, y], depth[y, x]), pose)
        predicted = np.zeros(bed.shape, int)
        predicted[y, x] = tree.query(((xyz - origin) @ basis)[:, :2])[1] + 1
        reference, n = ndimage.label(view.mask)
        overlap = np.bincount(
            (predicted * (n + 1) + reference).ravel(), minlength=(len(parts) + 1) * (n + 1)
        ).reshape(-1, n + 1)
        union = overlap.sum(axis=1)[:, None] + overlap.sum(axis=0) - overlap
        iou = overlap[1:, 1:] / np.maximum(1, union[1:, 1:])
        models, annotations = linear_sum_assignment(-iou)
        scores[k] = {}
        for m, a in zip(models, annotations):
            if overlap[m + 1, a + 1] == 0:
                continue
            valid = (predicted == m + 1) & (reference == a + 1) & (view.depth > 0)
            err = np.abs(view.depth[valid] - depth[valid]) * 1000
            scores[k][names[m]] = dict(
                annotation=int(a + 1),
                iou=float(iou[m, a]),
                paired_depth_pixels=len(err),
                depth_abs_median_mm=float(np.median(err)) if len(err) else None,
            )
        total = sum(item['paired_depth_pixels'] for item in scores[k].values())
        if total != measured['heldout'][k]['exported_glb']['visible_pit_depth_samples']:
            raise ValueError('Crossed slot identities or unassigned overlapping pixels in ' + k)
        mark(view.rgb, predicted, names, view.mask).save(folder / (k + '.jpg'), quality=96)
    save_json(output, dict(**record, heldout_per_opening=scores))
    print({k: len(v) for k, v in scores.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.run, a.id, a.out)
