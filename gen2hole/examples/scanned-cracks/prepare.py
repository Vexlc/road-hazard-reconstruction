"""Prepare registered colour, crack footprint and sparse metric heights from scanned pavement."""

import argparse
import hashlib
import os
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def load_map(material, name, inputs):
    info = material['files'][name]
    p = inputs / info['url'].split('/')[-1]
    if hashlib.sha256(p.read_bytes()).hexdigest() != info['sha256']:
        raise ValueError('Unexpected source contents: ' + str(p))
    return p


def read_height(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        raise ValueError('Expected 16-bit linear displacement')
    if raw.ndim == 3:
        if not np.array_equal(raw[:, :, 0], raw[:, :, 1]) or not np.array_equal(raw[:, :, 1], raw[:, :, 2]):
            raise ValueError('Different displacement channels')
        raw = raw[:, :, 0]
    return raw


def crack_mask(rgb, scores):
    """Connected dark valleys in the colour scan; no height samples enter segmentation."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype('f4')
    smooth = cv2.GaussianBlur(gray, (0, 0), 3.0)
    background = cv2.GaussianBlur(gray, (0, 0), 29)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ridge = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, disk)
    score = np.minimum(background - smooth + 3, ridge)
    weak, strong = scores
    mask = cv2.morphologyEx(
        (score > weak).astype('u1'), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    n, component, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    seeds = np.bincount(component[score > strong], minlength=n)
    keep = (stats[:, 4] > 500) & (seeds > 24) & (stats[:, 2:4].max(axis=1) > 85)
    keep[0] = False
    return keep[component]


def plane_from_scan(raw, mask, scale, mid, fit_stride):
    # A shared datum across crops. Only stride-aligned observations may inform this plane.
    rr, cc = np.mgrid[: raw.shape[0] : fit_stride, : raw.shape[1] : fit_stride]
    safe = ~mask[::fit_stride, ::fit_stride]
    a = np.stack([cc[safe], rr[safe], np.ones(safe.sum())], axis=1)
    z = (raw[::fit_stride, ::fit_stride][safe].astype(float) / 65535 - mid) * scale
    keep = np.ones(len(z), bool)
    for _ in range(4):
        coef = np.linalg.lstsq(a[keep], z[keep], rcond=None)[0]
        err = z - a @ coef
        keep = np.abs(err - np.median(err)) < max(
            1.0e-6, 3 * 1.4826 * np.median(np.abs(err - np.median(err)))
        )
    return coef.tolist()


def fit_vertical_scale(samples, normal_path, mask, c, r, step, pitch):
    """Regress terrain slopes against tangent normals in alternate interior 128-pixel blocks."""
    normals = cv2.imread(str(normal_path), cv2.IMREAD_UNCHANGED)
    if normals is None or normals.shape[2] != 3:
        raise ValueError('Expected the registered three-channel OpenGL normal map')
    normals = normals[:, :, ::-1].astype('f4') * 2 - 1
    length = (len(samples) - 1) * step + 1
    normals = normals[r : r + length, c : c + length]
    valid = normals[:, :, 2] > 0.15
    slopes = normals[:, :, :2] / np.maximum(normals[:, :, 2, None], 0.15)
    sigma_source = 4.0
    averaged = cv2.GaussianBlur(slopes, (0, 0), sigma_source)[::step, ::step]
    weights = cv2.GaussianBlur(valid.astype('f4'), (0, 0), sigma_source)[::step, ::step]
    drow, dcol = np.gradient(cv2.GaussianBlur(samples, (0, 0), sigma_source / step), step * pitch)
    surface = np.stack([-dcol, drow], axis=-1)
    rr, cc = np.indices(samples.shape) * step
    gr, gc = rr + r, cc + c
    block = 128
    inset = 16
    fit = (
        ((gr // block + gc // block) % 2 == 0)
        & (gr % block >= inset)
        & (gr % block < block - inset)
        & (gc % block >= inset)
        & (gc % block < block - inset)
    )
    fit &= (
        (np.minimum.reduce([rr, cc, length - 1 - rr, length - 1 - cc]) >= inset)
        & (weights > 0.99)
        & mask[::step, ::step]
    )
    a, b = surface[fit], averaged[fit]
    if len(a) < 32 or (a * a).sum() < 1e-5:
        raise ValueError('Insufficient sloped samples within training blocks')
    robust = np.ones(len(a))
    for _ in range(5):
        factor = float((robust[:, None] * a * b).sum() / (robust[:, None] * a * a).sum())
        err = np.linalg.norm(a * factor - b, axis=-1)
        robust = np.minimum(1, max(1e-5, 1.5 * np.median(err)) / np.maximum(err, 1e-8))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Normals disagree with the orientation of the displacement field')
    return dict(
        factor=factor, training_points=len(a), block_px=block, inset_px=inset, smoothing_px=sigma_source
    )


def prepare(survey, inputs, config, stage):
    refs, cal = stage / 'references', stage / 'calibration'
    refs.mkdir(parents=True)
    cal.mkdir()
    specs = json.loads(survey.read_text())
    designs = []
    for slug, material in specs['materials'].items():
        selected = [s for s in specs['states'] if s['material'] == slug]
        if not selected:
            continue
        base = inputs / slug
        p = load_map(material, 'Diffuse', base)
        rgb = np.array(Image.open(p).convert('RGB'))
        disp_path = load_map(material, 'Displacement', base)
        raw = read_height(disp_path)
        blend = load_map(material, 'blend', base)
        calibration = json.loads(
            subprocess.check_output([config['mesh_python'], HERE / 'blend_scale.py', blend], text=True)
        )
        if calibration['image'] != disp_path.name or rgb.shape[:2] != raw.shape:
            raise ValueError('Mismatched colour/displacement registration')
        res = raw.shape[1]
        width, height = np.array(material['dimensions_mm']) / 1000
        if not np.isclose(width / res, height / raw.shape[0], rtol=1.0e-5):
            raise ValueError('Expected square calibrated texels')
        fit_stride = int(np.lcm.reduce([s['step'] for s in selected]))
        for s in selected:
            key = s['id']
            c, r = s['origin_col_row']
            n = s['intervals']
            step = s['step']
            if (
                step < 1
                or n % step
                or min(c, r) < 0
                or c + n >= res
                or r + n >= len(raw)
                or c % step
                or r % step
            ):
                raise ValueError('Crop must align with the source grid')
            outline = crack_mask(rgb, s['mask_scores'])
            plane = plane_from_scan(raw, outline, calibration['scale_m'], calibration['midlevel'], fit_stride)
            roi = np.s_[r : r + n + 1, c : c + n + 1]
            mask = outline[roi]
            Image.fromarray(rgb[roi]).save(refs / (key + '.png'))
            Image.fromarray(np.uint8(mask) * 255).save(cal / (key + '_outline.png'))
            crop = raw[roi][::step, ::step]
            yy, xx = np.mgrid[r : r + n + 1 : step, c : c + n + 1 : step]
            heights = (crop.astype(float) / 65535 - calibration['midlevel']) * calibration['scale_m']
            vertical_fit = None
            if s.get('height_calibration') == 'normal_fit':
                normal_path = load_map(material, 'nor_gl', base)
                vertical_fit = fit_vertical_scale(heights, normal_path, mask, c, r, step, width / res)
            heights -= plane[0] * xx + plane[1] * yy + plane[2]
            if vertical_fit is not None:
                heights *= vertical_fit['factor']
            np.savez_compressed(cal / (key + '_samples.npz'), height=heights.astype('f4'))
            spec = dict(
                s,
                category='pothole',
                model='crack',
                family=slug,
                depth_profile='scanned_relief',
                span=float(n * width / res),
                source_pitch_m=float(width / res),
                tile_shape=list(raw.shape),
                road_plane_pixel=plane,
                plane_fit_stride=fit_stride,
                material_scale=calibration,
                source_page=material['page'],
                source_author=material['author'],
                source_files=material['files'],
                prompt='Registered photographed surface colour and scan displacement',
            )
            if vertical_fit is not None:
                spec['normal_scale_fit'] = vertical_fit
            designs.append(spec)
    (cal / 'designs.json').write_text(json.dumps(designs, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--survey', type=Path, default=HERE / 'survey.json')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--config', type=Path, default=HERE.parents[1] / 'config/local.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.out.is_symlink():
        p.error('Choose a new output directory')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out.parent, prefix='.scan-') as tmp:
        stage = Path(tmp) / 'input'
        prepare(a.survey, a.inputs, json.loads(a.config.read_text()), stage)
        stage.rename(a.out)
