"""Compare four slab states through a shared gap using identical camera rays and Gaussian depth."""

import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageOps
from scipy import ndimage
import torch

from atlas_gaussians import ATLAS, catalog, digest, load, projection, render

audit = import_module('check-atlas-gaussians')
Plate = import_module('render-pit-survey').Plate
STATES = {'P144': '原位', 'P145': '凸起', 'P146': '下沉', 'P147': '脱落'}


def read(path):
    return json.loads(path.read_text())


def clearance(surface, xy):
    # Measure the actual opening over the photographed edge rather than reading the design clearance.
    x, y = xy
    positions = x + np.linspace(-0.018, 0.018, 721)
    points = np.column_stack((positions, np.full(len(positions), y), np.full(len(positions), 0.3)))
    rays = np.zeros((len(points), 6), dtype='f4')
    rays[:, :3] = points
    rays[:, 5] = -1
    mesh, groups = surface
    hit = mesh.cast_rays(o3d.core.Tensor(rays), nthreads=2)
    depths = hit['t_hit'].numpy()
    if not np.isfinite(depths).all():
        raise ValueError('Missing local clearance samples')
    names = np.array(groups)[hit['geometry_ids'].numpy()]
    labels, _ = ndimage.label(names == 'slab_base')
    middle = int(np.argmin(abs(positions - x)))
    keep = np.flatnonzero(labels == labels[middle])
    if (
        not labels[middle]
        or keep[0] == 0
        or names[keep[0] - 1] != 'slab_top'
        or keep[-1] + 1 >= len(positions)
    ):
        raise ValueError('Clearance section must enter the gap from the top of the slab')
    if names[keep[-1] + 1] not in ('apron', 'pit_or_crack'):
        raise ValueError('Clearance section must leave the gap onto fixed pavement')
    return dict(
        width_mm=float(len(keep) * 0.05),
        sampling_interval_mm=0.05,
        base_z_m=float(np.median(0.3 - depths[keep])),
    )


def compare(run):
    rows = catalog()
    ev = run / 'evaluation'
    shots = {}
    reference = None
    for key in STATES:
        row = rows[key]
        capture = read(ev / 'browser' / (key + '-socket.json'))
        w, h = capture['width'], capture['height']
        camera = capture['camera']
        view, K = projection(camera, w, h)
        if reference is None:
            reference = (view, K, (w, h))
        if not (
            np.allclose(view, reference[0], atol=1e-7, rtol=0)
            and np.allclose(K, reference[1], atol=1e-6, rtol=0)
            and (w, h) == reference[2]
        ):
            raise ValueError('The four captures must share calibrated camera rays')
        if abs(camera['phi']) > 1e-6 or any(
            digest(ATLAS / row[k]) != capture['sources'][k] for k in ('ply', 'glb')
        ):
            raise ValueError('Expected a vertical camera and matching GLB/PLY fingerprints')
        surface = audit.read_surface(ATLAS / row['glb'])
        z, front, geometry = audit.cast_glb(surface[0], camera, w, h)
        groups = np.full(z.shape, '', dtype='U24')
        valid = np.isfinite(z)
        groups[valid] = np.asarray(surface[1])[geometry[valid]]
        rgb, alpha, depth = render(load(row), camera, w, h)
        with Image.open(ev / 'browser' / (key + '-socket-gaussian.png')) as im:
            web = np.asarray(im.convert('RGB'))
        if web.shape != rgb.shape or np.abs(web.astype(float) - rgb)[alpha > 0.85].mean() >= 2:
            raise ValueError('CUDA render disagrees with browser capture')
        with Image.open(ev / 'browser' / (key + '-socket-mesh.png')) as im:
            coverage = np.asarray(
                im.convert('RGBA').getchannel('A').resize((w, h), Image.Resampling.BILINEAR)
            )
        exposed = (groups == 'slab_base') & front
        if not exposed[h // 2, w // 2]:
            raise ValueError('The central ray must reach the exposed foundation')
        shots[key] = dict(z=z, depth=depth, seen=exposed & (alpha > 0.85) & (coverage > 245))
        if key == 'P144':
            x, _, minus_y = camera['target']
            section = clearance(surface, (x, -minus_y))
    common = np.logical_and.reduce([s['seen'] for s in shots.values()])
    if int(common.sum()) < 100:
        raise ValueError('Insufficient shared visible foundation pixels')
    depths = shots['P144']['z'][common]
    records, heatmaps = {}, {}
    for key, s in shots.items():
        if np.abs(s['z'][common] - depths).max() > 1e-5:
            raise ValueError('Foundation intersection moved between states')
        diff = (s['depth'][common] - s['z'][common]) * 1000.0
        colours = np.full((h, w, 3), [226, 228, 219], dtype='u1')
        colours[common] = audit.DEPTH_COLORS[np.digitize(abs(diff), [2, 10, 50])]
        heatmaps[key] = Image.fromarray(colours)
        records[key] = dict(
            median_absolute_error_mm=float(np.median(np.abs(diff))),
            p95_absolute_error_mm=float(np.percentile(np.abs(diff), 95)),
            median_signed_error_mm=float(np.median(diff)),
            fraction_within_10mm=float((np.abs(diff) <= 10).mean()),
        )
    report = dict(
        section=section,
        camera=camera,
        image_size=[w, h],
        common_foundation_pixels=int(common.sum()),
        states=records,
    )
    return report, heatmaps


def plate(run, out, report, heatmaps):
    ev = run / 'evaluation'
    p = Plate(1592, 973)
    p.text(29, 24, '从同一条板缝看进基层', 44, True)
    n = report['common_foundation_pixels']
    p.text(
        30,
        92,
        f'同一实拍碎板的四种状态。相机垂直对准模型边缘，横穿板缝的局部宽度约 {report["section"]["width_mm"]:.1f} mm。',
        23,
    )
    for i, (key, label) in enumerate(STATES.items()):
        x = 30 + i * 389
        p.text(x, 144, key + '　' + label, 28, True)
        for y, kind, title in [
            (201, 'mesh', '带纹理网格'),
            (442, 'gaussian', '原始高斯'),
            (684, 'depth', '共同可见基层'),
        ]:
            p.text(x, y, title, 19)
            if kind == 'depth':
                image = heatmaps[key]
            else:
                with Image.open(ev / 'browser' / (key + '-socket-' + kind + '.png')) as src:
                    image = src.convert('RGB')
            image = ImageOps.contain(image, (370, 196), Image.Resampling.LANCZOS)
            p.image.paste(image, (x, y + 32))
        score = report['states'][key]['median_absolute_error_mm']
        p.text(x, 913, f'基层深度中位绝对差  {score:.1f} mm', 18, True)
    p.text(
        30,
        949,
        f'{n:,} 条相同像素射线均在四态中首先命中同一处基层。深度误差依次以深绿 < 2、浅绿 2–10、黄 10–50、红 ≥ 50 mm 显示。',
        16,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'out', 'figure'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report, heatmaps = compare(args.run)
    plate(args.run, args.figure, report, heatmaps)
    audit.save_json(args.out, report)
    print(report)
