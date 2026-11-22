"""Evaluate two PLY samplings on the same foundation rays, including uniformly spaced unseen rim locations."""

import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from atlas_gaussians import digest, load, render

audit = import_module('check-atlas-gaussians')
slabs = import_module('check-slab-clearance')
Plate = import_module('render-pit-survey').Plate


def read(path):
    return json.loads(path.read_text())


def retained_points(first, second):
    def coordinates(vertices):
        return np.stack([vertices[f] for f in 'xyz'], axis=1).copy().view('V12').ravel()

    a, b = [PlyData.read(p, mmap='r')['vertex'].data for p in (first, second)]
    _, old, new = np.intersect1d(coordinates(a), coordinates(b), return_indices=True)
    if not len(old):
        raise ValueError('Missing original points away from the sampling band')
    difference = 0.0
    for k in a.dtype.names:
        if k.startswith('f_'):
            difference = max(difference, float(abs(a[k][old] - b[k][new]).max()))
        elif not np.array_equal(a[k][old], b[k][new]):
            raise ValueError('A retained Gaussian changed shape, opacity or position')
    with (
        np.load(first.with_name('splat_material.npz')) as coarse,
        np.load(second.with_name('splat_material.npz')) as fine,
    ):
        if not np.array_equal(coarse['part_names'], fine['part_names']) or any(
            not np.array_equal(coarse[f][old], fine[f][new]) for f in ('road', 'part', 'ambient_visibility')
        ):
            raise ValueError('Material or ambient visibility changed on a retained point')
    return dict(
        area_count=len(a), edge_count=len(b), retained_points=len(old), max_retained_sh_change=difference
    )


def heldout_views(reference, meta):
    # Select by arclength alone before rendering either PLY. The ring is counterclockwise in local XY.
    ring = np.array(meta['slab']['boundary_xy'])
    seg = np.roll(ring, -1, axis=0) - ring
    if (ring[:, 0] * seg[:, 1] - ring[:, 1] * seg[:, 0]).sum() <= 0:
        raise ValueError('Expected an outward-facing right normal')
    length = np.linalg.norm(seg, axis=1)
    for target in (np.arange(8) + 0.5) * length.sum() / 8:
        j = int(np.searchsorted(length.cumsum(), target))
        centre = ring[j] + seg[j] / 2 + seg[j, ::-1] * [1, -1] / length[j] * meta['source_grid_pitch_m'] / 2
        yield dict(reference, target=[float(centre[0]), 0.0, -float(centre[1])])


def evaluate(baseline, fine, out, figure):
    base_assets, fine_assets = [p.resolve() / 'series/assets/pothole' for p in (baseline, fine)]
    coarse_checks, edge_checks = [read(p / 'series/checks.json') for p in (baseline, fine)]
    rebuilds = [
        read(path / 'evaluation/measurements.json')
        if (path / 'evaluation/measurements.json').is_file()
        else {}
        for path in (baseline, fine)
    ]
    evidence = baseline / 'evaluation/browser'
    keys = list(slabs.STATES)
    info = read(evidence / 'P144-socket.json')
    camera = info['camera']
    w, h = info['width'], info['height']
    assets = {}
    source = {}
    for k in keys:
        coarse, root = base_assets / k, fine_assets / k
        hashes = edge_checks[k]['sha256']
        p, q = coarse / 'asset.ply', root / 'asset.ply'
        if not (
            digest(coarse / 'asset.glb') == digest(root / 'asset.glb') == hashes['asset.glb']
            and digest(p) == coarse_checks[k]['sha256']['asset.ply']
            and digest(q) == hashes['asset.ply']
            and digest(coarse / 'surface.npz') == digest(root / 'surface.npz') == hashes['surface.npz']
        ):
            raise ValueError('Mesh, surface field or PLY identity mismatch')
        first = dict(id=k, ply=str(p), gaussians=coarse_checks[k]['gaussians'])
        second = dict(id=k, ply=str(q), gaussians=edge_checks[k]['gaussians'])
        source[k] = dict(
            **retained_points(p, q),
            glb_sha256=hashes['asset.glb'],
            area_ply_sha256=digest(p),
            edge_ply_sha256=hashes['asset.ply'],
        )
        for name, directory, record in zip(('area', 'edges'), (coarse, root), rebuilds):
            hashes = record.get(k, {}).get('independent_rebuild_sha256')
            if hashes:
                if any(digest(directory / filename) != expected for filename, expected in hashes.items()):
                    raise ValueError('Independent rebuild differs for ' + k + '/' + name)
                source[k][name + '_independent_rebuild_sha256'] = hashes
        assets[k] = audit.read_surface(coarse / 'asset.glb'), load(first), load(second)
    meta = read(base_assets / keys[0] / 'metadata.json')
    views = [camera, *heldout_views(camera, meta)]
    stats = []
    illustrations = {}
    for i, cam in enumerate(views):
        shots = {}
        for k, ((scene, groups), original, refined) in assets.items():
            z, front, g = audit.cast_glb(scene, cam, w, h)
            finite = np.isfinite(z)
            surface = np.full(g.shape, '', dtype='U24')
            surface[finite] = np.array(groups)[g[finite]]
            rgb, alpha, depth = render(original, cam, w, h)
            improved, opacity, new_depth = render(refined, cam, w, h)
            visible = front & (surface == 'slab_base') & (alpha > 0.85)
            if i == 0:
                capture = read(evidence / (k + '-socket.json'))
                if capture['camera'] != cam or capture['sources']['ply'] != source[k]['area_ply_sha256']:
                    raise ValueError('Original camera or cloud changed')
                with Image.open(evidence / (k + '-socket-mesh.png')) as im:
                    coverage = np.array(
                        im.convert('RGBA').getchannel('A').resize((w, h), Image.Resampling.BILINEAR)
                    )
                visible &= coverage > 245
                if k in ('P145', 'P147'):
                    illustrations[k] = [rgb, improved]
            shots[k] = dict(z=z, visible=visible, depth=depth, refined=new_depth, opacity=opacity)
        common = np.logical_and.reduce([s['visible'] for s in shots.values()])
        if common.sum() < 100:
            raise ValueError('Insufficient shared baseline rays at ' + str(cam))
        data = {}
        for k, s in shots.items():
            if np.max(abs(s['z'][common] - shots['P144']['z'][common])) > 1.0e-5:
                raise ValueError('The first mesh intersection differs between states')
            if not np.isfinite(s['refined'][common]).all():
                raise ValueError('Refined cloud lost a baseline ray')
            measures = {}
            for field in ('depth', 'refined'):
                difference = abs(s[field][common] - s['z'][common]) * 1000.0
                measures[field] = dict(
                    median_mm=float(np.median(difference)),
                    p95_mm=float(np.percentile(difference, 95)),
                    fraction_within_10mm=float((difference <= 10.0).mean()),
                )
                if i == 0 and k in illustrations:
                    image = np.full((h, w, 3), [226, 228, 219], dtype='u1')
                    image[common] = audit.DEPTH_COLORS[np.digitize(difference, [2, 10, 50])]
                    illustrations[k].append(image)
            data[k] = dict(
                area=measures['depth'],
                edges=measures['refined'],
                refined_alpha_over_85=int((s['opacity'][common] > 0.85).sum()),
            )
        stats.append(
            dict(
                view='original' if i == 0 else 'rim_' + str(i),
                target=cam['target'],
                common_foundation_rays=int(common.sum()),
                states=data,
            )
        )
    report = dict(sources=source, camera=camera, image_size=[w, h], views=stats)
    audit.save_json(out, report)
    draw(figure, evidence, report, illustrations)
    return report


def draw(path, evidence, report, images):
    p = Plate(1580, 930)
    p.text(30, 26, '加密断边后，高斯能看到多深的板缝', 43, True)
    p.text(31, 96, '从同一俯视机位对照两种 PLY 采样。网格局部缝宽约 4 毫米，基层设计高程 −85 毫米。', 23)
    width = 292
    for j, title in enumerate(['带纹理网格', '原始高斯', '断边加密高斯', '原始深度误差', '加密后深度误差']):
        p.text(30 + j * 310, 161, title, 23, True)
    for i, k in enumerate(('P145', 'P147')):
        y = 214 + i * 310
        p.text(30, y, ('碎板凸起 35 mm' if i == 0 else '整块碎板移走') + '　' + k, 23, True)
        with Image.open(evidence / (k + '-socket-mesh.png')) as im:
            mesh = im.convert('RGB').resize(tuple(report['image_size']), Image.Resampling.LANCZOS)
        for j, image in enumerate([mesh, *map(Image.fromarray, images[k])]):
            w, h = image.size
            crop = image.crop((w // 2 - 116, h // 2 - 84, w // 2 + 116, h // 2 + 84)).resize(
                (width, 211), Image.Resampling.LANCZOS
            )
            p.image.paste(crop, (30 + j * 310, y + 42))
        v = report['views'][0]['states'][k]
        median = v['edges']['median_mm']
        median_label = '< 0.01' if median < 0.01 else f'{median:.2f}'
        p.text(
            30,
            y + 268,
            f"共同基层像素的深度中位绝对差　{v['area']['median_mm']:.1f} → {median_label} mm　　　第 95 百分位　{v['area']['p95_mm']:.1f} → {v['edges']['p95_mm']:.1f} mm",
            19,
        )
    p.text(
        30,
        852,
        f"深度统计取完整机位的 {report['views'][0]['common_foundation_rays']:,} 个共同像素，上图放大中央局部。底色为统计范围之外。",
        19,
    )
    for j, t in enumerate(['< 2 mm', '2–10 mm', '10–50 mm', '≥ 50 mm']):
        x = 30 + j * 175
        p.draw.rectangle((x, 897, x + 22, 914), fill=tuple(map(int, audit.DEPTH_COLORS[j])))
        p.text(x + 32, 893, t, 18)
    path.parent.mkdir(parents=True, exist_ok=True)
    p.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('baseline', 'fine', 'out', 'figure'):
        parser.add_argument('--' + k, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    result = evaluate(args.baseline, args.fine, args.out, args.figure)
    print([(v['view'], v['common_foundation_rays']) for v in result['views']])
