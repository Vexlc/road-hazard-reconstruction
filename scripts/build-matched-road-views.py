"""Render alternative road assets at their verified original placements on identical vehicle-camera pixels."""

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gen2hole/downstream/sensors'))
from camera_geometry import camera_hits, crop_box, foreground, outline_points, roof_edge, track
from render_pairs import gather, render
from road_geometry import read_json, sha256
from road_comparison_data import reference_image, write_catalog

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'tools/asset_atlas'
RELEASE = (ROOT / 'gen2hole/published').resolve()
SENSORS = RELEASE / '道路病害_传感器与几何增强'
digest = lru_cache(maxsize=None)(sha256)
COLORS = [
    (0, 0, 0),
    (0, 0, 0),
    (220, 128, 63),
    (172, 86, 64),
    (63, 161, 190),
    (239, 188, 83),
    (118, 89, 174),
    (74, 73, 122),
    (144, 149, 137),
    (59, 156, 127),
    (121, 194, 169),
    (49, 108, 97),
    (195, 87, 131),
]


def inventory():
    rows = json.loads((ATLAS / 'catalog.js').read_text().removeprefix('window.roadAssets=')[:-2])
    rows = {r['id']: r for r in rows}
    paths = sorted(
        [*RELEASE.glob('*/scenes/*/scene.json'), *(ROOT / 'runs').glob('*/series/scenes/*/scene.json')]
    )
    eligible = {(ATLAS / r['glb']).resolve().parents[3] for r in rows.values()}
    eligible.add(ROOT / 'runs/chamber-pose-final/series')
    groups = defaultdict(dict)
    for path in paths:
        bundle = path.parents[2]
        if bundle not in eligible:
            continue
        scene = read_json(path)
        if bundle.parent == RELEASE:
            sensors = SENSORS
        else:
            sensors = bundle.parent / 'sensors'
            if (bundle.parent / 'sensors-multisite').is_dir():
                sensors = bundle.parent / 'sensors-multisite'
        case = scene['id']
        lidar = sensors / 'cases' / (case + '.npz')
        record = read_json(lidar.with_suffix('.json'))
        checks = read_json(path.parent / 'complete.json')['files']
        if not (
            checks['scene.json'] == digest(path) == record['scene_sha256']
            and digest(lidar) == record['delta_sha256']
            and scene['checkpoint_reloaded']
        ):
            raise ValueError('Changed scene or laser replay at ' + str(path))
        for number, pl in enumerate(scene['placements'], 1):
            key = pl['asset_id']
            folder = bundle / 'assets' / pl['category'] / key
            row = rows.get(key)
            if row is None:
                # This synthetic calibration control has a verified placement outside the public atlas.
                if bundle != ROOT / 'runs/chamber-pose-final/series' or key != 'A98':
                    continue
                meta = read_json(folder / 'metadata.json')
                hashes = read_json(bundle / 'checks.json')[key]['sha256']
                preview = bundle / 'views' / (key + '_inspection.jpg')
                row = dict(
                    label='标定样例 · 深度观测开盖',
                    method='合成 RGB-D 标定样例',
                    width=meta['width_m'],
                    length=meta['length_m'],
                    preview='../../' + str(preview.relative_to(ROOT)),
                    reference='../../'
                    + str((bundle / 'references/pose_capture_image.png').relative_to(ROOT)),
                    depth=meta['depth_measured_m'],
                    depthLabel='合成深度观测所得井深',
                    glb=str(folder / 'asset.glb'),
                    gaussianSources=dict(glb=hashes['asset.glb'], ply=hashes['asset.ply']),
                )
            source = row['gaussianSources']
            if (
                (ATLAS / row['glb']).resolve() != folder.resolve() / 'asset.glb'
                or digest(folder / 'asset.glb') != source['glb']
                or digest(folder / 'asset.ply') != source['ply']
                or pl.get('asset_ply_sha256', source['ply']) != source['ply']
                or pl['size_multiplier'] != 1
                or pl['reload_max_pixel_error'] != 0
            ):
                raise ValueError('Placement and atlas asset differ: ' + key)
            replay = record['assets'][number - 1]
            if (replay['asset_id'], replay['instance'], replay['slot']) != (key, number, pl['slot']):
                raise ValueError('Misaligned laser instance: ' + key)
            group = groups[scene['site'] + '-' + pl['slot']]
            if key in group:
                raise ValueError('Repeated asset at one anchor: ' + key)
            cp = path.parent / 'trained_model/iteration_7000.pth'
            group[key] = dict(
                folder=folder,
                row=row,
                scene=path,
                pl=pl,
                checkpoint=cp,
                checkpoint_sha=checks[str(cp.relative_to(path.parent))],
                lidar=lidar,
                number=number,
            )
    return groups


@lru_cache(maxsize=None)
def outline(folder):
    return outline_points(folder)


def common_views(members, raw, site):
    rows = list(members.values())
    anchor = rows[0]['pl']
    center = np.array(anchor['center_world'])
    basis = np.array(anchor['basis_columns'])
    for r in rows:
        p = r['pl']
        if not np.allclose(p['center_world'], center, atol=1e-7, rtol=0) or not np.allclose(
            p['basis_columns'], basis, atol=1e-7, rtol=0
        ):
            raise ValueError('Different poses at the same anchor')
    cloud = np.concatenate([outline(r['folder']) for r in rows])
    world = cloud[ConvexHull(cloud).vertices] @ basis.T + center
    result = {}
    for camera in ('front_camera', 'back_camera'):
        times, poses, K, size = track(str(raw), site, camera)
        options = []
        for frame, pose in enumerate(poses):
            cam = (world - pose[:3, 3]) @ pose[:3, :3]
            if cam[:, 2].min() < 0.5:
                continue
            uv = (cam @ K.T)[:, :2] / cam[:, 2:3]
            lo, hi = uv.min(0), uv.max(0)
            if (lo < 10).any() or (hi > np.array(size) - 10).any():
                continue
            if (
                camera == 'back_camera'
                and hi[1] + 3 >= roof_edge(site, np.linspace(lo[0], hi[0], 32)).min()
            ):
                continue
            distance = float(np.linalg.norm(pose[:3, 3] - center))
            options.append((frame, distance, crop_box([*lo, *hi], size)))
        if not options:
            continue
        near = min(options, key=lambda row: row[1])
        further = [r for r in options if r[1] >= near[1] + 4]
        choices = [('near', near)]
        if further:
            choices.append(('far', min(further, key=lambda row: abs(row[1] - max(22, near[1] + 10)))))
        # A fixed pixel span for near and far preserves the apparent angular size as the car moves.
        w = max(rect[2] - rect[0] for _, (_, _, rect) in choices)
        h = max(rect[3] - rect[1] for _, (_, _, rect) in choices)
        for distance_name, (frame, distance, rect) in choices:
            x = int(np.clip(round((rect[0] + rect[2] - w) / 2), 0, size[0] - w))
            y = int(np.clip(round((rect[1] + rect[3] - h) / 2), 0, size[1] - h))
            rect = [x, y, x + w, y + h]
            name = camera.split('_')[0] + '-' + distance_name
            result[name] = dict(
                camera=camera,
                frame=frame,
                distance_m=distance,
                rect=rect,
                size=list(size),
                K=K.tolist(),
                pose=poses[frame].tolist(),
                timestamp=float(times[frame]),
            )
    if not result:
        raise ValueError('No common calibrated camera pose at ' + site + '-' + anchor['slot'])
    return result, dict(center=center.tolist(), basis=basis.tolist())


def background(raw, site, shot, folder):
    camera, frame, rect = shot['camera'], shot['frame'], shot['rect']
    src = raw / site / 'camera' / camera / f'{frame:02d}.jpg'
    with Image.open(src) as image:
        image = image.convert('RGB')
        shot['source_sha256'] = digest(src)
        before = image.crop(rect)
        before.save(folder / 'before.webp', lossless=True, method=2)
        ImageDraw.Draw(image).rectangle(rect, outline='#ffe08b', width=4)
        image.thumbnail((960, 540), Image.Resampling.LANCZOS)
        image.save(folder / 'road.jpg', quality=90)
    return np.array(before)


def laser_points(delta, instance, shot, labels, depth):
    i = shot['frame']
    lo, hi = delta['frame_offsets'][i : i + 2]
    keep = (delta['instance'][lo:hi] == instance) & (delta['action'][lo:hi] == 1)
    indices = np.flatnonzero(keep) + lo
    total = len(indices)
    pose, K = np.array(shot['pose']), np.array(shot['K'])
    cam = (delta['xyz'][indices] - pose[:3, 3]) @ pose[:3, :3]
    forward = cam[:, 2] > 0.05
    cam, indices = cam[forward], indices[forward]
    xy = np.floor((cam @ K.T)[:, :2] / cam[:, 2:3]).astype(int) - shot['rect'][:2]
    inside = ((xy >= 0) & (xy < np.array(labels.shape[::-1]))).all(1)
    indices, cam = indices[inside], cam[inside]
    x, y = xy[inside].T
    same = labels[y, x] == delta['part'][indices]
    residual = np.where(same, cam[:, 2] - depth[y, x], 0)
    points = []
    for j, idx in enumerate(indices):
        points.append(
            [
                int(x[j]),
                int(y[j]),
                int(delta['part'][idx]),
                int(delta['sensor'][idx]),
                round(1000 * float(residual[j]), 1) if same[j] else None,
                int(delta['row'][idx]),
                round(float(delta['range_before_m'][idx]) * 1e6),
                round(float(delta['range_after_m'][idx]) * 1e6),
            ]
        )
    return dict(total=total, points=points)


@torch.no_grad()
def alternatives(item, site, shots, photos, cfg, dest):
    p, path = item['pl'], item['checkpoint']
    key = p['asset_id']
    if digest(path) != item['checkpoint_sha']:
        raise ValueError('Native checkpoint changed: ' + str(path))
    splats, models, _ = gather(path, [p], {key: (item['folder'], None)})
    with np.load(item['lidar']) as f:
        delta = {
            k: f[k]
            for k in (
                'frame_offsets',
                'instance',
                'action',
                'xyz',
                'part',
                'sensor',
                'row',
                'range_before_m',
                'range_after_m',
            )
        }
    report = {}
    for name, shot in shots.items():
        pose, K, rect = np.array(shot['pose']), np.array(shot['K']), shot['rect']
        depth, parts, instance, _ = camera_hits(pose, K, rect, models)
        layers = foreground(
            cfg['pandaset_raw'],
            cfg['pandaset_processed'],
            SENSORS,
            site,
            shot['camera'],
            shot['frame'],
            rect,
            depth,
        )
        protected = np.logical_or.reduce(layers)
        target = (parts > 1) & ~protected
        labels = np.where(target, parts, 0).astype('u1')
        src = photos[name]
        rgb, alpha, _, _ = render(
            splats, pose, K, rect, src, depth, parts, instance, protected, models, site, shot['camera']
        )
        if (
            not np.array_equal(rgb[protected], src[protected])
            or alpha[protected].any()
            or target[[0, -1], :].any()
            or target[:, [0, -1]].any()
        ):
            raise ValueError(f'Foreground changed or target clipped at {site}/{key}/{name}')
        folder = dest / name
        changed = np.any(rgb != src, axis=-1)
        rgba = np.dstack((np.where(changed[:, :, None], rgb, 0), changed.astype('u1') * 255))
        Image.fromarray(rgba).save(folder / (key + '.webp'), lossless=True, method=2)
        mask = Image.frombytes('P', labels.shape[::-1], labels.tobytes())
        mask.putpalette([v for color in COLORS for v in color])
        mask.save(folder / (key + '.png'), transparency=0)
        part_counts = np.bincount(labels[target], minlength=len(COLORS))
        # Save an optical depth per sampled visible component for independent camera-ray readback.
        probes = []
        for code in np.flatnonzero(part_counts):
            y, x = np.where(labels == code)
            for idx in np.unique(np.linspace(0, len(y) - 1, min(7, len(y)), dtype=int)):
                probes.append([int(x[idx]), int(y[idx]), int(code), float(depth[y[idx], x[idx]])])
        report[name] = dict(
            parts=part_counts.tolist(),
            bounds=mask.getbbox(),
            visible=int(target.sum()),
            changed=int(changed.sum()),
            opaque=int((target & (alpha > 0.48)).sum()),
            protected=int(((parts > 1) & protected).sum()),
            laser=laser_points(delta, item['number'], shot, labels, depth),
            probes=probes,
        )
    return report


def build(stage, cfg, selection):
    groups = inventory()
    if selection:
        groups = {k: groups[k] for k in selection}
    raw = Path(cfg['pandaset_raw'])
    catalog = {}
    evidence = {}
    for name, members in sorted(groups.items()):
        site = name.split('-')[0]
        shots, anchor = common_views(members, raw, site)
        dest = stage / name
        photos = {}
        for key, shot in shots.items():
            path = dest / key
            path.mkdir(exist_ok=True, parents=True)
            photos[key] = background(raw, site, shot, path)
        group = dict(anchor=anchor, shots=shots, assets={})
        catalog[name] = group
        evidence[name] = {}
        print(
            name,
            len(members),
            'placements',
            [(n, s['frame'], round(s['distance_m'], 1)) for n, s in shots.items()],
            flush=True,
        )
        for key, item in sorted(members.items()):
            views = alternatives(item, site, shots, photos, cfg, dest)
            evidence[name][key] = dict(
                scene=str(item['scene'].relative_to(ROOT)),
                checkpoint_sha256=item['checkpoint_sha'],
                laser=str(item['lidar'].relative_to(ROOT)),
                instance=item['number'],
                probes={k: v.pop('probes') for k, v in views.items()},
            )
            r = item['row']
            group['assets'][key] = dict(
                label=r['label'],
                method=r['method'],
                width=r['width'],
                length=r['length'],
                views=views,
                preview=r['preview'],
                reference=reference_image(r),
                depth=r.get('depth'),
                depthLabel=r.get('depthLabel'),
                glb_sha256=r['gaussianSources']['glb'],
                detail='../../docs/02-asset-reconstruction.md#从观测恢复井盖姿态'
                if key == 'A98'
                else 'index.html#' + key,
            )
        del photos
    index = defaultdict(dict)
    for name, group in catalog.items():
        for key, item in group['assets'].items():
            index[key][name] = item['glb_sha256']
    (stage / 'index.json').write_text(json.dumps(index, sort_keys=True, indent=2) + '\n')
    write_catalog(stage, catalog)
    (stage / 'evidence.json').write_text(json.dumps(evidence, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/road-comparisons')
    parser.add_argument('--groups', nargs='+', help='Optional subset of site-slot pairs')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new output directory')
    cfg = read_json(ROOT / 'gen2hole/config/local.json')
    cv2.setNumThreads(2)
    torch.set_num_threads(3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.matched-road-', dir=args.out.parent) as tmp:
        stage = Path(tmp) / 'views'
        build(stage, cfg, args.groups)
        stage.rename(args.out)
