"""Read back matched crops and laser projections, and test camera rays with a second GLB reader."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gen2hole/downstream/sensors'))
from verify_mesh import bvh
from road_comparison_data import read_catalog

ROOT = Path(__file__).resolve().parents[1]
SENSOR_CODE = ROOT / 'gen2hole/downstream/sensors'


def read(path):
    return json.loads(path.read_text())


def require(test, message):
    if not test:
        raise ValueError(message)


@lru_cache(maxsize=None)
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_shot(dest, group, key, shot, raw, proc):
    site = key.split('-')[0]
    frame, camera = shot['frame'], shot['camera']
    source = raw / site / 'camera' / camera
    photo = source / f'{frame:02d}.jpg'
    require(digest(photo) == shot['source_sha256'], str(photo))
    intrinsic = read(source / 'intrinsics.json')
    K = np.array([[intrinsic['fx'], 0, intrinsic['cx']], [0, intrinsic['fy'], intrinsic['cy']], [0, 0, 1]])
    pose = np.eye(4)
    native = read(source / 'poses.json')[frame]
    pose[:3, 3] = [native['position'][k] for k in 'xyz']
    pose[:3, :3] = Rotation.from_quat([native['heading'][k] for k in 'xyzw']).as_matrix()
    require(
        np.array_equal(K, shot['K'])
        and np.allclose(pose, shot['pose'], rtol=0, atol=1e-9)
        and read(source / 'timestamps.json')[frame] == shot['timestamp'],
        'Source camera changed',
    )
    dist = np.linalg.norm(np.array(group['anchor']['center']) - pose[:3, 3])
    require(abs(dist - shot['distance_m']) < 1e-8, 'Incorrect camera distance')
    rect = shot['rect']
    with Image.open(photo) as im, Image.open(dest / 'before.webp') as before:
        require(
            list(im.size) == shot['size']
            and np.array_equal(np.array(im.convert('RGB').crop(rect)), np.array(before.convert('RGB'))),
            'Shared road background differs from the source',
        )
    channel = 0 if camera == 'front_camera' else 5
    with Image.open(proc / site / 'dynamic_masks/all' / f'{frame:03d}_{channel}.png') as mask:
        protected = (
            np.array(mask.convert('L').resize(shot['size'], Image.Resampling.NEAREST).crop(rect)) > 0
        )
    if camera == 'back_camera':
        roof = read(SENSOR_CODE / 'ego_back_profile.json')[site]
        left, top, right, bottom = rect
        edge = np.interp(np.arange(left, right), roof['x'], roof['first_protected_y'])
        protected |= np.arange(top, bottom)[:, None] >= edge
    return protected


def local_cast(tree, ids, group, shot, x, y):
    pose = np.array(shot['pose'])
    K = np.array(shot['K'])
    basis = np.array(group['anchor']['basis'])
    origin = (pose[:3, 3] - group['anchor']['center']) @ basis
    px = np.array([shot['rect'][0] + x + 0.5, shot['rect'][1] + y + 0.5, 1.0])
    direction = (np.linalg.solve(K, px) @ pose[:3, :3].T) @ basis
    point, _, face, _ = tree.ray_cast(origin, direction)
    if point is None:
        return 0, None
    optical = (np.array(point) - origin) @ direction / (direction @ direction)
    return int(ids[face]), float(optical)


part_tree = lru_cache(maxsize=3)(bvh)


def verify_sample(mesh, tree, ids, group, shot, x, y, code, expected):
    part, depth = local_cast(tree, ids, group, shot, x, y)
    if part == code and abs(depth - expected) <= 0.001:
        return abs(depth - expected), None
    # Along a silhouette the two float32 triangle engines can choose different faces.
    # Count every centre mismatch; verify the reported face AND depth within one thousandth of a pixel.
    for dx, dy in [(0, 0.001), (0, -0.001), (0.001, 0), (-0.001, 0)]:
        side, z = local_cast(tree, ids, group, shot, x + dx, y + dy)
        if side == code and abs(z - expected) <= 0.001:
            return abs(z - expected), "boundary"
    if depth is not None and part != code and abs(depth - expected) <= 0.00005:
        # Lid-edge and lid-face triangles can meet or overlap at indistinguishable depths.
        separate, face_ids = part_tree(mesh, code)
        _, alternative = local_cast(separate, face_ids, group, shot, x, y)
        if (
            alternative is not None
            and abs(alternative - depth) <= 0.00005
            and abs(alternative - expected) <= 0.00005
        ):
            return abs(depth - expected), "coincident"
    raise ValueError(f'Independent GLB at pixel {x},{y}: {part, depth} vs {code, expected}')


@lru_cache(maxsize=40)
def recorded_beams(site, frame):
    base = ROOT / 'gen2hole/published/道路病害_传感器与几何增强/base' / site
    source = read(base / 'source.json')['frames'][frame]
    raw = (
        Path(read(ROOT / 'gen2hole/config/local.json')['pandaset_raw'])
        / site
        / 'lidar'
        / f'{frame:02d}.pkl.gz'
    )
    file = base / f'{frame:03d}.npz'
    require(
        digest(raw) == source['lidar_sha256'] and digest(file) == source['base_sha256'],
        'Captured scan changed',
    )
    with np.load(file) as cloud, np.load(base / 'sensor_track.npz') as track:
        t, emitter, xyz = cloud['time'], cloud['sensor'], cloud['xyz']
        timestamps, centers = track['ts'], track['origins']
    idx = np.searchsorted(timestamps, t, side='right').clip(1, len(timestamps) - 1)
    w = (t - timestamps[idx - 1]) / (timestamps[idx] - timestamps[idx - 1])
    origin = centers[emitter, idx - 1] + w[:, None] * (centers[emitter, idx] - centers[emitter, idx - 1])
    return emitter, origin, xyz


def check_returns(data, instance, shot, view, labels, group, mesh, tree, ids, site):
    f = shot['frame']
    sl = slice(*data['frame_offsets'][f : f + 2])
    use = np.flatnonzero((data['action'][sl] == 1) & (data['instance'][sl] == instance)) + sl.start
    require(view['laser']['total'] == len(use), 'Wrong laser instance or frame')
    pose = np.array(shot['pose'])
    cam = (data['xyz'][use] - pose[:3, 3]) @ pose[:3, :3]
    use, cam = use[cam[:, 2] > 0.05], cam[cam[:, 2] > 0.05]
    homog = cam @ np.array(shot['K']).T
    xy = np.floor(homog[:, :2] / homog[:, 2:3]).astype(int) - shot['rect'][:2]
    keep = ((xy >= 0) & (xy < labels.shape[::-1])).all(1)
    xy, cam, use = xy[keep], cam[keep], use[keep]
    points = view['laser']['points']
    emitter, origins, original = recorded_beams(site, f)
    beam_rows = data['row'][use]
    require(
        len(set(beam_rows)) == len(use) and np.array_equal(emitter[beam_rows], data['sensor'][use]),
        'Raw point rows or emitter IDs disagree',
    )
    before = data['range_before_m'][use]
    after = data['range_after_m'][use]
    recorded = original[beam_rows] - origins[beam_rows]
    measured = np.linalg.norm(recorded, axis=1)
    simulated = data['xyz'][use] - origins[beam_rows]
    require(
        np.allclose(before, measured, atol=0.0001, rtol=0)
        and np.allclose(after, np.linalg.norm(simulated, axis=1), atol=0.0001, rtol=0),
        'Replay and raw beam lengths disagree',
    )
    require((measured > 0).all() and (after > 0).all(), 'Invalid range')
    ray_residual = np.linalg.norm(simulated - recorded * (after / measured)[:, None], axis=1)
    require((ray_residual <= 0.0001).all(), 'Simulated return is off the original ray')
    require(len(xy) == len(points), 'Incorrect number of projected rays')
    matched = consistent = samples = boundary = coincident = 0
    for idx, (x, y, code, sensor, error, row, raw_range, placed_range) in enumerate(points):
        require(
            row == beam_rows[idx]
            and raw_range == round(float(before[idx]) * 1e6)
            and placed_range == round(float(after[idx]) * 1e6),
            'Browser range differs from sensor replay',
        )
        require(
            [x, y] == xy[idx].tolist()
            and code == data['part'][use[idx]]
            and sensor == data['sensor'][use[idx]],
            'Simulated beam landed on a different pixel',
        )
        require((labels[y, x] == code) == (error is not None), 'Incorrect camera/laser part agreement')
        if error is not None:
            matched += 1
            consistent += abs(error) <= 50
            if matched == 1 or idx % 61 == 0:
                _, edge = verify_sample(
                    mesh, tree, ids, group, shot, x, y, code, cam[idx, 2] - error / 1000
                )
                boundary += edge == "boundary"
                coincident += edge == "coincident"
                samples += 1
    return dict(
        projected_beams=len(points),
        views_without_projected_beams=int(not points),
        same_part_beams=matched,
        same_part_within_50mm=consistent,
        laser_depth_checks=samples,
        laser_subpixel_boundaries=boundary,
        laser_coincident_parts=coincident,
    )


def previous_maps(folder, case, key):
    result = {}
    for path in (folder / 'camera' / case / 'views.json', folder / 'matched_camera' / case / 'views.json'):
        if path.is_file():
            for row in read(path):
                if row['available'] and row['asset_id'] == key:
                    result.setdefault((row['camera'], row['frame']), (path.parent / row['geometry'], row))
    return result


def compare_existing(previous, shot, new_parts):
    old = previous.get((shot['camera'], shot['frame']))
    if not old:
        return {}
    path, obs = old
    rect = np.array(shot['rect'])
    other = np.array(obs['crop_xyxy'])
    lo = np.maximum(rect[:2], other[:2])
    hi = np.minimum(rect[2:], other[2:])
    require((hi > lo).all(), 'Previous capture does not intersect')

    def roi(box):
        left, top = lo - box[:2]
        right, bottom = hi - box[:2]
        return np.s_[top:bottom, left:right]

    with np.load(path) as g:
        inst = g['instance'][roi(other)]
        original = g['part'][roi(other)]
        # Other placed assets can occlude this instance in a multi-asset reference.
        target = np.where(inst == obs['instance'], original, 0)
        current = new_parts[roi(rect)]
        keep = (inst == 0) | (inst == obs['instance'])
        union = ((target > 0) | (current > 0)) & keep
        mismatch = (target != current) & union
    if mismatch.sum() > max(2, union.sum() * 0.001):
        raise ValueError(f'Prior calibrated part map changed for {path}: {mismatch.sum()}/{union.sum()}')
    return dict(
        previous_camera_overlaps=1,
        previous_part_pixels=int(union.sum()),
        previous_part_differences=int(mismatch.sum()),
    )


def common_beams(group, a, b):
    views = {}
    for shot in group['shots']:

        def hits(key):
            return {p[5]: p for p in group['assets'][key]['views'][shot]['laser']['points']}

        left, right = hits(a), hits(b)
        pairs = [(left[i], right[i]) for i in sorted(left.keys() & right.keys())]
        require(all(x[3] == y[3] and x[6] == y[6] for x, y in pairs), 'Origin changed between alternatives')
        delta = [abs(x[7] - y[7]) / 1000 for x, y in pairs]
        result = dict(shared_projected_rows=len(pairs), over_10mm=sum(v > 10 for v in delta))
        if pairs:
            first, second = pairs[int(np.argmax(delta))]
            result.update(
                max_difference_mm=max(delta),
                median_difference_mm=float(np.median(delta)),
                farthest_apart=dict(
                    scan_row=first[5],
                    sensor=first[3],
                    first_part=first[2],
                    second_part=second[2],
                    original_m=first[6] / 1e6,
                    first_m=first[7] / 1e6,
                    second_m=second[7] / 1e6,
                ),
            )
        views[shot] = result
    return views


def check(run):
    groups = read_catalog(run)
    atlas = ROOT / 'tools/asset_atlas'
    catalog = {
        r['id']: r
        for r in json.loads((atlas / 'catalog.js').read_text().removeprefix('window.roadAssets=')[:-2])
    }
    proof = read(run / 'evidence.json')
    index = read(run / 'index.json')
    cfg = read(ROOT / 'gen2hole/config/local.json')
    raw, proc = Path(cfg['pandaset_raw']), Path(cfg['pandaset_processed'])
    frames = {}
    located = defaultdict(list)
    totals = defaultdict(Counter)
    for g, group in groups.items():
        for n, shot in group['shots'].items():
            frames[g, n] = inspect_shot(run / g / n, group, g, shot, raw, proc)
        for camera in ('front', 'back'):
            pair = [s for n, s in group['shots'].items() if n.startswith(camera)]
            w_h = {(s['rect'][2] - s['rect'][0], s['rect'][3] - s['rect'][1]) for s in pair}
            require(len(w_h) <= 1, 'Apparent scale changes with the crop')
        require(set(group['assets']) == set(proof[g]), 'Missing placement proof')
        for key in group['assets']:
            located[key].append(g)
    require(set(located) == set(index), 'Missing location index')
    errors = []
    methods = Counter()
    for j, (key, locations) in enumerate(sorted(located.items())):
        require(set(locations) == set(index[key]), 'Location index differs for ' + key)
        identity = {k: v for k, v in groups[locations[0]]['assets'][key].items() if k != 'views'}
        reference = identity['reference']['image']
        if key in catalog:
            source = catalog[key]
            sources = {v['image'] for v in source.get('sourceViews', [])} | {source['reference']}
            require(
                reference in sources
                and identity['preview'] == source['preview']
                and identity['depth'] == source.get('depth')
                and identity['depthLabel'] == source.get('depthLabel')
                and identity['method'] == source['method'],
                'Source evidence differs for ' + key,
            )
        else:
            require(
                key == 'A98'
                and identity['method'] == '合成 RGB-D 标定样例'
                and reference.endswith('chamber-pose-final/series/references/pose_capture_image.png'),
                'Unregistered source reference',
            )
        for entry in (reference, identity['preview']):
            with Image.open(atlas / entry) as img:
                img.verify()
        methods[identity['method']] += 1
        tree = ids = None
        for g in locations:
            group = groups[g]
            item = proof[g][key]
            row = group['assets'][key]
            require(
                all(row[k] == v for k, v in identity.items()), 'Asset changed between placements: ' + key
            )
            checks = totals[g]
            scene_file = ROOT / item['scene']
            scene = read(scene_file)
            p = scene['placements'][item['instance'] - 1]
            require(
                scene['site'] + '-' + p['slot'] == g
                and p['asset_id'] == key
                and p['size_multiplier'] == 1
                and np.allclose(p['center_world'], group['anchor']['center'], atol=1e-7, rtol=0)
                and np.allclose(p['basis_columns'], group['anchor']['basis'], atol=1e-7, rtol=0),
                'Anchor differs: ' + key,
            )
            mesh = scene_file.parents[2] / 'assets' / p['category'] / key / 'asset.glb'
            require(digest(mesh) == row['glb_sha256'] == index[key][g], 'Mismatch in published GLB')
            if tree is None:
                tree, ids = bvh(mesh)
            lidar = ROOT / item['laser']
            rec = read(lidar.with_suffix('.json'))
            require(
                rec['scene_sha256'] == digest(scene_file) and rec['delta_sha256'] == digest(lidar),
                'Replay changed',
            )
            with np.load(lidar) as f:
                delta = {
                    name: f[name]
                    for name in (
                        'frame_offsets',
                        'action',
                        'instance',
                        'part',
                        'sensor',
                        'xyz',
                        'row',
                        'range_before_m',
                        'range_after_m',
                    )
                }
            previous = previous_maps(lidar.parents[1], scene['id'], key)
            for n, shot in group['shots'].items():
                dest = run / g / n
                view = row['views'][n]
                with Image.open(dest / (key + '.webp')) as data:
                    overlay = np.array(data.convert('RGBA'))
                with Image.open(dest / (key + '.png')) as data:
                    require(data.mode == 'P' and data.info.get('transparency') == 0, 'Invalid part overlay')
                    labels = np.array(data)
                yy, xx = np.nonzero(labels)
                bounds = (
                    [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
                    if len(xx)
                    else None
                )
                require(view['bounds'] == bounds, 'Zoom extent differs from visible mesh parts')
                changed = overlay[:, :, 3] > 0
                protected = frames[g, n]
                require(
                    np.array_equal(np.unique(overlay[:, :, 3]), np.unique(changed.astype('u1') * 255))
                    and changed.sum() == view['changed']
                    and not changed[protected].any()
                    and not labels[protected].any(),
                    'Painted over a protected camera pixel or changed layer opacity',
                )
                require(
                    labels.shape == protected.shape
                    and np.bincount(labels[labels > 0], minlength=13).tolist() == view['parts']
                    and view['visible'] == (labels > 0).sum()
                    and 0 <= view['opaque'] <= view['visible'],
                    'Wrong part counts',
                )
                for x, y, code, z in item['probes'][n]:
                    require(code == labels[y, x], 'Recorded ray and visible part layer disagree')
                    try:
                        error, edge = verify_sample(mesh, tree, ids, group, shot, x, y, code, z)
                    except ValueError as problem:
                        raise ValueError(f'{g} {key} {n}: {problem}') from problem
                    checks['camera_subpixel_boundaries'] += int(edge == 'boundary')
                    checks['camera_coincident_parts'] += int(edge == 'coincident')
                    if edge != "boundary":
                        errors.append(error)
                checks.update(
                    check_returns(
                        delta, item['instance'], shot, view, labels, group, mesh, tree, ids, scene['site']
                    )
                )
                checks.update(compare_existing(previous, shot, labels))
                checks.update(
                    views=1,
                    visible_pixels=view['visible'],
                    changed_pixels=view['changed'],
                    empty_mesh_views=int(view['visible'] == 0),
                    camera_ray_checks=len(item['probes'][n]),
                )
            checks['placements'] += 1
        del tree, ids
        if (j + 1) % 20 == 0:
            print(j + 1, 'assets', flush=True)
    total = Counter()
    for counts in totals.values():
        total.update(counts)
    examples = [
        (g, a, b)
        for g, a, b in [
            ('109-01', 'P112', 'A119'),
            ('109-01', 'A119', 'M118'),
            ('109-01', 'P122', 'P130'),
            ('034-01', 'A117', 'P135'),
            ('109-05', 'P143', 'M138'),
            ('109-05', 'A127', 'A128'),
            ('110-01', 'P02_wet_wheel_track', 'A93'),
            ('109-03', 'M11', 'M03_brick_lined'),
        ]
        if g in groups and a in groups[g]['assets'] and b in groups[g]['assets']
    ]
    pairs = {g + '/' + a + '-' + b: common_beams(groups[g], a, b) for g, a, b in examples}
    return dict(
        shared_beam_examples=pairs,
        assets=len(index),
        reconstruction_methods=dict(methods),
        locations=len(groups),
        camera_poses=len(frames),
        metrics=dict(total),
        max_center_ray_depth_error_mm=float(max(errors) * 1000),
        by_location={k: dict(v) for k, v in totals.items()},
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/road-comparisons')
    parser.add_argument('--out', type=Path, default=ROOT / 'validation/matched-road-views.json')
    opt = parser.parse_args()
    result = check(opt.run)
    opt.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(result['metrics'])
