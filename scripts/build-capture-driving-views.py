"""Recover vehicle-camera views of individual RGB-D and stereo pits in the 055/056 joint splat renders."""

import argparse
from collections import Counter
import fcntl
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile

import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage

from pit_road import ROOT, ATLAS, PitBed
from road_geometry import glb_triangles, read_json, save_json, sha256

JOINT = ROOT / 'outputs/road_hazard_joint_scenes'
PROCESSED = ROOT / 'work/data/processed'
FRAMES = (37, 44)


@lru_cache(maxsize=4)
def camera(site, frame):
    path = PROCESSED / site
    with Image.open(path / 'images' / f'{frame:03d}_0.jpg') as photo:
        source_size = photo.size
    original = np.array(Image.open(JOINT / f'original_{site}_{frame}.jpg').convert('RGB'))
    h, w = original.shape[:2]
    fx, fy, cx, cy = np.loadtxt(path / 'intrinsics/0.txt').ravel()[:4] * (
        np.array([w, h, w, h]) / np.tile(source_size, 2)
    )
    pose = np.loadtxt(path / 'extrinsics' / f'{frame:03d}_0.txt')
    with Image.open(path / 'dynamic_masks/all' / f'{frame:03d}_0.png') as im:
        if im.size != source_size:
            raise ValueError('Dynamic mask and capture dimensions differ')
        dynamic = np.asarray(im.convert('L').resize((w, h), Image.Resampling.NEAREST)) > 0
    return original, dynamic, pose, np.array([fx, fy]), np.array([cx, cy])


@lru_cache(maxsize=1600)
def corners(path):
    with path.open('rb') as stream:
        head = stream.read(20)
        model = json.loads(stream.read(struct.unpack_from('<I', head, 12)[0]))
    mins, maxs = [], []
    for mesh in model['meshes']:
        for prim in mesh['primitives']:
            pos = model['accessors'][prim['attributes']['POSITION']]
            mins.append(pos['min'])
            maxs.append(pos['max'])
    v = np.array(list(product(*zip(np.min(mins, 0), np.max(maxs, 0)))))
    return v[:, [0, 2, 1]] * [1, -1, 1]


def camera_bounds(local, placement, cam):
    original, _, pose, fl, pp = cam
    basis, center = np.array(placement['basis_columns']), np.array(placement['center_world'])
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5) or placement['size_multiplier'] != 1:
        raise ValueError('Expected rigid placement at native scale')
    v = (local @ basis.T + center - pose[:3, 3]) @ pose[:3, :3]
    if v[:, 2].min() <= 0:
        return None
    uv = v[:, :2] / v[:, 2:] * fl + pp
    size = original.shape[1::-1]
    x0, y0 = np.maximum(0, np.floor(uv.min(0) - 2).astype(int))
    x1, y1 = np.minimum(size, np.ceil(uv.max(0) + 2).astype(int))
    return (int(x0), int(y0), int(x1), int(y1)) if x1 > x0 and y1 > y0 else None


def joint_surface(placements, rows):
    scene = o3d.t.geometry.RaycastingScene(nthreads=3)
    owners = []
    for i, placement in enumerate(placements):
        basis = np.asarray(placement['basis_columns'], dtype='f4')
        center = np.asarray(placement['center_world'], dtype='f4')
        for _, xyz, faces in glb_triangles(ATLAS / rows[placement['asset_id']]['glb'], names=True):
            label = scene.add_triangles(o3d.core.Tensor(xyz @ basis.T + center), o3d.core.Tensor(faces))
            if label != len(owners):
                raise ValueError('Unexpected triangle group')
            owners.append(i)
    return scene, np.array(owners)


def observation(bed, placement, cam, after, box, index, joint, threshold):
    if box is None:
        return dict(status='outside_frame')
    before, dynamic, pose, fl, pp = cam
    x, y, r, b = box
    py, px = np.mgrid[y:b, x:r] + 0.5
    uv = np.stack([px, py], axis=-1)
    direction = np.dstack(((uv - pp) / fl, np.ones(px.shape)))
    basis = np.array(placement['basis_columns'])
    center = np.array(placement['center_world'])
    world_dir = direction @ pose[:3, :3].T
    local_dir = world_dir @ basis
    eye = (pose[:3, 3] - center) @ basis
    rays = np.concatenate([np.broadcast_to(eye, local_dir.shape), local_dir], axis=2).astype('f4')
    hit = bed.scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
    dist = hit['t_hit'].numpy()
    xyz = eye + np.where(np.isfinite(dist), dist, 0)[..., None] * local_dir
    mask = np.isfinite(dist) & (np.sum(hit['primitive_normals'].numpy() * local_dir, axis=2) < 0)
    annotated = (
        ndimage.map_coordinates(
            bed.annotation, bed.grid_indices(xyz[:, :, :2]), order=1, mode='constant', cval=0
        )
        >= 0.5
    )
    mask &= annotated
    geom = int(mask.sum())
    gy, gx = np.where(mask)
    mask &= ~dynamic[y:b, x:r]
    static = int(mask.sum())
    world_rays = np.concatenate([np.broadcast_to(pose[:3, 3], world_dir.shape), world_dir], axis=2).astype(
        'f4'
    )
    scene, owners = joint
    first = scene.cast_rays(o3d.core.Tensor(world_rays), nthreads=2)
    first_z = first['t_hit'].numpy()
    other = np.zeros(mask.shape, bool)
    reached = np.isfinite(first_z)
    other[reached] = owners[first['geometry_ids'].numpy()[reached]] != index
    mask &= ~(other & (first_z + 0.001 < dist))
    pixels = int(mask.sum())
    record = dict(
        projected_front_pixels=geom,
        dynamic_masked_pixels=geom - static,
        other_asset_occluded_pixels=static - pixels,
        eligible_pixels=pixels,
    )
    if pixels < 12:
        return dict(record, status='insufficient_projected_area')
    diff = abs(after[y:b, x:r].astype('i2') - before[y:b, x:r]).mean(axis=2)
    changed = int((diff[mask] > threshold).sum())
    record.update(changed_pixels=changed, mean_abs_rgb=float(diff[mask].mean()), change_threshold=threshold)
    if changed < 8 or changed / pixels < 0.2:
        return dict(record, status='weak_image_change')
    return dict(
        record,
        status='visible',
        bbox=[x + int(gx.min()), y + int(gy.min()), x + int(gx.max()) + 1, y + int(gy.max()) + 1],
    )


def window(box, size):
    x, y, r, b = box
    w = min(size[0], max(112, round(max((r - x) * 2.8, (b - y) * 4.2))))
    h = min(size[1], round(w * 2 / 3))
    left = int(np.clip((x + r - w) // 2, 0, size[0] - w))
    top = int(np.clip((y + b - h) // 2, 0, size[1] - h))
    return [left, top, left + w, top + h]


def save_crop(img, box, dest):
    with NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        img.crop(box).save(tmp, format='WEBP', lossless=True, method=2)
    Path(tmp.name).chmod(0o644)
    Path(tmp.name).replace(dest)


def verify_saved(folder, report_path):
    report = read_json(report_path)
    for name in ('views', 'sources'):
        if sha256(folder / (name + '.json')) != report[name + '_sha256']:
            raise ValueError('Changed ' + name + ' manifest')
    for p, digest in read_json(folder / 'sources.json').items():
        if sha256(ROOT / p) != digest:
            raise ValueError('Changed verification input ' + p)
    rows = read_json(folder / 'views.json')

    @lru_cache(maxsize=8)
    def frame(path):
        with Image.open(path) as source:
            return source.convert('RGB')

    count = 0
    for key, data in sorted(rows.items(), key=lambda item: item[1]['case']):
        for number, shot in data['frames'].items():
            if shot['status'] != 'visible':
                continue
            x, y, r, b = shot['bbox']
            left, top, right, bottom = shot['crop']
            if not (left <= x < r <= right and top <= y < b <= bottom):
                raise ValueError(key + ' crop does not contain its projected bed')
            paths = [
                JOINT / f'original_{data["site"]}_{number}.jpg',
                JOINT / data['case'] / f'drive_{number}.jpg',
            ]
            for kind, path in zip(('before', 'after'), paths):
                crop = frame(path).crop(shot['crop'])
                with Image.open(folder / shot[kind]) as output:
                    if not np.array_equal(np.asarray(crop), np.asarray(output.convert('RGB'))):
                        raise ValueError(key + ' crop differs from paired vehicle frame')
                count += 1
    if count != 2 * report['observations']['visible']:
        raise ValueError('Incomplete paired crops')
    print(count, 'lossless crops match their original frame pixels')


def build(output, report_path):
    rows = {r['id']: r for r in json.loads((ATLAS / 'catalog.js').read_text()[18:-2])}
    assets = {}
    stats = Counter()
    sources = {}
    for casefile in sorted(JOINT.glob('scene_*/scene.json')):
        dossier = casefile.parent
        case = read_json(casefile)
        site = case['site']
        pl = case['placements']
        if [p['asset_id'] for p in pl] != case['assets']:
            raise ValueError(dossier.name + ' placement identity mismatch')
        measured = [
            (i, p) for i, p in enumerate(pl) if rows[p['asset_id']]['method'] in ('实拍 RGB-D', '实拍双目')
        ]
        if not measured:
            continue
        cams = {f: camera(site, f) for f in FRAMES}
        after = {f: np.array(Image.open(dossier / f'drive_{f}.jpg').convert('RGB')) for f in FRAMES}
        for f in FRAMES:
            data = PROCESSED / site
            for path in (
                casefile,
                JOINT / f'original_{site}_{f}.jpg',
                dossier / f'drive_{f}.jpg',
                data / 'intrinsics/0.txt',
                data / 'extrinsics' / f'{f:03d}_0.txt',
                data / 'dynamic_masks/all' / f'{f:03d}_0.png',
            ):
                name = str(path.relative_to(ROOT))
                if name not in sources:
                    sources[name] = sha256(path)
        if any(after[f].shape != cams[f][0].shape for f in FRAMES):
            raise ValueError('Camera image dimensions differ')
        thresholds = {}
        for f in FRAMES:
            upper = after[f].shape[0] // 3
            delta = abs(after[f][:upper].astype('i2') - cams[f][0][:upper]).mean(axis=2)
            thresholds[f] = max(8.0, float(np.quantile(delta, 0.999)))
        boxes = {
            f: [camera_bounds(corners(ATLAS / rows[p['asset_id']]['glb']), p, cams[f]) for p in pl]
            for f in FRAMES
        }
        joint = joint_surface(pl, rows)
        for index, p in measured:
            key = p['asset_id']
            row = rows[key]
            if key in assets:
                raise ValueError('Duplicate measured asset ' + key)
            bed = PitBed(ATLAS / row['glb'])
            if bed.kind != 'capture_annotation':
                raise ValueError('Missing source mask ' + key)
            frames = {}
            for f in FRAMES:
                shot = observation(bed, p, cams[f], after[f], boxes[f][index], index, joint, thresholds[f])
                stats[shot['status']] += 1
                if shot['status'] == 'visible':
                    crop = window(shot['bbox'], after[f].shape[1::-1])
                    shot['crop'] = crop
                    for kind, img in [('before', cams[f][0]), ('after', after[f])]:
                        dest = f'{key}-{f}-{kind}.webp'
                        save_crop(Image.fromarray(img), crop, output / dest)
                        shot[kind] = dest
                frames[str(f)] = shot
            assets[key] = dict(
                site=site,
                case=dossier.name,
                slot=p['slot'],
                glb_sha256=sha256(ATLAS / row['glb']),
                annotation_sha256=bed.annotation_sha,
                frames=frames,
            )
    if not assets:
        raise ValueError('No camera placements')
    expected = {k for k, r in rows.items() if r['method'] in ('实拍 RGB-D', '实拍双目')}
    if assets.keys() != expected:
        raise ValueError('Incomplete source inventory')
    visible = [k for k, a in assets.items() if any(f['status'] == 'visible' for f in a['frames'].values())]
    save_json(output / 'views.json', assets)
    save_json(output / 'sources.json', sources)
    report = dict(
        assets=len(assets),
        frames_per_asset=list(FRAMES),
        visible_assets=len(visible),
        visible_methods=dict(Counter(rows[k]['method'] for k in visible)),
        observations=dict(stats),
        gate=dict(
            minimum_eligible_pixels=12,
            minimum_changed_pixels=8,
            minimum_changed_fraction=0.2,
            minimum_mean_channel_difference_per_changed_pixel=8,
            control='At least the 99.9th percentile of the paired upper-third frame difference',
        ),
        views_sha256=sha256(output / 'views.json'),
        sources_sha256=sha256(output / 'sources.json'),
    )
    save_json(report_path, report)
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/capture-driving')
    parser.add_argument('--report', type=Path, default=ROOT / 'validation/capture-driving.json')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH if args.verify else fcntl.LOCK_EX)
        if args.verify:
            verify_saved(args.out, args.report)
        else:
            build(args.out, args.report)


if __name__ == '__main__':
    main()
