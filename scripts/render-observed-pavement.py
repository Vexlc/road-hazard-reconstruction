"""Check visible surface regions in the two vehicle scenes and show the photographed assets."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import Surface, read_json, save_json, sha256

Plate = import_module('render-pit-survey').Plate


def owners(models, eye, rays, depths):
    result = np.zeros(len(rays), 'u1')
    if not len(rays):
        return result
    for i, mesh in enumerate(models, 1):
        z, _, _ = mesh.cast(np.broadcast_to(eye, rays.shape), rays)
        hit = np.isfinite(z) & (abs(z - depths) < 0.0003)
        if np.any(hit & (result > 0)):
            raise ValueError('Two labelled surfaces occupy the same first hit')
        result[hit] = i
    if np.any(result == 0):
        raise ValueError('A recorded return missed the isolated surface')
    return result


def observations(run, key):
    folder = run / 'evaluation' / key
    bundle = run / 'series'
    asset = bundle / 'assets/pothole' / key
    names = read_json(asset / 'metadata.json')['region_names']
    if names == ['Detached pavement top', 'Detached pavement edges']:
        codes = [13, 14]
        hues = [[35, 204, 181], [235, 153, 61]]
    elif names == ['Fracture inset rim']:
        codes = [19]
        hues = [[244, 174, 46]]
    else:
        codes = [18]
        hues = [[35, 204, 181], [114, 217, 230]] * (len(names) // 2 + 1)
    isolated = bundle / 'comparison' / key / 'isolated.glb'
    evidence = read_json(run / 'evaluation/measurements.json')[key]
    if sha256(isolated) != evidence['comparison_meshes']['isolated']['sha256']:
        raise ValueError('Isolated mesh changed after native range checks')
    models = [Surface(isolated, include_names={name}) for name in names]
    records = []
    lidar = {}
    for path in sorted((bundle / 'scenes').glob('*/scene.json')):
        scene = read_json(path)
        (place,) = scene['placements']
        if place['asset_id'] != key:
            continue
        case = scene['id']
        B = np.array(place['basis_columns'])
        center = np.array(place['center_world'])
        with np.load(run / 'sensors/cases' / (case + '.npz')) as scan:
            use = (scan['action'] == 1) & np.isin(scan['part'], codes) & (scan['instance'] == 1)
            xyz = (scan['xyz'][use] - center) @ B
        eye = xyz + [0, 0, 1.0]
        hits = owners(models, eye, np.tile([0.0, 0.0, -1.0], (len(eye), 1)), np.ones(len(eye)))
        lidar[case] = dict(zip(names, np.bincount(hits, minlength=len(names) + 1)[1:].tolist()))
        camera = run / 'sensors/camera' / case
        for shot in read_json(camera / 'views.json'):
            if not shot['available']:
                continue
            direction = shot['camera'].split('_')[0]
            with np.load(camera / shot['geometry']) as geom:
                mask = (
                    np.isin(geom['part'], codes)
                    & (geom['instance'] == shot['instance'])
                    & ~geom['protected'].astype(bool)
                )
                z = geom['depth_z_m'][mask]
            y, x = np.where(mask)
            l, t, _, _ = shot['crop_xyxy']
            pose = np.array(shot['camera_to_world'])
            rays = (
                np.c_[x + l + 0.5, y + t + 0.5, np.ones(len(x))]
                @ np.linalg.inv(shot['K_full']).T
                @ pose[:3, :3].T
                @ B
            )
            hit = owners(models, (pose[:3, 3] - center) @ B, rays, z)
            counts = np.bincount(hit, minlength=len(names) + 1)[1:]
            before = np.array(Image.open(camera / shot['after_stem']).convert('RGB'))
            tagged = before.copy()
            tagged[mask] = 0.18 * before[mask] + 0.82 * np.asarray(hues)[hit - 1]
            a, b, c, d = shot['xyxy_target_in_crop']
            h, w = mask.shape
            crop = (max(0, a - 12), max(0, b - 12), min(w, c + 13), min(h, d + 13))
            stem = case + '-' + direction
            for suffix, rgb in [('texture', before), ('parts', tagged)]:
                Image.fromarray(rgb).crop(crop).save(folder / (stem + '-' + suffix + '.png'))
            records.append(
                dict(
                    site=scene['site'],
                    case=case,
                    frame=shot['frame'],
                    direction='前视' if direction == 'front' else '后视',
                    part_pixels=int(mask.sum()),
                    per_mesh=dict(zip(names, counts.tolist())),
                    texture=stem + '-texture.png',
                    overlay=stem + '-parts.png',
                    crop_in_replay=crop,
                )
            )
    save_json(folder / 'road-parts.json', records)
    return dict(sha256=sha256(isolated), camera=records, lidar=lidar)


def portfolio(run, output):
    p = Plate(1490, 1410)
    p.text(28, 23, '铺砖缺失与坑内碎片', 44, True)
    p.text(29, 87, '两张实拍 RGB-D 分别重建两处病害。整坑、分色和单独取出的部位采用相同的网格与机位。', 23)
    measured = read_json(run / 'evaluation/measurements.json')
    proof = {}
    for i, (key, label) in enumerate([('P149', '联锁铺砖缺失'), ('P150', '坑内九块碎片')]):
        x = 27 + 734 * i
        folder = run / 'evaluation' / key
        proof[key] = observations(run, key)
        p.text(x, 155, key + '　' + label, 29, True)
        for j, (title, path) in enumerate(
            [
                ('原始彩照', run / 'series/references' / (key + '.png')),
                ('纹理网格', folder / 'textured-render.jpg'),
                ('可见部位分色', folder / 'labels-render.jpg'),
                ('同位置取出的部位', folder / 'isolated-render.jpg'),
            ]
        ):
            xx = x + (j % 2) * 353
            yy = 220 + (j // 2) * 291
            p.text(xx, yy, title, 20, True)
            p.picture(path, xx, yy + 34, 342, 237)
        v = measured[key]
        p.text(x, 807, '同机位预留量程', 23, True)
        p.text(
            x,
            850,
            f'{v["reserved_pit_pixels"]:,} 个坑内像素　·　绝对误差中位数 {v["withheld_mesh"]["median_mm"]:.3f} 毫米',
            20,
        )
        p.text(x, 890, '黄色为 18 毫米宽的缺口内缘。' if i == 0 else '青色为人工沿照片划出的碎片可见面。', 19)
        for j, site in enumerate(['034', '109']):
            obs = next(s for s in proof[key]['camera'] if s['site'] == site and s['direction'] == '前视')
            y = 956 + j * 208
            p.text(
                x,
                y,
                f'PandaSet {site}　前视帧 {obs["frame"]}　·　部位覆盖 {obs["part_pixels"]:,} 个像素',
                20,
                True,
            )
            for col, name in enumerate(['texture', 'overlay']):
                p.picture(folder / obs[name], x + 353 * col, y + 39, 342, 124)
        p.text(x, 1371, '每行左侧为迁入外观，右侧为同像素的模型部位回投。', 17)
    p.save(output)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    a = parser.parse_args()
    save_json(a.report, portfolio(a.run, a.out))
