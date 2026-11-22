"""Compare native road appearances from the two PLYs while testing identical poses, geometry and laser rows."""

import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image

transfer = import_module('check-pit-transfer')


def compare(baseline, refined):
    new = transfer.check(refined)
    rows = {}
    for case, record in new['scenes'].items():
        roots = [p / 'sensors' for p in (baseline, refined)]
        scenes = [
            transfer.read(p / 'series/scenes' / case / 'scene.json')['placements'][0]
            for p in (baseline, refined)
        ]
        fields = (
            'asset_id',
            'center_world',
            'basis_columns',
            'slot',
            'size_multiplier',
            'source_reference_frame',
        )
        if (
            any(scenes[0][f] != scenes[1][f] for f in fields)
            or scenes[0]['asset_ply_sha256'] == scenes[1]['asset_ply_sha256']
        ):
            raise ValueError('Sampling comparison needs two different PLYs at one physical placement')
        scans = [r / 'cases' / (case + '.npz') for r in roots]
        with np.load(scans[0]) as a, np.load(scans[1]) as b:
            if a.files != b.files or any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a.files):
                raise ValueError('Native laser replay changed in ' + case)
            laser_rows = len(a['row'])
        folders = [r / 'camera' / case for r in roots]
        views = [
            {s['camera']: s for s in transfer.read(p / 'views.json') if s['available']} for p in folders
        ]
        if views[0].keys() != views[1].keys():
            raise ValueError('Native camera coverage changed in ' + case)
        cameras = {}
        for cam, base in views[0].items():
            fine = views[1][cam]
            for k in ('frame', 'K_full', 'camera_to_world', 'srgb_gain_by_instance', 'crop_xyxy'):
                if base[k] != fine[k]:
                    raise ValueError(f'{case}/{cam} changed {k}')
            if transfer.digest(folders[0] / base['before_stem']) != transfer.digest(
                folders[1] / fine['before_stem']
            ):
                raise ValueError(f'{case}/{cam} changed the source photograph')
            with np.load(folders[0] / base['geometry']) as a, np.load(folders[1] / fine['geometry']) as b:
                fields = ('depth_z_m', 'part', 'instance', 'world_normal', 'sun_shadow', 'protected')
                if any(not np.array_equal(a[k], b[k], equal_nan=True) for k in fields):
                    raise ValueError(f'{case}/{cam} changed mesh projection or foreground')
                part = a['part']
                opacity = [a['composite_alpha'], b['composite_alpha']]
            photos = []
            for p, obs in zip(folders, (base, fine)):
                with Image.open(p / obs['after_stem']) as im:
                    photos.append(np.asarray(im.convert('RGB'), dtype='i2'))
            diff = abs(photos[0] - photos[1]).mean(axis=2)
            components = {}
            for name, mask in [
                ('all_slab_parts', part >= 13),
                ('slab_top', part == 13),
                ('edges_or_socket', np.isin(part, [14, 17])),
                ('base', part == 16),
            ]:
                if mask.any():
                    components[name] = dict(
                        pixels=int(mask.sum()),
                        rgb_mae_0_255=float(diff[mask].mean()),
                        rgb_p95_0_255=float(np.percentile(diff[mask], 95)),
                        area_alpha_mean_0_255=float(opacity[0][mask].mean()),
                        edge_alpha_mean_0_255=float(opacity[1][mask].mean()),
                    )
            cameras[cam] = dict(frame=base['frame'], crop=base['crop_xyxy'], parts=components)
        rows[case] = dict(
            ply_sha256=scenes[1]['asset_ply_sha256'],
            asset=record['asset'],
            identical_lidar_delta_rows=laser_rows,
            camera=cameras,
        )
    return dict(native_transfer=new, comparison=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('baseline', 'refined', 'out'):
        p.add_argument('--' + k, type=Path, required=True)
    args = p.parse_args()
    report = compare(args.baseline, args.refined)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    print(len(report['comparison']), 'road placements')
