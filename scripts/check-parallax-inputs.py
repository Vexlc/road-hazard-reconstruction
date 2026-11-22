"""Repeat the RGB reconstruction from plain image files with all sensor archives out of scope."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole'))
from multiview_pits import build
from photo_parallax import fit_road, read_pair


def rejection(first, second, ruler, description):
    try:
        fit_road(first, second, ruler)
    except ValueError as error:
        return str(error)
    raise ValueError('Invalid input accepted: ' + description)


def check(run):
    cv2.setNumThreads(2)
    bundle = run / 'series'
    specs = json.loads((bundle / 'calibration/survey.json').read_text())
    original = json.loads((bundle / 'checks.json').read_text())
    result = {}
    with TemporaryDirectory(prefix='photo-pair-inputs-') as tmp:
        base = Path(tmp)
        empty = base / 'empty'
        empty.mkdir()
        independent = []
        for s in specs:
            key = s['id']
            target = base / key
            target.mkdir()
            frames = {}
            for name, suffix in zip(s['fit'], ['', '_second']):
                for source, filename in [
                    (bundle / 'references' / (key + suffix + '.png'), name + '.png'),
                    (bundle / 'calibration' / (key + suffix + '_outline.png'), name + '-mask.png'),
                ]:
                    (target / filename).write_bytes(source.read_bytes())
                frames[name] = dict(rgb=key + '/' + name + '.png', mask=key + '/' + name + '-mask.png')
            independent.append(dict(s, source='files', frames=frames))
        args = argparse.Namespace(
            survey=base / 'spec.json',
            config=ROOT / 'gen2hole/config/local.json',
            source=empty,
            catalog=base / 'unavailable-catalog.json',
            mesh_only=True,
            preview=False,
        )
        args.survey.write_text(json.dumps(independent))
        stage = base / 'rebuild'
        build(args, stage)
        rebuilt = json.loads((stage / 'series/checks.json').read_text())
        for s in independent:
            key = s['id']
            for field in ['asset.glb', 'surface.npz']:
                if original[key]['sha256'][field] != rebuilt[key]['sha256'][field]:
                    raise ValueError('File parity ' + key + ' ' + field)
            result[key] = dict(
                glb_sha256=rebuilt[key]['sha256']['asset.glb'],
                field_sha256=rebuilt[key]['sha256']['surface.npz'],
            )
        first = independent[0]
        photos = read_pair(first, empty, args.catalog, base)
        duplicate = rejection(photos[0], photos[0], first['ruler'], 'duplicated photograph')
        y, x = np.nonzero(photos[0].mask)
        middle = len(x) // 2
        bad_ruler = dict(
            first['ruler'], pixels=[[int(x[middle]), int(y[middle])], first['ruler']['pixels'][1]]
        )
        inside = rejection(*photos, bad_ruler, 'scale reference inside the pothole')
        different = None
        if len(independent) > 1:
            unrelated = read_pair(independent[1], empty, args.catalog, base)[1]
            different = rejection(photos[0], unrelated, first['ruler'], 'different potholes')
        pose, n, d, _ = fit_road(*photos, first['ruler'])
        other, n2, d2, _ = fit_road(
            *photos, dict(first['ruler'], length_m=first['ruler']['length_m'] * 1.5)
        )
        if not (
            np.allclose(pose[:3, :3], other[:3, :3])
            and np.allclose(n, n2)
            and np.allclose(pose[:3, 3] * 1.5, other[:3, 3])
            and np.isclose(d * 1.5, d2)
        ):
            raise ValueError('Metric camera scale does not follow the ruler')
    return dict(
        images_only=result,
        identical_view_rejected=duplicate,
        interior_scale_reference_rejected=inside,
        unrelated_pair_rejected=different,
        verified_camera_scale_multiplier=1.5,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(check(args.run), indent=2, allow_nan=False) + '\n')
