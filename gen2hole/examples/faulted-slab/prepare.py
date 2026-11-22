"""Crop a calibrated pavement scan in its source coordinate system for a family of rigid slab states."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


def prepare(parent, survey, dest):
    base = survey['parent']
    spec = next(r for r in json.loads((parent / 'calibration/designs.json').read_text()) if r['id'] == base)
    x, y, n = survey['crop_col_row_intervals']
    step = spec['step']
    if (
        spec['depth_profile'] != 'scanned_relief'
        or min(x, y) < 0
        or n <= 0
        or any(i % step for i in (x, y, n))
        or max(x, y) + n > spec['intervals']
    ):
        raise ValueError('Choose an in-bounds, grid-aligned crop of registered scan samples')
    ref = dest / 'references'
    cal = dest / 'calibration'
    ref.mkdir(parents=True)
    cal.mkdir()
    bbox = (x, y, x + n + 1, y + n + 1)
    with (
        Image.open(parent / 'references' / (base + '.png')) as source,
        Image.open(parent / 'calibration' / (base + '_outline.png')) as mark,
    ):
        if source.size != mark.size or source.size != (spec['intervals'] + 1,) * 2:
            raise ValueError('Parent RGB and mask do not match the source grid')
        color, mask = source.crop(bbox), mark.crop(bbox)
    with np.load(parent / 'calibration' / (base + '_samples.npz')) as archive:
        heights = archive['height'][y // step : (y + n) // step + 1, x // step : (x + n) // step + 1]
    if heights.shape != (n // step + 1,) * 2:
        raise ValueError('Parent scan samples do not span the crop')
    designs = []
    for state in survey['states']:
        key = state['id']
        params = {
            k: survey[k]
            for k in (
                'seed_col_row_in_crop',
                'base_z_m',
                'slab_bottom_z_m',
                'socket_clearance_m',
                'support_inset_m',
            )
        }
        spec_out = dict(
            spec,
            id=key,
            label=state['label'],
            parent=base,
            intervals=n,
            span=n * spec['source_pitch_m'],
            origin_col_row=np.add(spec['origin_col_row'], [x, y]).tolist(),
            slab=dict(params, offset_m=state['offset_m']),
        )
        designs.append(spec_out)
        color.save(ref / (key + '.png'))
        mask.save(cal / (key + '_outline.png'))
        np.savez_compressed(cal / (key + '_samples.npz'), height=heights)
    (cal / 'designs.json').write_text(json.dumps(designs, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True, help='Prepared scanned-cracks input root')
    parser.add_argument('--survey', type=Path, default=Path(__file__).with_name('survey.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error('Choose a new output directory')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / 'inputs'
        prepare(args.parent, json.loads(args.survey.read_text()), stage)
        stage.rename(args.out)
