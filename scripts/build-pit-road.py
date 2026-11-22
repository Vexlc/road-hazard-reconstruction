"""Prepare compact local road cutouts from each pit or crack mesh for the Gaussian viewer."""

import argparse
import base64
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
from scipy import ndimage

from pit_road import ATLAS, ROOT, PitBed
from road_geometry import sha256, save_json

PREFIX = 'document.currentScript.roadProfile='


def build(row, out):
    path = ATLAS / row['glb']
    bed = PitBed(path)
    span = (bed.high - bed.low)[:2].astype('f8')
    # Thin crack branches need more samples. Including the two border texels
    # keeps both sizes within WebGL 2's minimum supported texture dimension.
    long_side = 2046 if row['category'] == '裂缝' else 768
    w, h = np.maximum(4, np.rint(long_side * span / span.max()).astype(int))
    pitch = span / [w, h]
    # Include a complete row of undisturbed road outside every mesh boundary.
    x = bed.low[0] + (np.arange(w + 2) - 0.5) * pitch[0]
    y = bed.low[1] + (np.arange(h + 2) - 0.5) * pitch[1]
    xy = np.stack(np.meshgrid(x, y), axis=-1)
    z, labeled = bed.top(xy)
    negative = np.isfinite(z) & (z < 0)
    if labeled is not None:
        negative &= labeled
    if not negative.any():
        raise ValueError(row['id'] + ' has no observed subgrade relief')
    depth = float(np.percentile(-z[negative], 98))
    # The small offset preserves the source surface grains, scaling down for hairline cracks.
    plane = -min(0.003, 0.05 * depth)
    mask = bed.footprint(z, labeled, plane)
    if not mask.any():
        raise ValueError(row['id'] + ' empty pit opening')
    # Feather only the photographed surround. Preserve every pixel inside the opening.
    band = float(np.clip(0.07 * np.sqrt(mask.sum() * pitch.prod()), 0.004, 0.065))
    outside_distance = ndimage.distance_transform_edt(~mask, sampling=pitch[::-1])
    t = np.clip(1 - outside_distance / band, 0, 1)
    blend = np.rint(31 * t * t * (3 - 2 * t)).astype('u1')
    blend[mask] = 32
    bitmap = Image.frombytes('P', blend.shape[::-1], blend.tobytes())
    bitmap.putpalette([c for i in range(32) for c in (0, round(255 * i / 31), 0)] + [255, 255, 0])
    buf = BytesIO()
    bitmap.save(buf, format='PNG', compress_level=6)
    signature = sha256(path)
    if row.get('gaussianSources') and row['gaussianSources']['glb'] != signature:
        raise ValueError(row['id'] + ' atlas GLB fingerprint is stale')
    contact = bed.kind
    profile = dict(
        glb=signature,
        kind=contact,
        roadHeight=plane,
        probeHeight=-0.5 * depth,
        bounds=[
            float(x[0] - pitch[0] / 2),
            float(y[0] - pitch[1] / 2),
            float((w + 2) * pitch[0]),
            float((h + 2) * pitch[1]),
        ],
        blendWidth=band,
        mask='data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii'),
    )
    if bed.annotation is not None:
        profile['annotation_sha256'] = bed.annotation_sha
    dest = out / (row['id'] + '.js')
    with NamedTemporaryFile(dir=out, mode='w', encoding='utf8', delete=False) as stream:
        stream.write(PREFIX + json.dumps(profile, separators=(',', ':'), allow_nan=False) + ';\n')
    tmp = Path(stream.name)
    tmp.chmod(0o644)
    tmp.replace(dest)
    bed_signature = hashlib.sha256(
        json.dumps(
            [profile[k] for k in ('kind', 'roadHeight', 'probeHeight', 'bounds', 'blendWidth', 'mask')]
        ).encode()
    ).hexdigest()
    return dict(
        id=row['id'],
        method=row['method'],
        contact=contact,
        mask_pixels=int(mask.sum()),
        bed_signature=bed_signature,
        profile_sha256=sha256(dest),
        pixel_pitch_mm=float(max(pitch) * 1000),
        profile_bytes=dest.stat().st_size,
        registration_mm=bed.registration_mm,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/atlas-pit-road/profiles')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rows = json.loads((ATLAS / 'catalog.js').read_text().removeprefix('window.roadAssets=')[:-2])
    pits = [
        r
        for r in rows
        if r['category'] in ('坑洞', '方形浅坑', '裂缝', '混凝土碎板') and (not args.ids or r['id'] in args.ids)
    ]
    if not pits or (args.ids and set(args.ids) != {r['id'] for r in pits}):
        raise ValueError('Unknown pit asset')
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(lambda row: build(row, args.out), pits):
            results.append(r)
            if len(results) % 100 == 0:
                print(len(results), '/', len(pits), flush=True)
    report = dict(
        assets=len(results),
        smallest_opening_pixels=min(r['mask_pixels'] for r in results),
        profiles_sha256=hashlib.sha256(
            ''.join(
                r['id'] + ':' + r['profile_sha256'] + '\n' for r in sorted(results, key=lambda r: r['id'])
            ).encode()
        ).hexdigest(),
        methods=dict(Counter(r['method'] for r in results)),
        cutouts=dict(Counter(r['contact'] for r in results)),
        profile_bytes=sum(r['profile_bytes'] for r in results),
        max_pitch_mm=max(r['pixel_pitch_mm'] for r in results),
        max_profile_bytes=max(r['profile_bytes'] for r in results),
        capture_grid_max_height_error_mm=max(
            (r['registration_mm'] for r in results if r['registration_mm'] is not None), default=None
        ),
    )
    by_key = {r['id']: r['bed_signature'] for r in results}
    wet_groups = sorted({tuple(r['states']) for r in pits if r.get('water')})
    report['shared_water_beds'] = []
    for group in wet_groups:
        if all(k in by_key for k in group):
            if len({by_key[k] for k in group}) != 1:
                raise ValueError('The dry and wet versions have different road cutouts: ' + str(group))
            report['shared_water_beds'].append(list(group))
    print(report)
    if args.report:
        save_json(args.report, report)


if __name__ == '__main__':
    main()
