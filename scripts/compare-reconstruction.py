#!/usr/bin/env python3
"""Check rebuilt geometry and Gaussian parameters with explicit numeric tolerances."""
import argparse
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--rebuilt', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((ROOT / 'outputs/road_hazard_expansion/asset_manifest.json').read_text())
    checks = []
    for row in manifest:
        new = args.rebuilt / row['metadata']
        if not new.exists():
            continue
        a = PlyData.read(args.reference / row['ply'])['vertex'].data
        b = PlyData.read(args.rebuilt / row['ply'])['vertex'].data
        fraction = abs(len(a) - len(b)) / len(a)
        assert fraction <= .0001, (row['id'], 'point-count fraction', fraction)
        count_before, count_after = len(a), len(b)
        unmatched = 0
        nearest_xy_max = 0.
        if len(a) != len(b):
            # Boundary alpha thresholds can add/remove individual grid points.
            # Match physical XY positions instead of shifting all array indices.
            query = np.stack([b[n] for n in ('x', 'y')], axis=1)
            reference = np.stack([a[n] for n in ('x', 'y')], axis=1)
            distances, indices = cKDTree(reference).query(query, workers=1)
            nearest_xy_max = float(distances.max())
            assert nearest_xy_max < .002
            common = distances < .00001
            unmatched = int((~common).sum())
            assert unmatched / len(b) < .0001
            # Newly visible boundary samples have no identical source sample;
            # their attributes are not comparable as an ordered vertex pair.
            a, b = a[indices[common]], b[common]
        groups = {
            'position_m': (['x', 'y', 'z'], .002, .0001),
            'log_scale': ([f'scale_{i}' for i in range(3)], .5, .01),
            'quaternion_component': ([f'rot_{i}' for i in range(4)], .3, .01),
            'sh_coefficient': ([f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(9)], .04, .001),
            'opacity_probability': (['opacity'], .01, .001),
        }
        record = {'id': row['id'], 'category': row['category'],
                  'points_reference': count_before, 'points_rebuilt': count_after,
                  'point_count_difference_fraction': fraction, 'checks': {}}
        record.update(unmatched_boundary_points=unmatched, nearest_xy_max_error_m=nearest_xy_max)
        for group, (names, tolerance, percentile_tolerance) in groups.items():
            if group == 'opacity_probability':
                errors = np.stack([np.abs(1/(1+np.exp(-a[n])) - 1/(1+np.exp(-b[n]))) for n in names])
            else:
                errors = np.stack([np.abs(a[n] - b[n]) for n in names])
            maximum = float(np.max(errors))
            p99 = float(np.percentile(errors, 99))
            assert np.isfinite(errors).all() and maximum <= tolerance and p99 <= percentile_tolerance, (row['id'], group, maximum, p99)
            record['checks'][group] = {'max_abs_error': maximum, 'p99_abs_error': p99,
                                      'max_tolerance': tolerance, 'p99_tolerance': percentile_tolerance, 'pass': True}
        with np.load(args.reference / row['surface']) as aa, np.load(args.rebuilt / row['surface']) as bb:
            assert aa['height'].shape == bb['height'].shape
            error = float(np.max(np.abs(aa['height'] - bb['height'])))
            rms = float(np.sqrt(np.mean((aa['height'] - bb['height']) ** 2)))
            assert error < .002 and rms < .0001
            record['checks']['height_m'] = {'max_abs_error': error, 'rms_error': rms,
                                            'max_tolerance': .002, 'rms_tolerance': .0001, 'pass': True}
        checks.append(record)
    assert checks, 'No rebuilt assets found'
    report = {'assets_checked': len(checks), 'passed': True, 'assets': checks}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'assets_checked': len(checks), 'passed': True}))


if __name__ == '__main__':
    main()
