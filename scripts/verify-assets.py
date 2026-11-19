#!/usr/bin/env python3
"""Verify full or selectively downloaded asset sets against the published hashes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, default=ROOT / 'outputs/road_hazard_expansion')
    p.add_argument('--category', action='append')
    p.add_argument('--asset', action='append')
    p.add_argument('--report', type=Path)
    p.add_argument('--allow-missing', action='store_true', help='Verify only IDs with a metadata.json.')
    p.add_argument('--mesh-only', action='store_true')
    args = p.parse_args()
    manifest = json.loads((ROOT / 'outputs/road_hazard_expansion/asset_manifest.json').read_text())
    if args.category:
        manifest = [r for r in manifest if r['category'] in args.category]
    if args.asset:
        manifest = [r for r in manifest if r['id'] in args.asset]
        if {r['id'] for r in manifest} != set(args.asset):
            p.error('Unknown or mismatched asset IDs.')
    checked, failures, omitted = [], [], []
    for row in manifest:
        folder = args.library / row['category'] / row['id']
        if args.allow_missing and not (folder / 'metadata.json').exists():
            omitted.append(row['id'])
            continue
        record = {'id': row['id'], 'category': row['category'], 'files': {}}
        for kind in (['glb'] if args.mesh_only else ['glb', 'ply']):
            file = args.library / row[kind]
            actual = digest(file) if file.is_file() else None
            passed = actual == row[kind + '_sha256']
            record['files'][kind] = {'sha256': actual, 'expected': row[kind + '_sha256'], 'match': passed}
            if not passed:
                failures.append({'id': row['id'], 'kind': kind, 'actual': actual})
        for name in ['surface.npz', 'metadata.json']:
            if not (folder / name).is_file():
                failures.append({'id': row['id'], 'missing': name})
        checked.append(record)
    result = {'checked': len(checked), 'omitted': len(omitted), 'failures': failures, 'assets': checked}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['checked', 'omitted', 'failures']}, indent=2))
    if failures or not checked:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
