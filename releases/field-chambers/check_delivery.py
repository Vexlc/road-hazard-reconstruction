"""Check the five files in a rebuilt chamber against the published reference hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', choices=('M544', 'M545'))
    parser.add_argument('folder', type=Path, help='Directory containing asset.glb')
    parser.add_argument('--mesh-only', action='store_true', help='Compare GLB and surface field')
    args = parser.parse_args()
    reports = Path(__file__).resolve().parent / 'validation'
    if args.asset == 'M544':
        row = json.loads((reports / 'rusted-cover-files.json').read_text())['native']
    else:
        row = json.loads((reports / 'M545-files.json').read_text())
    failures = []
    for name, expected in row['sha256'].items():
        if args.mesh_only and name not in ('asset.glb', 'surface.npz'):
            continue
        path = args.folder / name
        actual = digest(path) if path.is_file() else 'missing'
        print(name, actual)
        if actual != expected:
            failures.append(name)
    if failures:
        raise SystemExit('Reference mismatch: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
