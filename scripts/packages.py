#!/usr/bin/env python3
"""List, check and safely restore independently downloadable packages."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe archive entry: ' + name)
    return p


def restore(path, dest, record=None):
    if record and record.get('format') == 'raw':
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / record.get('restore_name', path.name))
    elif path.suffix.lower() in {'.mp4', '.gif'}:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / path.name)
    elif path.suffix.lower() == '.zip':
        # NPZ files inside a zstd tar can leave a ZIP end record near the end
        # of the outer file. is_zipfile() alone then misclassifies the archive.
        with path.open('rb') as stream:
            if stream.read(4) not in (b'PK\x03\x04', b'PK\x05\x06'):
                raise ValueError('ZIP extension without a ZIP header: ' + str(path))
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                safe(info.filename)
                if info.filename in {'asset_manifest.json', 'README.md'}:
                    # Each old volume has a partial manifest; retain the complete
                    # versioned manifest shipped with the source checkout.
                    continue
                if info.is_dir():
                    (dest / info.filename).mkdir(parents=True, exist_ok=True)
                else:
                    target = dest / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output)
    elif path.name.lower().endswith(('.tar.zst', '.tzst')):
        with path.open('rb') as stream:
            if stream.read(4) != b'\x28\xb5\x2f\xfd':
                raise ValueError('zstd extension without a zstd frame: ' + str(path))
        # Streaming zstd avoids another full uncompressed archive on disk.
        with subprocess.Popen(['zstd', '-q', '-d', '-c', str(path)], stdout=subprocess.PIPE) as proc:
            with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
                for member in archive:
                    safe(member.name)
                    if not (member.isfile() or member.isdir()):
                        raise ValueError('Links/devices are not accepted: ' + member.name)
                    target = dest / member.name
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, target.open('wb') as output:
                            shutil.copyfileobj(source, output)
            if proc.wait():
                raise RuntimeError('zstd failed')
    else:
        raise ValueError('Unsupported package format: ' + str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['list', 'check', 'restore'])
    p.add_argument('--group', action='append')
    p.add_argument('--downloads', type=Path, default=ROOT / 'downloads')
    p.add_argument('--destination', type=Path, default=ROOT)
    args = p.parse_args()
    records = json.loads((ROOT / 'catalog/downloads.json').read_text())['packages']
    if args.group:
        known = {r['group'] for r in records}
        if set(args.group) - known:
            p.error('Unknown package group(s): ' + ', '.join(set(args.group) - known))
        records = [r for r in records if r['group'] in args.group]
    for r in records:
        if args.operation == 'list':
            print(f"{r['group']:22} {r['bytes']/1024**2:9.1f} MiB  {r['file']}\n  {r.get('url', 'not uploaded')}")
            continue
        path = args.downloads / r['file']
        if not path.is_file() or sha256(path) != r['sha256']:
            raise SystemExit('Missing file or SHA-256 mismatch: ' + str(path))
        print('VERIFIED', r['file'], flush=True)
        if args.operation == 'restore':
            restore(path, args.destination / r.get('restore_to', ''), r)
            print('RESTORED', r['file'], flush=True)


if __name__ == '__main__':
    main()
