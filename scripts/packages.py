#!/usr/bin/env python3
"""List, check and safely restore independently downloadable packages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from tempfile import NamedTemporaryFile
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified(path, record):
    return path.is_file() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']


def fetch_file(record, folder):
    target = folder / record['file']
    if verified(target, record):
        return target
    url = 'https://drive.usercontent.google.com/download?' + urlencode(
        {'id': record['id'], 'export': 'download', 'confirm': 't'}
    )
    for attempt in range(3):
        with NamedTemporaryFile(dir=folder, delete=False) as stream:
            draft = Path(stream.name)
        try:
            with urlopen(Request(url, headers={'User-Agent': 'RoadGS/1.0'}), timeout=90) as src:
                if src.headers.get_content_type() == 'text/html':
                    raise ValueError('Google Drive returned a page instead of the requested bytes')
                with draft.open('wb') as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)
            if not verified(draft, record):
                raise ValueError('Downloaded size or SHA-256 differs from the manifest')
            draft.replace(target)
            print('FETCHED', target.name, flush=True)
            return target
        except (HTTPException, OSError, ValueError) as exc:
            if attempt == 2:
                raise SystemExit(f"{target.name}: {exc}. Browser download: {record['url']}") from exc
            time.sleep(2 ** attempt)
        finally:
            draft.unlink(missing_ok=True)


def fetch(record, folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / record['file']
    if verified(target, record):
        return
    if 'parts' not in record:
        fetch_file(record, folder)
        return
    cache = folder / (target.name + '.parts')
    cache.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as workers:
        parts = list(workers.map(lambda r: fetch_file(r, cache), record['parts']))
    with NamedTemporaryFile(dir=folder, delete=False) as out:
        draft = Path(out.name)
    try:
        with draft.open('wb') as out:
            for part in parts:
                with part.open('rb') as src:
                    shutil.copyfileobj(src, out, 1024 * 1024)
        if not verified(draft, record):
            raise SystemExit('Assembled package checksum mismatch: ' + target.name)
        draft.replace(target)
    finally:
        draft.unlink(missing_ok=True)


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
    p.add_argument('operation', choices=['list', 'fetch', 'check', 'restore'])
    p.add_argument('--group', action='append')
    p.add_argument('--downloads', type=Path, default=ROOT / 'downloads')
    p.add_argument('--destination', type=Path, default=ROOT)
    args = p.parse_args()
    if args.operation == 'fetch' and not args.group:
        p.error('Select the download groups with --group.')
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
        if args.operation == 'fetch':
            fetch(r, args.downloads)
        path = args.downloads / r['file']
        if not path.is_file() or sha256(path) != r['sha256']:
            raise SystemExit('Missing file or SHA-256 mismatch: ' + str(path))
        print('VERIFIED', r['file'], flush=True)
        if args.operation == 'restore':
            restore(path, args.destination / r.get('restore_to', ''), r)
            print('RESTORED', r['file'], flush=True)


if __name__ == '__main__':
    main()
