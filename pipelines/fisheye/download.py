"""Fetch the published reduced-resolution views and COLMAP model, not the feature DB."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen


REPO = 'nerfstudioteam/datasets'
REVISION = '461701c17e83c3f4d2481db32315aa7df703d2f8'
SCENES = ('dozer', 'redwoods2')


def listing(scene, folder):
    url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{scene}/{folder}?limit=1000'
    with urlopen(url, timeout=45) as response:
        return json.load(response)


def fetch(entry, root):
    name = entry['path']
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = entry.get('lfs', {}).get('oid')

    def valid(path):
        if not path.exists() or path.stat().st_size != entry['size']:
            return False
        with path.open('rb') as stream:
            if digest:
                return hashlib.file_digest(stream, 'sha256').hexdigest() == digest
            blob = hashlib.sha1(f'blob {entry["size"]}\0'.encode())
            blob.update(stream.read())
            return blob.hexdigest() == entry['oid']

    if valid(target):
        return
    for attempt in range(4):
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}?download=true&attempt={attempt}'
            with urlopen(url, timeout=90) as response, temp_path.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            if not valid(temp_path):
                raise ValueError(f'Invalid download: {name}')
            temp_path.replace(target)
            return
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
        finally:
            temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('scene', choices=SCENES)
    args = parser.parse_args()
    entries = [item for item in listing(args.scene, 'colmap/sparse/0') if item['path'].endswith('.bin')]
    entries += [item for item in listing(args.scene, 'images_4') if item['type'] == 'file']
    print(f'{args.scene}: fetching {len(entries)} files ({sum(item["size"] for item in entries) / 1e6:.1f} MB)', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: fetch(item, args.root), entries))
    print(f'{args.scene}: verified all file hashes', flush=True)


if __name__ == '__main__':
    main()
