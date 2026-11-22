"""Fetch the pinned CC0 maps and source materials; verify before replacing or reusing a file."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

SURVEY = Path(__file__).with_name('survey.json')


def fetch(dest, spec):
    for slug, material in spec['materials'].items():
        base = dest / slug
        base.mkdir(parents=True, exist_ok=True)
        for entry in material['files'].values():
            path = base / entry['url'].split('/')[-1]
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']:
                continue
            with NamedTemporaryFile(dir=base, delete=False) as stream:
                draft = Path(stream.name)
            try:
                request = Request(entry['url'], headers={'User-Agent': 'RoadGS/1.0 public asset research'})
                with urlopen(request, timeout=90) as response, draft.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
                if (
                    draft.stat().st_size != entry['bytes']
                    or hashlib.sha256(draft.read_bytes()).hexdigest() != entry['sha256']
                ):
                    raise ValueError('Source hash mismatch: ' + path.name)
                draft.replace(path)
            finally:
                draft.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    fetch(a.out, json.loads(SURVEY.read_text()))
