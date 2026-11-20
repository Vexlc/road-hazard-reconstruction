"""Stage review media privately; retain the last complete videos during rendering."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

from run import atomic_json, sha


@contextmanager
def publication(destination, index_name):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stage = destination / '.staging' / uuid.uuid4().hex
    stage.mkdir(parents=True)
    atomic_json(stage / 'transaction.json', {'state': 'rendering', 'started_unix': time.time()})
    try:
        yield stage
        index_path = stage / index_name
        index = json.loads(index_path.read_text(encoding='utf-8'))
        files = sorted(p for p in stage.iterdir() if p.is_file()
                       and p.name not in (index_name, 'transaction.json'))
        if not files or not any(p.suffix == '.mp4' for p in files):
            raise ValueError('No completed video to publish')
        if any(p.stat().st_size == 0 for p in files):
            raise ValueError('Empty media in completed transaction')
        index['publication'] = {'schema': 1, 'transaction': stage.name,
                                'files_sha256': {p.name: sha(p) for p in files}}
        atomic_json(index_path, index)
        atomic_json(stage / 'transaction.json', {'state': 'prepared', 'files': [p.name for p in files]})
        # Files remain untouched until *all* writers close successfully. Each
        # file replacement is atomic; the hash-bearing index is published last.
        # Readers can detect the short multi-file switch with the index hashes.
        for path in files:
            os.replace(path, destination / path.name)
        os.replace(index_path, destination / index_name)
        atomic_json(stage / 'transaction.json', {'state': 'committed', 'completed_unix': time.time()})
    except BaseException as error:
        atomic_json(stage / 'transaction.json', {'state': 'interrupted', 'error': str(error)})
        raise


def cached_orbit(path, stamp, signature, shape):
    """Reject partial, malformed and content-corrupt cache entries before use."""
    import numpy as np
    try:
        meta = json.loads(stamp.read_text(encoding='utf-8'))
        if meta.get('source_and_framing_sha256') != signature:
            return None
        if meta.get('npy_sha256') != sha(path):
            return None
        frames = np.load(path, mmap_mode='r', allow_pickle=False)
        if frames.dtype != np.uint8 or frames.shape != shape:
            return None
        return frames
    except (OSError, ValueError, KeyError):
        return None


def save_orbit(path, stamp, frames, metadata):
    import numpy as np
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, frames, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json(stamp, dict(metadata, npy_sha256=sha(path)))
