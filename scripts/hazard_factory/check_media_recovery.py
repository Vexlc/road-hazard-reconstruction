"""Interrupt a real renderer in an isolated fixture and repair a damaged orbit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import numpy as np

from run import atomic_json, HERE, sha


def main(args):
    job = json.loads(args.job.read_text(encoding='utf-8'))
    root = Path(job['output']).parent
    fixture = Path(job['work']) / 'media_recovery' / uuid.uuid4().hex
    scenes = fixture / '迁入场景'; scenes.mkdir(parents=True)
    work = fixture / 'work'; work.mkdir()
    # Use completed first-group inputs, never mutate the real native scenes.
    for site in job['configs']:
        (scenes / f'{site}_01').symlink_to(Path(job['output']) / f'{site}_01', target_is_directory=True)
    destination = fixture / '五类动态展示'; destination.mkdir()
    source = root / '五类动态展示'
    for path in source.iterdir():
        if path.is_file():os.link(path, destination / path.name)
    baseline = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
    original = {p.name: sha(p) for p in source.iterdir() if p.is_file()}
    test_job = dict(job, output=str(scenes), work=str(work))
    job_path = fixture / 'job.json'; atomic_json(job_path, test_job)
    env = dict(os.environ, HAZARD_FACTORY_JOB=str(job_path), PYTHONUTF8='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
    killed = False
    with (fixture / 'renderer.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, str(HERE / 'present.py')], env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 50
        while time.monotonic() < deadline and child.poll() is None:
            partials = list((destination / '.staging').glob('*/*.mp4'))
            if partials and partials[0].stat().st_size > 262144:
                partial_size = partials[0].stat().st_size
                os.killpg(child.pid, signal.SIGKILL); child.wait(); killed = True; break
            time.sleep(.1)
        if not killed:
            if child.poll() is None:os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise AssertionError(f'No running partial video observed; inspect {fixture}')
    assert all(sha(destination / name) == digest for name, digest in baseline.items())
    assert all(sha(source / name) == digest for name, digest in original.items())
    # The fixture renderer produced an actual CUDA orbit before interruption.
    from present import orbit, IW, IH, FRAMES
    from media_publish import cached_orbit
    path = next((work / 'orbits').glob('*.npy')); stamp = path.with_suffix('.json')
    metadata = json.loads(stamp.read_text(encoding='utf-8'))
    before = np.load(path, allow_pickle=False).copy()
    with path.open('r+b') as stream:
        stream.seek(-1, os.SEEK_END); last = stream.read(1)
        stream.seek(-1, os.SEEK_END); stream.write(bytes([last[0] ^ 255]))
    assert cached_orbit(path, stamp, metadata['source_and_framing_sha256'], (FRAMES, IH, IW, 3)) is None
    models = {m['id']:dict(m, ply=str(Path(lib) / m['ply'])) for lib in job['libraries']
              for m in json.loads((Path(lib) / 'asset_manifest.json').read_text(encoding='utf-8'))}
    repaired = orbit(models[path.stem], work / 'orbits')
    maximum_error = int(np.max(abs(before.astype('i2') - repaired.astype('i2'))))
    assert maximum_error <= 1, maximum_error
    new_meta = json.loads(stamp.read_text(encoding='utf-8'))
    assert new_meta['npy_sha256'] == sha(path)
    assert cached_orbit(path, stamp, new_meta['source_and_framing_sha256'], (FRAMES, IH, IW, 3)) is not None
    report = {'passed':True, 'real_renderer_process_group_interrupted':True,
              'observed_partial_mp4_bytes':partial_size,
              'published_files_byte_unchanged':len(baseline),
              'source_delivery_byte_unchanged':True,
              'damaged_orbit_rejected':True, 'actual_cuda_orbit_rebuilt':True,
              'rebuilt_max_pixel_error_255':maximum_error,
              'repaired_cache_hash_verified':True, 'fixture':str(fixture),
              'scope':'Real interrupted render plus content-corrupt orbit recovery. Full five-video successful publication is verified separately.'}
    atomic_json(root / '展示中断与缓存恢复验证.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    main(parser.parse_args())
