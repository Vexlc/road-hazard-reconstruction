#!/usr/bin/env python3
"""Transactional, appendable metric road-hazard generation. Standard library launcher."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from recipes import canonical, recipe

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def fingerprint():
    source = PROJECT / 'outputs/road_hazard_expansion/reproduction'
    files = [*(HERE/name for name in ('run.py', 'recipes.py', 'geometry.py', 'quality.py')),
             source/'make_variants.py', source/'variant_catalog.py',
             source/'export_gaussians.py', source/'verify_library.py',
             PROJECT/'work/asset_pipeline/gaussian_assets.py',
             PROJECT/'work/road_hazard_pipeline/sample_hazards.py']
    files += sorted((PROJECT/'work/road_hazard_sources').glob('*/*_Color.png'))
    files += sorted((PROJECT/'work/road_hazard_sources').glob('*/*_Displacement.png'))
    hashes = {str(p.relative_to(PROJECT)): sha(p) for p in files}
    return {'sha256': hashlib.sha256(canonical(hashes).encode()).hexdigest(), 'files': hashes}


def valid(directory, expected, fingerprint_hash):
    try:
        done = json.loads((directory/'complete.json').read_text(encoding='utf-8'))
        if done['recipe'] != expected or done['pipeline_sha256'] != fingerprint_hash:
            return False
        return all((directory/name).is_file() and sha(directory/name) == digest
                   for name, digest in done['files'].items()) and bool(done['files'])
    except (OSError, KeyError, ValueError):
        return False


def run(args):
    sys.path.insert(0, str(PROJECT/'outputs/road_hazard_expansion/reproduction'))
    from variant_catalog import variants
    catalog = list(variants())
    output, work = args.output.resolve(), args.work.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    (work/'logs').mkdir(exist_ok=True)
    (output/'.staging').mkdir(exist_ok=True)
    (output/'.quarantine').mkdir(exist_ok=True)
    cache = work/'meshes'
    cache.mkdir(exist_ok=True)
    source = fingerprint()
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               PYTHONPATH=':'.join(str(PROJECT/p) for p in (
                   'work/asset_pipeline', 'work/road_hazard_pipeline',
                   'outputs/road_hazard_expansion/reproduction')),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
               CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='8.9')
    mesh_python = PROJECT/'work/envs/assets/bin/python'
    gpu_python = PROJECT/'work/envs/streetcrafter/bin/python'
    outcome = []
    for index in range(args.start, args.start+args.count):
        cfg = recipe(args.seed, index, catalog)
        name, category = cfg['id'], cfg['category']
        dest = output/category/name
        dest.parent.mkdir(exist_ok=True)
        with lock(work/'locks'/f'{name}.lock'):
            if valid(dest, cfg, source['sha256']):
                print('VERIFIED CACHE', name, flush=True)
                outcome.append({'id': name, 'status': 'cached'})
                continue
            error = None
            for attempt in range(args.retries+1):
                stage = output/'.staging'/uuid.uuid4().hex/category/name
                stage.mkdir(parents=True)
                stage_cache = work/'staging_meshes'/stage.parents[1].name
                stage_cache.mkdir(parents=True)
                recipe_file = stage/'recipe.json'
                atomic_json(recipe_file, cfg)
                log = work/'logs'/f'{name}_{int(time.time())}_{attempt}.log'
                try:
                    commands = [
                        [mesh_python, HERE/'geometry.py', PROJECT, recipe_file, stage, stage_cache],
                        [gpu_python, PROJECT/'outputs/road_hazard_expansion/reproduction/export_gaussians.py',
                         '--out', stage.parents[1], '--meshes', stage_cache, '--samples', work, '--asset', name],
                        [gpu_python, HERE/'quality.py', PROJECT, stage, stage_cache, recipe_file],
                    ]
                    with log.open('w', encoding='utf-8') as stream:
                        for command in commands:
                            result = subprocess.run(list(map(str, command)), cwd=PROJECT, env=env,
                                                    stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
                            if result.returncode:
                                raise RuntimeError(f'{Path(command[1]).name} exited {result.returncode}; {log}')
                    files = {p.name: sha(p) for p in stage.iterdir() if p.is_file()}
                    os.replace(stage_cache/(name+'.npz'), cache/(name+'.npz'))
                    atomic_json(stage/'complete.json', {'recipe': cfg, 'pipeline_sha256': source['sha256'],
                                'files': files, 'completed_unix': time.time(),
                                'runtime': {'mesh_python': str(mesh_python), 'gaussian_python': str(gpu_python)}})
                    if dest.exists():
                        os.replace(dest, output/'.quarantine'/f'{name}_{uuid.uuid4().hex}')
                    os.replace(stage, dest)
                    print('PUBLISHED', name, flush=True)
                    outcome.append({'id': name, 'status': 'built', 'attempt': attempt+1})
                    error = None
                    break
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    error = str(exc)
                    print('FAILED', name, attempt+1, error, flush=True)
                    # Preserve logs and partial artifacts for diagnosis; never publish partial assets.
                    atomic_json(stage/'failure.json', {'error': error, 'attempt': attempt+1})
            if error:
                outcome.append({'id': name, 'status': 'failed', 'error': error})
    # Rebuild the registry from verified committed directories, including earlier seed streams.
    with lock(work/'locks/registry.lock'):
        manifest = []
        committed = [p for category in ('pothole', 'missing', 'raised')
                     for p in (output/category).glob('*/complete.json')]
        for complete in sorted(committed):
            done = json.loads(complete.read_text(encoding='utf-8'))
            directory = complete.parent
            if not valid(directory, done['recipe'], source['sha256']):
                continue
            meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            for key, file in [('ply', 'asset.ply'), ('glb', 'asset.glb'), ('surface', 'surface.npz'),
                              ('metadata', 'metadata.json'), ('preview', 'preview.jpg')]:
                meta[key] = str((directory/file).relative_to(output))
            meta['quality'] = json.loads((directory/'quality.json').read_text(encoding='utf-8'))
            meta['pipeline_sha256'] = source['sha256']
            manifest.append(meta)
        atomic_json(output/'asset_manifest.json', manifest)
        atomic_json(output/'pipeline_provenance.json', source)
        atomic_json(work/'last_run.json', {'assets': outcome, 'registry_count': len(manifest),
                                         'pipeline_sha256': source['sha256']})
    if any(x['status'] == 'failed' for x in outcome):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261120)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--retries', type=int, default=1)
    args = p.parse_args()
    if args.start < 0 or args.count < 1 or args.retries < 0:
        p.error('start/retries must be nonnegative and count positive')
    run(args)
