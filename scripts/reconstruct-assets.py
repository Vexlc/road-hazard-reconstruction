#!/usr/bin/env python3
"""Rebuild selected published asset IDs from the archived original inputs."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import os
import multiprocessing
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / 'outputs/road_hazard_expansion/reproduction'
CATEGORIES = ('stereo', 'rgbd', 'pothole', 'missing', 'raised')
RGBD_NAMES = None
CAPTURE_INPUTS = None


def mesh_job(job):
    global RGBD_NAMES, CAPTURE_INPUTS
    row, inputs, output, scratch = job
    name, cat = row['id'], row['category']
    meshes, samples = scratch / 'meshes', scratch / 'samples'
    sys.path.insert(0, str(PIPELINE))
    if cat == 'stereo':
        import batch_stereo
        result = batch_stereo.build((inputs / 'stereo' / f"dataset{row['source_dataset']}",
                                    output / cat, samples,
                                    row['source_dataset'], row['source_frame'], True))
        if 'rejected' in result:
            raise RuntimeError(result)
    elif cat == 'rgbd':
        import batch_rgbd
        if CAPTURE_INPUTS is None:
            CAPTURE_INPUTS = json.loads((ROOT / 'catalog/capture-inputs.json').read_text())
        if RGBD_NAMES is None:
            batch_rgbd.init_worker(inputs / 'rgbd')
            RGBD_NAMES = True
        # One timestamp can have several RGB/label variants. Reconstruct the
        # exact published member, not whichever variant sorts first today.
        expected = CAPTURE_INPUTS[name]
        cached = output / cat / name / 'capture_metadata.json'
        if cached.exists() and json.loads(cached.read_text()).get('inputs') != expected:
            raise ValueError('Cached output uses different input members; choose an empty --output: ' + name)
        for kind, archive in zip(('rgb', 'depth', 'mask'), batch_rgbd.ARCHIVES):
            payload = archive.read(expected[kind]['filename'])
            if hashlib.sha256(payload).hexdigest() != expected[kind]['sha256']:
                raise ValueError('Original input checksum mismatch: ' + name + '/' + kind)
        result = batch_rgbd.build((expected['rgb']['filename'], output / cat,
                                   scratch / 'rgbd', samples,
                                   row['pinhole_nominal_fx_fy_px']))
        if 'rejection' in result:
            raise RuntimeError(result)
    else:
        from make_variants import build
        from variant_catalog import variants
        design = next(r for r in variants(3) if r['id'] == name)
        build((design, inputs / 'pbr', output, meshes))
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--category', choices=CATEGORIES, action='append')
    p.add_argument('--asset', action='append', help='Exact ID; repeat for several assets.')
    p.add_argument('--sample', action='store_true', help='First published ID in each selected category.')
    p.add_argument('--output', type=Path, default=ROOT / 'work/rebuilt-assets')
    p.add_argument('--inputs', type=Path, default=ROOT / 'work/reconstruction-inputs')
    p.add_argument('--gaussian-python', type=Path,
                   default=ROOT / 'work/envs/streetcrafter/bin/python')
    p.add_argument('--mesh-only', action='store_true')
    p.add_argument('--workers', type=int, default=2, help='CPU mesh workers; GPU export is sequential.')
    args = p.parse_args()
    rows = json.loads((ROOT / 'outputs/road_hazard_expansion/asset_manifest.json').read_text())
    rows = sorted(rows, key=lambda r: (CATEGORIES.index(r['category']), r['id']))
    if args.category:
        rows = [r for r in rows if r['category'] in args.category]
    if args.asset:
        rows = [r for r in rows if r['id'] in args.asset]
        if {r['id'] for r in rows} != set(args.asset):
            p.error('An asset ID does not exist in the selected categories.')
    if args.sample:
        rows = [next(r for r in rows if r['category'] == cat)
                for cat in CATEGORIES if any(r['category'] == cat for r in rows)]
    if not rows:
        p.error('No assets selected.')
    if args.workers < 1:
        p.error('--workers must be positive.')
    args.output = args.output.resolve()
    args.inputs = args.inputs.resolve()
    scratch = args.output.parent / (args.output.name + '-scratch')
    meshes = scratch / 'meshes'
    samples = scratch / 'samples'
    for folder in (args.output, meshes, samples, scratch / 'rgbd'):
        folder.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (args.output / row['category']).mkdir(exist_ok=True)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    jobs = [(r, args.inputs, args.output, scratch) for r in rows]
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for name in pool.map(mesh_job, jobs):
            print('MESH', name, flush=True)
    if not args.mesh_only:
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([
            str(ROOT / 'work/asset_pipeline'), str(ROOT / 'work/road_hazard_pipeline'),
            env.get('PYTHONPATH', '')])
        env.setdefault('CUDA_HOME', '/usr/local/cuda')
        env.setdefault('OMP_NUM_THREADS', '4')
        command = [str(args.gaussian_python.resolve()), str(PIPELINE / 'export_gaussians.py'),
                   '--out', str(args.output), '--meshes', str(meshes), '--samples', str(samples), '--force']
        for row in rows:
            command += ['--asset', row['id']]
        subprocess.run(command, env=env, check=True)
    report = {'assets': [r['id'] for r in rows], 'categories': sorted({r['category'] for r in rows}),
              'inputs': str(args.inputs), 'output': str(args.output), 'gaussians': not args.mesh_only}
    (args.output / 'reconstruction-run.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
