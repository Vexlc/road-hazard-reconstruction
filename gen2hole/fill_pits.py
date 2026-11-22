"""Build standing-water states from a reconstructed metric bed and a road anchor or explicit up vector."""

import argparse
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from gen2hole import ROOT, load, run_step, save, sha


def build(args, stage):
    survey = load(args.survey)
    keys = [s['id'] for s in survey['states']]
    if not keys or len(set(keys)) != len(keys) or any(not re.fullmatch(r'P[0-9]{2,}', k) for k in keys):
        raise ValueError('Specify unique pothole IDs')
    cfg = load(args.config)
    bundle = stage / 'series'
    refs, cal, rebuild = (bundle / k for k in ('references', 'calibration', 'rebuild'))
    refs.mkdir(parents=True)
    cal.mkdir()
    for name in ('common', 'series'):
        shutil.copytree(
            ROOT / 'recipes' / name,
            rebuild,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )
    save(cal / 'survey.json', survey)
    for s in survey['states']:
        key, bed, site = s['id'], s['bed'], s['site']
        source = survey['beds'][bed]
        base = args.runs / source['run'] / 'series'
        meta = load(base / 'calibration' / (bed + '.json'))
        anchor = survey['anchors'][site]
        if isinstance(anchor, list):
            up = anchor
            provenance = dict(up_source='Supplied world up in local asset coordinates')
        else:
            scene_file = args.runs / anchor
            scene = load(scene_file)
            (place,) = [p for p in scene['placements'] if p['category'] == 'pothole']
            if scene['site'] != site or place['size_multiplier'] != 1:
                raise ValueError('Anchor site/scale differs from the requested site')
            up = place['basis_columns'][2]
            provenance = dict(
                up_source='World Z expressed in the basis of the fitted road anchor',
                anchor_sha256=sha(scene_file),
                slot=place['slot'],
            )
        for suffix in ('.png', '_ortho.png', '_pavement.jpg'):
            shutil.copyfile(base / 'references' / (bed + suffix), refs / (key + suffix))
        for suffix in ('.npz', '_capture.json'):
            shutil.copyfile(base / 'calibration' / (bed + suffix), cal / (key + suffix))
        water = dict(
            input_liters=s['input_liters'],
            seed_uv=s.get('seed_uv', source.get('seed_uv')),
            site=site,
            up_local=up,
            **provenance,
        )
        meta.update(
            id=key,
            label=s['label'],
            state='standing_water',
            family=bed,
            dry_source_id=bed,
            dry_glb_sha256=sha(base / 'assets/pothole' / bed / 'asset.glb'),
            source_image='references/' + key + '.png',
            measurement_record='calibration/' + key + '_capture.json',
            water=water,
            recipe_version='measured-bed-standing-water-v1',
        )
        save(cal / (key + '.json'), meta)
    save(
        bundle / 'run.json',
        dict(
            edition='series',
            calibration='standing_water',
            ids=keys,
            published={},
            mesh_only=args.mesh_only,
            preview=args.preview,
            reference_sha256={p.name: sha(p) for p in refs.iterdir()},
        ),
    )
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='3',
        MKL_NUM_THREADS='3',
        PYTHONDONTWRITEBYTECODE='1',
    )
    run_step(cfg['mesh_python'], rebuild / 'fused_pit_mesh.py', keys, env)
    if not args.mesh_only:
        run_step(
            cfg['compute_python'],
            rebuild / 'export_splats.py',
            ['--cache', bundle / 'mesh_cache', *keys],
            env,
        )
    if args.preview:
        run_step(
            cfg['mesh_python'],
            rebuild / 'render_mesh.py',
            ['--cache', bundle / 'mesh_cache', '--out', bundle / 'views', *keys],
            env,
        )
    run_step(cfg['mesh_python'], rebuild / 'verify.py', [], env)
    save(stage / 'run.json', dict(editions=['series'], assets=keys))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--survey', required=True, type=Path)
    p.add_argument('--runs', type=Path, default=ROOT.parent / 'runs')
    p.add_argument('--config', type=Path, default=ROOT / 'config/local.json')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--mesh-only', action='store_true')
    args = p.parse_args()
    if args.out.exists() or args.out.is_symlink():
        p.error('Choose a new run directory')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.standing-water-', dir=args.out.parent) as tmp:
        stage = Path(tmp) / 'run'
        build(args, stage)
        stage.rename(args.out)
