"""Reconstruct potholes from registered RGB-D, stereo captures or a pair of RGB photographs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from fuse_pit import fuse, photo_texture, road_texture
from gen2hole import ROOT, run_step
from pit_captures import read_views, registration
from photo_parallax import build_pair


def put(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def build(args, stage):
    specs = json.loads(args.survey.read_text())
    keys = [s['id'] for s in specs]
    if len(set(keys)) != len(keys) or not keys or any(not re.fullmatch(r'P[0-9]{2,}', k) for k in keys):
        raise ValueError('Use a distinct pothole ID for each asset')
    photo_pairs = sum(s.get('mode') == 'rgb_parallax' for s in specs)
    calibration_mode = 'multiview_depth'
    if photo_pairs:
        calibration_mode = 'rgb_parallax' if photo_pairs == len(specs) else 'mixed_multiview'
    cfg = json.loads(args.config.read_text())
    bundle = stage / 'series'
    cal = bundle / 'calibration'
    refs = bundle / 'references'
    refs.mkdir(parents=True)
    cal.mkdir()
    put(cal / 'survey.json', specs)
    rebuild = bundle / 'rebuild'
    for directory in ('common', 'series'):
        shutil.copytree(
            ROOT / 'recipes' / directory,
            rebuild,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )
    for spec in specs:
        key = spec['id']
        if spec.get('mode') == 'rgb_parallax':
            build_pair(spec, args.source, args.catalog, args.survey.parent, bundle)
            continue
        views = read_views(spec, args.source, args.catalog, args.survey.parent)
        poses, alignment = registration(spec, views)
        fitting = [views[k] for k in spec['fit']]
        field, color, stats = fuse(fitting, poses)
        primary = fitting[0]
        Image.fromarray(primary.rgb).save(refs / (key + '.png'))
        Image.fromarray(np.uint8(primary.mask) * 255).save(cal / (key + '_outline.png'))
        texture, texture_stats = photo_texture(fitting, poses, field, color, stats)
        Image.fromarray(texture).save(refs / (key + '_ortho.png'))
        grain = road_texture(color, field['color_observed'], field['defect_mask'])
        Image.fromarray(grain).save(refs / (key + '_pavement.jpg'), quality=95)
        np.savez_compressed(cal / (key + '.npz'), **field)
        record = dict(
            spec=spec,
            anchor=spec['anchor'],
            fit=spec['fit'],
            holdout=spec['holdout'],
            poses_view_to_anchor={k: p.tolist() for k, p in poses.items()},
            road_feature_alignment=alignment,
            inputs={
                k: dict(files=v.sources, K=v.K.tolist(), size_px=list(v.rgb.shape[1::-1]))
                for k, v in views.items()
            },
            geometry=stats,
            photographic_texture=texture_stats,
            pose_constraint='Road features at least 8 pixels outside all labelled pit masks',
            withheld='Held-out interior depth, color, and labels do not enter the surface, texture or road plane. Exterior features locate its camera.',
        )
        put(cal / (key + '_capture.json'), record)
        source = str(Path('references') / (key + '.png'))
        if spec['source'] == 'PothRGBD':
            calibration = 'Nominal 462 px focal lengths, image-centred principal point, 1 mm per depth unit'
        elif spec['source'] == 'FanStereo':
            calibration = (
                'Pinhole projection fitted to native XYZ grid, source units converted from mm to m'
            )
        else:
            calibration = 'Supplied per-frame pinhole intrinsics and optical depth units'
        meta = dict(
            id=key,
            label=spec['label'],
            category='pothole',
            family=key,
            state='measured',
            recipe_version='multiview-depth-v1',
            source_image=source,
            source_sha256=sha256((bundle / source).read_bytes()).hexdigest(),
            geometry_source='Road-plane-aligned metric height field from registered fitting captures',
            scale_kind=calibration,
            measurement_record='calibration/' + key + '_capture.json',
            seed=307101 + int(key[1:]),
            **stats,
        )
        put(cal / (key + '.json'), meta)
        print(
            key,
            spec['label'],
            stats['pit_components'],
            'components',
            round(stats['depth_p95_m'] * 1000, 1),
            'mm P95',
            flush=True,
        )
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in refs.iterdir()}
    put(
        bundle / 'run.json',
        dict(
            edition='series',
            calibration=calibration_mode,
            ids=keys,
            mesh_only=args.mesh_only,
            preview=args.preview,
            published={},
            reference_sha256=hashes,
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
    put(stage / 'run.json', dict(editions=['series'], assets=keys))
    if calibration_mode == 'rgb_parallax':
        return
    evaluation = stage / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    run_step(
        cfg['compute_python'],
        ROOT.parent / 'scripts/check-pit-views.py',
        [
            '--run',
            stage,
            '--out',
            evaluation / 'measurements.json',
            '--source',
            args.source,
            '--catalog',
            args.catalog,
            '--files-base',
            args.survey.parent,
        ],
        env,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--survey', required=True, type=Path)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'work/reconstruction-inputs')
    parser.add_argument('--catalog', type=Path, default=ROOT.parent / 'catalog/capture-inputs.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/local.json')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--mesh-only', action='store_true')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error('Choose a new run directory')
    cv2.setNumThreads(2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix='.multiview-') as tmp:
        stage = Path(tmp) / 'run'
        build(args, stage)
        stage.rename(args.out)


if __name__ == '__main__':
    main()
