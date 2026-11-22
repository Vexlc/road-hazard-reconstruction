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
    if all(s.get('mode') == 'single_rgbd' for s in specs):
        calibration_mode = 'single_rgbd'
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
        single = spec.get('mode') == 'single_rgbd'
        if single:
            if 'depth_completion' in spec:
                raise ValueError('Photographic depth completion needs multiple fitting images')
            from single_depth_surfaces import split_depth, surface_regions, target_view

            annotated = target_view(spec.get('surface_regions', {}), views[spec['anchor']])
            training, reserved = split_depth(annotated)
            poses, alignment = {training.key: np.eye(4)}, {}
            fitting = [training]
        else:
            poses, alignment = registration(spec, views)
            fitting = [views[k] for k in spec['fit']]
        completion = spec.get('depth_completion')
        fine = {}
        if completion is not None:
            pitch, reach = float(completion['grid_pitch_m']), float(completion['support_radius_cells'])
            if not np.isfinite([pitch, reach]).all() or pitch <= 0 or reach <= 0:
                raise ValueError('Expected positive grid spacing and support radius')
            fine = dict(spacing=pitch, reach_cells=reach)
        field, color, stats = fuse(fitting, poses, completion=completion, **fine)
        if single and 'surface_regions' in spec:
            field['region_ids'], stats['region_names'] = surface_regions(spec['surface_regions'], fitting[0], field, stats)
        if completion is not None:
            control, _, control_stats = fuse(fitting, poses, **fine)
            if not np.array_equal(control['width'], field['width']) or not np.array_equal(control['view_count'], field['view_count']):
                raise ValueError('The range-only control must use the same grid and native measurements')
            np.savez_compressed(cal / (key + '_range_only.npz'), height=control['height'], width=control['width'],
                                defect_mask=control['defect_mask'])
            put(cal / (key + '_range_only.json'), control_stats)
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
        if single:
            record.update(pose_constraint='Fitting camera defines the sole source frame', withheld='Original ranges in 4x4 sensor blocks are zeroed before all geometry and texture fitting. RGB and the source outline remain available.',
                          reserved_depth_pixels=int(reserved.sum()), reserved_pit_depth_pixels=int((reserved & primary.mask).sum()))
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
            recipe_version='single-depth-holdout-v1' if single else 'multiview-depth-v1',
            source_image=source,
            source_sha256=sha256((bundle / source).read_bytes()).hexdigest(),
            geometry_source='Road-plane-aligned metric height field from one RGB-D capture' if single else 'Road-plane-aligned metric height field from registered fitting captures',
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
