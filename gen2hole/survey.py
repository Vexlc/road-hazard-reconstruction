"""Reconstruct a family of movable road covers from photographed components or marked openings."""
import argparse
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from gen2hole import ROOT, load, run_step, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--survey', type=Path, required=True, help='JSON with source images, landmarks, dimensions and poses')
    p.add_argument('--out', type=Path, required=True, help='new run directory')
    p.add_argument('--config', type=Path, default=ROOT/'config/local.json')
    p.add_argument('--mesh-only',action='store_true')
    p.add_argument('--preview',action='store_true')
    args = p.parse_args()
    survey, envs = load(args.survey), load(args.config)
    ids = [s['id'] for s in survey['states']]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(('M' if s.get('absent') else 'A')+r'[0-9]{2,}',s['id']) for s in survey['states']):
        p.error('Every state needs a unique A or M ID, e.g. A91 or M91')
    perforated = 'perforations' in survey
    roles = ('frame', 'lid') if not perforated and any(not s.get('absent') for s in survey['states']) else ('frame',)
    paths = {name: args.survey.parent/survey[name]['image'] for name in roles}
    if any(not path.is_file() for path in paths.values()):
        p.error('The requested component images must exist relative to the survey')
    run = args.out.absolute()
    if run.exists() or run.is_symlink():
        p.error(f"Choose a new output directory: {run}")
    run.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.survey-',dir=run.parent) as temp:
        stage = Path(temp)/'run'
        build_bundle(args,survey,envs,paths,ids,stage)
        stage.rename(run)
    print(run/'series')


def build_bundle(args,survey,envs,paths,ids,run):
    bundle = run/'series'
    refs = bundle/'references'
    refs.mkdir(parents=True)
    (bundle/'calibration').mkdir()
    for name,path in paths.items():
        if 'perforations' in survey and name == 'frame' and path.suffix.lower() != '.png':
            original = 'source_frame' + path.suffix.lower()
            shutil.copy2(path, refs / original)
            survey = dict(survey, frame=dict(survey['frame'], original_reference=original))
        else:
            shutil.copy2(path,refs/(name+'.png'))
    if 'perforations' in survey:
        profile = dict(survey['perforations'])
        for name in ('mask', 'lining', 'displacement', 'pavement'):
            if name in profile:
                original = args.survey.parent / profile[name]
                filename = ('source_pavement' if name == 'pavement' else name) + original.suffix.lower()
                shutil.copyfile(original, refs / filename)
                profile[name] = filename
        survey = dict(survey, perforations=profile)
    shutil.copy2(ROOT/'inputs/shared/back.png', refs/'back.png')
    for group,fields in [('rgbd',('depth','bottom_mask')),('pose_capture',('image','depth','lid_mask'))]:
        if group not in survey:continue
        sensor = dict(survey[group])
        if sensor.get('source_kind') not in ('sensor_capture','synthetic_benchmark'):
            raise ValueError(f"{group}.source_kind must identify sensor_capture or synthetic_benchmark")
        for key in fields:
            if key in sensor:
                original = args.survey.parent/sensor[key]
                name = group+'_'+key+original.suffix.lower()
                shutil.copy2(original,refs/name)
                sensor[key] = name
        survey = dict(survey,**{group:sensor})
    if 'capture_record' in survey:
        shutil.copyfile(args.survey.parent/survey['capture_record'],bundle/'calibration/capture-fit.json')
        shutil.copyfile(args.survey.parent/survey['capture_crop'],refs/'capture.png')
        for source in ('annotations.json','Dataset-Terms.txt'):
            original=args.survey.parent/source
            if original.is_file():
                shutil.copyfile(original,bundle/'calibration'/source)
        survey=dict(survey,capture_record='capture-fit.json',capture_crop='capture.png')
    save(bundle/'calibration/survey.json',survey)
    code = bundle/'rebuild'
    for directory in ('common','series'):
        shutil.copytree(ROOT/'recipes'/directory,code,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sha_refs = {f.name:sha(f) for f in refs.iterdir()}
    save(bundle/'run.json',dict(edition='series',calibration='photo_components',ids=ids,mesh_only=args.mesh_only,
                               preview=args.preview,published={},reference_sha256=sha_refs))
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    mesh, compute = envs['mesh_python'], envs['compute_python']
    if 'perforations' in survey:
        run_step(compute,code/'trace_grate.py',[],env)
        run_step(mesh,code/'perforated_cover.py',[],env)
    else:
        run_step(mesh,code/'survey_chamber.py',[],env)
    if not args.mesh_only:
        run_step(compute,code/'export_splats.py',['--cache',bundle/'mesh_cache',*ids],env)
    if args.preview:
        run_step(mesh,code/'render_mesh.py',['--cache',bundle/'mesh_cache','--out',bundle/'views',*ids],env)
    run_step(mesh,code/'verify.py',[],env)
    save(run/'run.json',dict(editions=['series'],assets=ids))


if __name__ == '__main__':
    main()
