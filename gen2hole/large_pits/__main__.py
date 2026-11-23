"""Build and replay an oblique-photo dry pit from a self-contained set of reconstruction inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parents[2]
ENGINE = PROJECT/'gen2hole'
EXAMPLE = ENGINE/'examples/oblique-road-pits'
PAVEMENT = ('Asphalt031_4K-PNG_Color.png', 'Asphalt031_4K-PNG_Displacement.png')
REPO = 'Vexlc/road-hazard-reconstruction'
RELEASE = 'large-pits-v1.0.0'
EXAMPLES = ('P190', 'P198', 'P199', 'P200', 'P201')


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(python, script, *args):
    env = dict(os.environ, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='3')
    env.setdefault('CUDA_HOME','/usr/local/cuda')
    subprocess.run([str(python), str(script), *map(str,args)], env=env, check=True)


def files(root):
    return {p.relative_to(root).as_posix(): checksum(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'checksums.json'}


def prepare(a):
    if a.out.exists():
        raise ValueError('Choose a new input directory')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=a.out.parent) as tmp:
        stage = Path(tmp)/'inputs'
        (stage/'asphalt').mkdir(parents=True)
        profile = json.loads(a.profile.read_text())
        key = profile['id']
        if json.loads(a.spec.read_text())['id'] != key:
            raise ValueError('Outline and detail profile identify different assets')
        for src, dst in [(a.image,key+'.jpg'), (a.spec,'outline.json'), (a.profile,'profile.json')]:
            shutil.copyfile(src,stage/dst)
        for rec in profile['materials'].values():
            for field in ('albedo','displacement'):
                name = Path(rec[field]['url']).name
                if hashlib.md5((a.maps/name).read_bytes()).hexdigest() != rec[field]['md5']:
                    raise ValueError('Incorrect material '+name)
                shutil.copyfile(a.maps/name, stage/name)
        for name in PAVEMENT:
            shutil.copyfile(a.pavement/name,stage/'asphalt'/name)
        depth, insets = key+'-depth.npz', key+'-insets.npz'
        if a.weights:
            run(a.moge_python,ENGINE/'infer_oblique_depth.py','--image',stage/(key+'.jpg'),'--spec',stage/'outline.json',
                '--weights',a.weights,'--device',a.device,'--out',stage/depth)
            # The inset camera only needs the original RGB, outline and full-image intrinsics.
            layout = Path(tmp)/'inference'
            for folder, names in [('references',{key+'.jpg':key+'.jpg'}),
                                  ('calibration',{key+'-outline.json':'outline.json', depth:depth})]:
                (layout/folder).mkdir(parents=True)
                for target,source in names.items():
                    (layout/folder/target).symlink_to((stage/source).resolve())
            run(a.moge_python,ENGINE/'infer_oblique_inset.py','--bundle',layout,'--profile',stage/'profile.json',
                '--weights',a.weights,'--device',a.device,'--out',stage/insets)
        else:
            if not a.inset_cache:
                raise ValueError('--depth-cache requires --inset-cache')
            shutil.copyfile(a.depth_cache,stage/depth)
            shutil.copyfile(a.inset_cache,stage/insets)
        (stage/'checksums.json').write_text(json.dumps(files(stage),indent=2)+'\n')
        stage.rename(a.out)


def build(a, cfg):
    expected = json.loads((a.inputs/'checksums.json').read_text())
    actual = files(a.inputs)
    if actual != expected:
        raise ValueError('Input checksum mismatch: '+', '.join(k for k in sorted(set(actual)|set(expected))
                                                                if actual.get(k) != expected.get(k)))
    args = ['--specs',a.inputs/'outline.json','--inputs',a.inputs,'--detail-profile',a.inputs/'profile.json',
            '--detail-inputs',a.inputs,'--pavement',a.inputs/'asphalt','--config',a.config,'--out',a.out]
    if not a.baseline:
        args.append('--visible-surface')
    run(cfg['compute_python'],ENGINE/'oblique_pits.py',*args)


def example(a, cfg):
    if a.out.exists():
        raise ValueError('Choose a new output directory: '+str(a.out))
    for name in ('mesh_python', 'compute_python'):
        if not Path(cfg[name]).is_file():
            raise ValueError(f'Missing {name}. Install the environments in docs/03-environment-and-relocation.md')
    is_p190 = a.asset == 'P190'
    name = 'p190-inputs.zip' if is_p190 else 'lane-pits-inputs.zip'
    entries = Path(__file__).with_name('release-sha256.txt').read_text().splitlines()
    hashes = {filename: digest for digest, filename in (line.split('  ', 1) for line in entries)}
    cache = PROJECT/'work/large-pits-downloads'
    archive = a.archive or cache/name
    if not a.archive and not archive.exists():
        if not shutil.which('gh'):
            raise ValueError(f'Download {name} from the GitHub release and pass --archive FILE, or install GitHub CLI and sign in.')
        cache.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=cache) as tmp:
            subprocess.run(['gh','release','download',RELEASE,'--repo',REPO,'--pattern',name,'--dir',tmp],check=True)
            downloaded = Path(tmp)/name
            if checksum(downloaded) != hashes[name]:
                raise ValueError('Downloaded archive checksum mismatch: '+name)
            downloaded.replace(archive)
    if checksum(archive) != hashes[name]:
        raise ValueError('Input archive checksum mismatch: '+str(archive))
    prefix = 'inputs/' if is_p190 else a.asset+'/'
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=a.out.parent) as tmp, ZipFile(archive) as kit:
        inputs = Path(tmp)/prefix
        members = [n for n in kit.namelist() if n.startswith(prefix)]
        kit.extractall(tmp, members=members)
        build(argparse.Namespace(inputs=inputs,config=a.config,out=a.out,baseline=False), cfg)
    if is_p190:
        record = json.loads((PROJECT/'validation/large-dry-pit-P190.json').read_text())
        expected = record['models']['v3']['native_checks']['sha256']
    else:
        record = json.loads((PROJECT/'validation/lane-pits-batch.json').read_text())
        expected = record['assets'][a.asset]['native_hashes']
    actual = json.loads((a.out/'series/checks.json').read_text())[a.asset]['sha256']
    if actual != expected:
        changed = [n for n in sorted(set(expected)|set(actual)) if expected.get(n)!=actual.get(n)]
        raise ValueError('Built files differ from this release: '+', '.join(changed))
    print(f'{a.asset}: {len(expected)} native files match the release. {a.out}/series/assets/pothole/{a.asset}')


def drive(a, cfg):
    run(sys.executable,ENGINE/'downstream/place.py','--bundle',a.bundle,'--assets',a.asset,'--site',a.site,
        '--group',a.group,'--config',a.config)
    args = ['--bundle',a.bundle,'--case',f'{a.site}_{a.group:02}','--raw',cfg['pandaset_raw'],
            '--processed',cfg['pandaset_processed'],'--out',a.out]
    if a.scans:
        args.extend(['--scans',a.scans])
    if not a.baseline:
        args.append('--surface-lighting')
    run(cfg['compute_python'],PROJECT/'scripts/render-placed-pit-drive.py',*args)


def audit(a,cfg):
    args = ['--bundle',a.bundle,'--baseline',a.baseline,'--asset',a.asset,'--out',a.out,'--roads',*a.roads]
    if a.baseline_road:
        args.extend(['--baseline-road',a.baseline_road])
    run(cfg['compute_python'],Path(__file__).with_name('audit.py'),*args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare',help='Create portable inputs from a photo and scans')
    for name in ('image','maps','pavement','out'):
        prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--spec',type=Path,default=EXAMPLE/'P190.json')
    prep.add_argument('--profile',type=Path,default=EXAMPLE/'P190-detail.json')
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument('--weights',type=Path)
    source.add_argument('--depth-cache',type=Path)
    prep.add_argument('--inset-cache',type=Path)
    prep.add_argument('--moge-python',type=Path,default=Path(sys.executable))
    prep.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    build_cmd = sub.add_parser('build',help='Build GLB, Gaussians and metric surface from verified inputs')
    build_cmd.add_argument('--inputs',type=Path,required=True)
    build_cmd.add_argument('--baseline',action='store_true',help='Reproduce v2 with the same inputs')
    demo = sub.add_parser('example',help='Rebuild a released example and verify its native files')
    demo.add_argument('asset',choices=EXAMPLES)
    demo.add_argument('--archive',type=Path,help='Use a downloaded release input ZIP')
    demo.add_argument('--out',type=Path,help='New output directory (default: runs/reproduce-<asset>)')
    demo.add_argument('--config',type=Path,default=ENGINE/'config/local.json')
    drive_cmd = sub.add_parser('drive',help='Place a built asset and render the two real cameras')
    drive_cmd.add_argument('--bundle',type=Path,required=True)
    drive_cmd.add_argument('--site',choices=['034','109','110'],required=True)
    drive_cmd.add_argument('--group',type=int,default=190)
    drive_cmd.add_argument('--asset',default='P190')
    drive_cmd.add_argument('--scans',type=Path,help='Optional prepared PandaSet LiDAR cache')
    drive_cmd.add_argument('--baseline',action='store_true',help='Replay the original v2 lighting')
    check = sub.add_parser('audit',help='Compare delivered GLBs and verify the road videos')
    check.add_argument('--bundle',type=Path,required=True)
    check.add_argument('--baseline',type=Path,required=True)
    check.add_argument('--asset',default='P190')
    check.add_argument('--roads',nargs='*',type=Path,default=[])
    check.add_argument('--baseline-road',type=Path)
    for command in (drive_cmd,build_cmd,check):
        command.add_argument('--config',type=Path,default=ENGINE/'config/local.json')
        command.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    try:
        if a.command == 'prepare':
            prepare(a)
        else:
            if a.command == 'example':
                if a.out is None:
                    a.out = PROJECT/'runs'/('reproduce-'+a.asset.lower())
                if not a.config.is_file() and a.config == ENGINE/'config/local.json':
                    subprocess.run([sys.executable,str(PROJECT/'scripts/configure-gen2hole.py')],check=True)
            cfg = json.loads(a.config.read_text())
            {'drive':drive,'build':build,'audit':audit,'example':example}[a.command](a,cfg)
    except (OSError,ValueError,subprocess.CalledProcessError) as err:
        p.exit(1, f'{a.command}: {err}\n')


if __name__ == '__main__':
    main()
