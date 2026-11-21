"""Replay an image-based scene with project-local paths and compare its archived RGB."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT=Path(__file__).resolve().parents[1]


def child():
    import hashlib
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    import torch
    from create_scene import create_scene
    from street_gaussian.config import cfg
    from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
    case=Path(os.environ['IMAGEFIT_REPLAY_CASE']);config=json.loads((case/'scene.json').read_text(encoding='utf8'))
    if Path(cfg.model_path).resolve()!=case.resolve():raise ValueError('YAML model_path does not point to this case')
    frame=int(os.environ['IMAGEFIT_REPLAY_FRAME'])
    start=json.loads((case.parent/'placement_plan.json').read_text(encoding='utf8'))['drive_start_by_site'][config['site']]
    torch.set_num_threads(4)
    with torch.no_grad():
        scene=create_scene();g=scene.gaussians
        cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
        cam=cameras[frame];w,h=960,540;k=cam.K.clone();k[0]*=w/cam.image_width;k[1]*=h/cam.image_height
        cam.image_width,cam.image_height=w,h;cam.set_intrinsic(k)
        result=StreetGaussianRenderer().render(cam,g)['rgb']
        rgb=np.uint8(result.clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5)
        bg_count=len(g.background._xyz)
    expected=config['retained_background_points']+config['host_road_guard']['points']+sum(p['asset_points']+p['apron_points'] for p in config['placements'])
    if expected != bg_count:
        raise ValueError(f"Expected {expected} background Gaussians, found {bg_count}")
    record={'case':case.name,'frame':frame,'background_gaussians':bg_count,'raw_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
    if start<=frame<start+40:
        r=imageio.get_reader(str(case/'global_before_after.mp4'))
        video=r.get_data(frame-start)[:,960:,:];r.close()
        diff=video.astype('f4')-rgb
        record.update(video_reference_rgb_mae=float(abs(diff).mean()),video_reference_psnr_db=float(20*np.log10(255/np.sqrt((diff**2).mean()))))
        if record['video_reference_rgb_mae'] >= 4.2:
            raise ValueError(f"Frame differs from archived video: {record}")
    output=os.environ.get('IMAGEFIT_REPLAY_PNG')
    if output:Image.fromarray(rgb).save(output)
    print('IMAGEFIT_REPLAY',json.dumps(record),flush=True)


def launcher():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=ROOT)
    parser.add_argument('--bundle',type=Path,help='One released or experimental asset package')
    parser.add_argument('--case',nargs='+',required=True,help='Case IDs or all')
    parser.add_argument('--frame',type=int,default=None)
    parser.add_argument('--png',type=Path,help='Output directory for rendered frames')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();p=args.project.resolve()
    if args.bundle:
        bundles = [args.bundle.resolve()]
    else:
        registry = json.loads((ROOT/'catalog/roadgs-releases.json').read_text(encoding='utf-8'))
        current = registry['releases'][registry['current']]
        bundles = [ROOT/current['directory']/current['packages'][key]['directory']
                   for key in ('image9', 'image100')]
    cases = {path.parent.name: path.parent for bundle in bundles
             for path in (bundle/'scenes').glob('*/scene.json')}
    ids = sorted(cases) if args.case == ['all'] else args.case
    if not ids or any(case_id not in cases for case_id in ids):
        parser.error(f"Missing scene(s) in the selected packages: {args.case}")
    records=[]
    for case_id in ids:
        case=cases[case_id]
        data=json.loads((case/'scene.json').read_text(encoding='utf8'))
        frame=args.frame if args.frame is not None else data['placements'][1]['source_reference_frame']
        if not 0<=frame<80:raise ValueError('Expected source frame 0..79')
        env=dict(os.environ,PYTHONUTF8='1',CUDA_HOME=os.environ.get('CUDA_HOME','/usr/local/cuda'),
                 TORCH_CUDA_ARCH_LIST=os.environ.get('TORCH_CUDA_ARCH_LIST','8.9'),
                 IMAGEFIT_REPLAY_CASE=str(case),IMAGEFIT_REPLAY_FRAME=str(frame),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',
                 PYTHONPATH=':'.join(str(p/q) for q in ['work/street_crafter','work/EasyVolcap','work/asset_pipeline','work/road_hazard_pipeline']))
        if args.png:
            args.png.mkdir(exist_ok=True,parents=True)
            env['IMAGEFIT_REPLAY_PNG']=str((args.png/(case_id+'_'+str(frame)+'.png')).resolve())
        site = data['site']
        parent = p/'work/hazard_factory_v3'/f'pandaset_{site}.yaml'
        if not parent.is_file():
            parser.error(f"Missing {parent}. Run scripts/hazard_factory/configure.py in the project.")
        # Absolute paths in the delivered YAML identify the original machine. Feed
        # the renderer a temporary, equivalent YAML without editing the release.
        import yaml
        config = yaml.safe_load((case/'render_config.yaml').read_text(encoding='utf-8'))
        config.update(model_path=str(case), parent_cfg=str(parent))
        with tempfile.TemporaryDirectory(prefix='roadgs-replay-') as temp:
            local = Path(temp)/'replay.yaml'
            local.write_text(yaml.safe_dump(config), encoding='utf-8')
            proc=subprocess.run([str(p/'work/envs/streetcrafter/bin/python'),'-u',str(Path(__file__).resolve()),
                                '--config',str(local),'mode','trajectory'],cwd=p/'work/street_crafter',
                                env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output=proc.stdout.decode('utf8',errors='replace')
        if proc.returncode:raise RuntimeError(output[-4200:])
        record=json.loads([s for s in output.splitlines() if s.startswith('IMAGEFIT_REPLAY ')][-1].split(' ',1)[1])
        records.append(record);print(record,flush=True)
    if args.report:args.report.write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    if os.environ.get('IMAGEFIT_REPLAY_CASE'):child()
    else:launcher()
