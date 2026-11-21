"""Open a released checkpoint via its own YAML in a fresh StreetCrafter process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


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
    bundle=case.parents[1]
    plan=bundle/'migration'/case.name/'plan.json'
    if not plan.is_file():
        plan=bundle/'scenes/placement_plan.json'
    start=json.loads(plan.read_text(encoding='utf8'))['drive_start_by_site'][config['site']]
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
    assert expected==bg_count
    record={'case':case.name,'frame':frame,'background_gaussians':bg_count,'raw_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
    if start<=frame<start+40:
        r=imageio.get_reader(str(case/'global_before_after.mp4'))
        video=r.get_data(frame-start)[:,960:,:];r.close()
        diff=video.astype('f4')-rgb
        record.update(video_reference_rgb_mae=float(abs(diff).mean()),video_reference_psnr_db=float(20*np.log10(255/np.sqrt((diff**2).mean()))))
        assert record['video_reference_rgb_mae']<4.2
    output=os.environ.get('IMAGEFIT_REPLAY_PNG')
    if output:Image.fromarray(rgb).save(output)
    print('IMAGEFIT_REPLAY',json.dumps(record),flush=True)


def launcher():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True,help='Rebuilt edition or original release directory')
    parser.add_argument('--project',type=Path,help='StreetCrafter project override')
    parser.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1]/'config/local.json')
    parser.add_argument('--case',nargs='+',required=True,help='Case IDs or all')
    parser.add_argument('--frame',type=int,default=None)
    parser.add_argument('--png',type=Path,help='Output directory for rendered frames')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();runtime=json.loads(args.runtime.read_text(encoding='utf8'))
    p=(args.project or Path(runtime['road_project'])).resolve();root=args.bundle.resolve()
    python=p/'work/envs/streetcrafter/bin/python' if args.project else runtime['compute_python']
    parents = {}
    if not args.project and runtime.get('road_config'):
        road_config = Path(runtime['road_config'])
        if road_config.is_file():
            parents = json.loads(road_config.read_text(encoding='utf-8'))['configs']
    ids=[d.parent.name for d in sorted((root/'scenes').glob('*/scene.json'))] if args.case==['all'] else args.case
    records=[]
    for case_id in ids:
        case=(root/'scenes'/case_id).resolve()
        data=json.loads((case/'scene.json').read_text(encoding='utf8'))
        frame=args.frame if args.frame is not None else data['placements'][min(1,len(data['placements'])-1)]['source_reference_frame']
        if not 0<=frame<80:raise ValueError('Expected source frame 0..79')
        env=dict(os.environ,PYTHONUTF8='1',CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.9',
                 IMAGEFIT_REPLAY_CASE=str(case),IMAGEFIT_REPLAY_FRAME=str(frame),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',
                 PYTHONPATH=':'.join(str(p/q) for q in ['work/street_crafter','work/EasyVolcap','work/asset_pipeline','work/road_hazard_pipeline']))
        site = data['site']
        parent = parents.get(site, p/'work/hazard_factory_v3'/f'pandaset_{site}.yaml')
        env.pop('IMAGEFIT_REPLAY_PARENT', None)
        if Path(parent).is_file():
            env['IMAGEFIT_REPLAY_PARENT'] = str(Path(parent).resolve())
        if args.png:
            args.png.mkdir(exist_ok=True,parents=True)
            env['IMAGEFIT_REPLAY_PNG']=str((args.png/(case_id+'_'+str(frame)+'.png')).resolve())
        proc=subprocess.run([str(python),'-u',str(Path(__file__).resolve()),
                            '--config',str(case/'render_config.yaml'),'mode','trajectory'],cwd=p/'work/street_crafter',
                            env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output=proc.stdout.decode('utf8',errors='replace')
        if proc.returncode:raise RuntimeError(output[-4200:])
        record=json.loads([s for s in output.splitlines() if s.startswith('IMAGEFIT_REPLAY ')][-1].split(' ',1)[1])
        records.append(record);print(record,flush=True)
    if args.report:args.report.write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    if os.environ.get('IMAGEFIT_REPLAY_CASE'):
        # Read YAML in the renderer's environment. Preserve all scene settings
        # while resolving the checkpoint and its parent in the installed project.
        import yaml
        index = sys.argv.index('--config') + 1
        settings = yaml.safe_load(Path(sys.argv[index]).read_text(encoding='utf-8'))
        settings['model_path'] = os.environ['IMAGEFIT_REPLAY_CASE']
        if os.environ.get('IMAGEFIT_REPLAY_PARENT'):
            settings['parent_cfg'] = os.environ['IMAGEFIT_REPLAY_PARENT']
        with tempfile.TemporaryDirectory(prefix='gen2hole-replay-') as folder:
            local = Path(folder)/'replay.yaml'
            local.write_text(yaml.safe_dump(settings), encoding='utf-8')
            sys.argv[index] = str(local)
            child()
    else:
        launcher()
