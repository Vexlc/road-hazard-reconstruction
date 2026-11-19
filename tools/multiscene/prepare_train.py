"""Prepare and reconstruct three contrasting real road sequences."""
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

PROJECT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parents[2]
WORK=BASE/'tools/multiscene'
OUT=BASE/'outputs/道路病害_多场景迁入'
REPO=PROJECT/'work/street_crafter'
PY=Path(os.environ.get('STREETCRAFTER_PYTHON',PROJECT/'work/envs/streetcrafter/bin/python'))
SDK=PROJECT/'work/pandaset_sdk'
sdk_module=next((p for p in SDK.rglob('sequence.py') if p.parent.name=='pandaset'),None)
SDK_IMPORT=sdk_module.parent.parent if sdk_module else SDK
ENV=dict(os.environ,PYTHONPATH=f'{SDK_IMPORT}:{PROJECT}/work/EasyVolcap:{PROJECT}/work/asset_pipeline:{REPO}',
         CUDA_HOME=os.environ.get('CUDA_HOME','/usr/local/cuda'),TORCH_CUDA_ARCH_LIST=os.environ.get('TORCH_CUDA_ARCH_LIST','8.9'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
         OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HOME=str(PROJECT/'work/hf_cache'))

def run(scene,stage,args,cwd):
    log=WORK/f'{stage}_{scene}.log'
    start=time.time()
    print(f'START {scene} {stage}',flush=True)
    (WORK/'train_progress.json').write_text(json.dumps({'scene':scene,'stage':stage,'started_unix':start},indent=2))
    with log.open('w') as stream:
        p=subprocess.run([str(PY),*map(str,args)],cwd=cwd,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
    info={'scene':scene,'stage':stage,'elapsed_seconds':round(time.time()-start,2),'returncode':p.returncode,'log':str(log)}
    (WORK/f'{stage}_{scene}_status.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)
    if p.returncode: raise RuntimeError(f'{scene} {stage} failed; see {log}')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    template=yaml.safe_load((REPO/'configs/pandaset_056_warmup.yaml').read_text())
    for scene in ['034','109','110']:
        raw=WORK/'data/raw'/scene
        if not (raw/'camera/front_camera/poses.json').is_file():
            run(scene,'download',[WORK/'download_scene.py','--scene',scene],BASE)
        config=dict(template)
        model=OUT/'重建场景'/scene
        config.update(parent_cfg=str(REPO/'configs/waymo_val_121.yaml'),
                      source_path=str(WORK/'data/processed'/scene),model_path=str(model),
                      record_dir=str(WORK/'training_records'/scene),exp_name=f'road_diversity_{scene}',resume=False)
        path=WORK/f'pandaset_{scene}.yaml'
        path.write_text(yaml.safe_dump(config,sort_keys=False))
        def completed(stage):
            p=WORK/f'{stage}_{scene}_status.json'
            return p.exists() and json.loads(p.read_text())['returncode']==0
        if not completed('prepare'):
            run(scene,'prepare',[REPO/'data_processor/pandaset_processor/pandaset_convertor.py',
                 '--data_root',WORK/'data/raw','--target_dir',WORK/'data/processed','--scene_ids',int(scene),
                 '--workers',1,'--process_keys','timestamps','images','calib','pose','dynamic_masks','objects'],REPO/'data_processor')
        if not completed('lidar'):
            run(scene,'lidar',[REPO/'data_processor/pandaset_processor/pandaset_convertor.py',
                 '--data_root',WORK/'data/raw','--target_dir',WORK/'data/processed','--scene_ids',int(scene),
                 '--workers',1,'--process_keys','lidar_forward'],REPO/'data_processor')
        if not completed('sky'):
            run(scene,'sky',[WORK/'sky_masks.py',scene],BASE)
        final=model/'trained_model/iteration_7000.pth'
        if not final.exists(): run(scene,'train',[REPO/'train.py','--config',path,'mode','train'],REPO)
        (WORK/f'{scene}_ready.json').write_text(json.dumps({'scene':scene,'config':str(path),'checkpoint':str(final)},indent=2))
    print('ALL THREE RECONSTRUCTIONS READY',flush=True)

if __name__=='__main__': main()
