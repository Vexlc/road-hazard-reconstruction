from pathlib import Path
import os,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/road_hazards';REPO=ROOT/'work/street_crafter'
catalog=json.loads((OUT/'catalog.json').read_text());progress=ROOT/'work/hazard_insertion_status.json';results=[]
for cfg in catalog:
    name=cfg['id'];sid='055' if cfg['kind']=='asphalt' else '056';lab='B' if sid=='055' else 'A'
    dest=ROOT/'outputs/road_hazard_insertions'/f'{lab}_{sid}_{name}'
    if (dest/'summary.json').exists():results.append({'id':name,'status':'complete','scene':sid});continue
    for _ in range(1200):
        if (OUT/name/'gaussians/metadata.json').exists():break
        time.sleep(1)
    else:raise RuntimeError('Gaussian asset not ready: '+name)
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'work/EasyVolcap'),PWD=str(REPO),CUDA_HOME='/usr/local/cuda',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_CUDA_ARCH_LIST='8.9',PYTHONUNBUFFERED='1',ROAD_HAZARD_ID=name)
    log=ROOT/'work'/f'insert_hazard_{name}.log'
    with log.open('w') as f:
        code=subprocess.run([str(ROOT/'work/envs/streetcrafter/bin/python'),'insert_gaussian_assets.py','--config',f'configs/pandaset_{sid}_warmup.yaml','mode','trajectory'],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    result={'id':name,'scene':sid,'status':'complete' if code==0 else 'failed','returncode':code};results.append(result);progress.write_text(json.dumps(results,indent=2));print(result,flush=True)
    if code:raise RuntimeError(name)
