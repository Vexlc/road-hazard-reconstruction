from pathlib import Path
import os,json,subprocess,concurrent.futures,threading
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/street_crafter'
catalog=json.loads((ROOT/'outputs/road_hazards/catalog.json').read_text());results=[];lock=threading.Lock()
def run(cfg):
    name=cfg['id'];sid='055' if cfg['kind']=='asphalt' else '056';lab='B' if sid=='055' else 'A';dest=ROOT/'outputs/road_hazard_insertions'/f'{lab}_{sid}_{name}'
    summary=json.loads((dest/'summary.json').read_text())
    if summary.get('road_appearance_transfer') and summary.get('render_resolution')==[1920,1080]:return {'id':name,'status':'complete'}
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'work/EasyVolcap'),PWD=str(REPO),CUDA_HOME='/usr/local/cuda',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_CUDA_ARCH_LIST='8.9',PYTHONUNBUFFERED='1')
    with (ROOT/'work'/f'polish_{name}.log').open('w') as f:
        code=subprocess.run([str(ROOT/'work/envs/streetcrafter/bin/python'),'polish_road_hazard.py','--config',str(dest/'render_config.yaml')],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    result={'id':name,'status':'complete' if code==0 else 'failed','returncode':code}
    with lock:
        results.append(result);(ROOT/'work/hazard_polish_status.json').write_text(json.dumps(results,indent=2));print(result,flush=True)
    if code:raise RuntimeError(name)
    return result
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(run,catalog))
