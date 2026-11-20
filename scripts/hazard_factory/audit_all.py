"""Run read-only native visibility probes in each configured scene."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from run import PROJECT,HERE,atomic_json


def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'));work=Path(job['work'])
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',
             ROAD_BATCH_WORK=str(work),HAZARD_FACTORY_JOB=str(args.job.resolve()),
             PYTHONPATH=':'.join(str(PROJECT/p) for p in ('work/EasyVolcap','work/asset_pipeline',
                 'work/road_hazard_pipeline','outputs/road_hazard_expansion/reproduction','work/street_crafter')),
             OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    def run(site):
        log=work/f'visibility_{site}.log'
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([str(PROJECT/'work/envs/streetcrafter/bin/python'),
                    str(HERE/'audit_visibility.py'),'--config',job['configs'][site],'mode','trajectory'],
                    cwd=PROJECT/'work/street_crafter',env=env,stdout=stream,stderr=subprocess.STDOUT)
        print('VISIBILITY END',site,result.returncode,flush=True)
        return {'site':site,'returncode':result.returncode,'log':str(log)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run,job['configs']))
    atomic_json(work/'visibility_runs.json',results)
    if any(r['returncode'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);main(p.parse_args())
