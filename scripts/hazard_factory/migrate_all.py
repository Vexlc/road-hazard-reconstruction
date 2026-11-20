"""Run explicit calibrated scene jobs, retrying failed jobs once sequentially."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

from run import PROJECT, HERE, atomic_json


def main(args):
    path=args.job.resolve()
    job=json.loads(path.read_text(encoding='utf-8'))
    new=[m for library in map(Path,job['libraries'])
         for m in json.loads((library/'asset_manifest.json').read_text(encoding='utf-8'))
         if m.get('recipe_version')=='physical-v1']
    counts=[sum(m['category']==c for m in new) for c in ('pothole','missing','raised')]
    if not min(counts) or len(set(counts))!=1:
        raise ValueError('Five-category batch jobs require balanced new triplets; append the missing stream indices first.')
    work=Path(job['work']);work.mkdir(parents=True,exist_ok=True)
    plan_backup=Path(job['output']).parent/'场景分组计划.json'
    if not (work/'case_plan.json').exists() and plan_backup.is_file():
        shutil.copy2(plan_backup,work/'case_plan.json')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',
             ROAD_BATCH_WORK=str(work),HAZARD_FACTORY_JOB=str(path),
             PYTHONPATH=':'.join(str(PROJECT/p) for p in (
                 'work/EasyVolcap','work/asset_pipeline','work/road_hazard_pipeline',
                 'outputs/road_hazard_expansion/reproduction','work/street_crafter')),
             CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.9',
             OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    env['HF_HOME']=str(PROJECT/'work/hf_cache')
    if job.get('annotated_road_cache'):
        # Config parsing uses the supplied reconstruction environment, so the
        # standard-library launcher does not acquire a separate YAML dependency.
        source=subprocess.check_output([str(PROJECT/'work/envs/streetcrafter/bin/python'),'-c',
                    "import json,sys,yaml;print(json.dumps(yaml.safe_load(open(sys.argv[1]))['source_path']))",
                    next(iter(job['configs'].values()))],env=env,text=True)
        raw=Path(job.get('raw_data_root',str(Path(json.loads(source)).parents[1]/'raw')))
        if raw.is_dir():
            with (work/'annotated_road.log').open('w',encoding='utf-8') as stream:
                checked=subprocess.run([str(PROJECT/'work/envs/streetcrafter/bin/python'),str(HERE/'annotated_road.py'),
                    '--raw',str(raw),'--output',job['annotated_road_cache'],'--scenes',*job['configs']],
                    env=env,stdout=stream,stderr=subprocess.STDOUT)
            if checked.returncode:raise RuntimeError('annotated road preparation failed; see annotated_road.log')
        else:
            from run import sha
            cache=Path(job['annotated_road_cache'])
            provenance=json.loads((cache/'provenance.json').read_text(encoding='utf-8'))
            for site in job['configs']:
                for record in provenance['scenes'][site]['frames']:
                    if sha(cache/site/f'{record["frame"]:03d}.npz')!=record['output_sha256']:
                        raise ValueError('annotated-road cache damaged; raw data required for repair')
    with (work/'road_masks.log').open('w',encoding='utf-8') as stream:
        masks=subprocess.run([str(PROJECT/'work/envs/streetcrafter/bin/python'),str(HERE/'road_masks.py')],
                              cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if masks.returncode:raise RuntimeError('road support preparation failed; see road_masks.log')
    def run(site,attempt):
        log=work/f'migration_{site}_attempt{attempt}.log'
        print('START',site,'attempt',attempt,flush=True)
        started=time.time()
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.run([str(PROJECT/'work/envs/streetcrafter/bin/python'),
                     str(HERE/'migrate.py'),'--config',job['configs'][site],'mode','trajectory'],
                     cwd=PROJECT/'work/street_crafter',env=env,stdout=stream,stderr=subprocess.STDOUT)
        status={'scene':site,'attempt':attempt,'returncode':process.returncode,
                'seconds':time.time()-started,'log':str(log)}
        atomic_json(work/f'migration_{site}_status.json',status)
        print('END',site,process.returncode,flush=True)
        return status
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run,site,1):site for site in job['configs']}
        for future in as_completed(pending):results.append(future.result())
    for result in list(results):
        if result['returncode']:
            results.append(run(result['scene'],2))
    final={r['scene']:r for r in results}
    atomic_json(work/'migration_runs.json',{'attempts':results,'final':final})
    if any(r['returncode'] for r in final.values()):raise SystemExit(1)
    atomic_json(plan_backup,json.loads((work/'case_plan.json').read_text(encoding='utf-8')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.workers<=2:p.error('workers must be 1 or 2')
    main(args)
