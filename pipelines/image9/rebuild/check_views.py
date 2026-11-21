"""Run the sensor transfer and the original v3 shaft-visibility probe on published checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--sites',nargs='+',default=['034','109','110'])
    ap.add_argument('--mode',choices=['sensor','visibility','both'],default='both')
    args=ap.parse_args();project=args.project.resolve()
    job=json.loads(args.job.read_text());work=Path(job['work'])
    root=Path(job['output']).parent
    job.update(visibility_output=str(root/'checks'/'shaft_visibility'),visibility_width=640,visibility_stride=1)
    path=work/'review_job.json';path.write_text(json.dumps(job,ensure_ascii=False,indent=2))
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',ROAD_BATCH_WORK=str(work),HAZARD_FACTORY_JOB=str(path),ROAD_PROJECT=str(project),
             CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
             PYTHONPATH=':'.join(str(project/p) for p in ('work/EasyVolcap','work/asset_pipeline','work/road_hazard_pipeline',
                                'outputs/road_hazard_expansion/reproduction','work/street_crafter')))
    scripts={'sensor':Path(__file__).with_name('sensor_views.py').resolve(),
             'visibility':project/'scripts/hazard_factory/audit_visibility.py'}
    for mode,script in scripts.items():
        if args.mode not in ['both',mode]:continue
        for site in args.sites:
            print(mode,site,flush=True)
            with (work/(mode+'_'+site+'.log')).open('w') as stream:
                subprocess.run([str(project/'work/envs/streetcrafter/bin/python'),'-u',str(script),'--config',job['configs'][site],'mode','trajectory'],
                               cwd=project/'work/street_crafter',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    main()
