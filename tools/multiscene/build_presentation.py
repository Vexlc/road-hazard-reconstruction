"""Render source comparisons and assemble the requested five-panel videos."""
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess

from prepare_train import WORK,REPO,PY,ENV

ENV=dict(ENV,ROAD_BATCH_WORK=str(WORK),PYTHONUTF8='1',PYTHONIOENCODING='utf-8')

def run(job):
    name,args=job
    print('START',name,flush=True)
    with (WORK/'composite3'/f'{name}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([str(PY),*map(str,args)],cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    print('END',name,result.returncode,flush=True)
    if result.returncode:raise RuntimeError(name+' failed')

jobs=[('views_'+site,[WORK/'render_presentation_views.py','--config',WORK/f'pandaset_{site}.yaml','mode','trajectory']) for site in ['034','109','110']]
jobs.append(('orbits',[WORK/'render_presentation_orbits.py']))
(WORK/'composite3').mkdir(parents=True,exist_ok=True)
for job in jobs:run(job)
run(('compose',[WORK/'compose_presentation.py']))
run(('verify',[WORK/'verify_presentation.py']))
print('PRESENTATION COMPLETE',flush=True)
