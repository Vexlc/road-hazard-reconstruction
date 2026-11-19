"""Build all thirty enlarged stereo scenarios, then their combined comparison."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
from prepare_train import WORK,REPO,PY,ENV

ENV=dict(ENV,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',ROAD_BATCH_WORK=str(WORK))
ENV.pop('SCALED_ONLY_BATCHES',None)

def run(site):
    print('START',site,flush=True)
    with (WORK/'enlarged10'/f'render_{site}.log').open('w',encoding='utf-8') as log:
        p=subprocess.run([str(PY),str(WORK/'render_enlarged_stereo.py'),'--config',str(WORK/f'pandaset_{site}.yaml'),
                          'mode','trajectory'],cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,site
    print('DONE',site,flush=True)

(WORK/'enlarged10').mkdir(parents=True,exist_ok=True)
for site in ['034','109','110']:run(site)
for script in ['compose_enlarged_stereo.py','verify_enlarged_stereo.py']:
    with (WORK/'enlarged10'/f'{script}.log').open('w',encoding='utf-8') as log:
        p=subprocess.run([str(PY),str(WORK/script)],cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,script
print('ENLARGED STEREO COMPLETE',flush=True)
