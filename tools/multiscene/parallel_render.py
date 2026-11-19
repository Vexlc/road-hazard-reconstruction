"""Re-render corrected shaft cleanup independently in three road scenes."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
from finish_pipeline import WORK,REPO,PY,ENV

def run(site):
    print('CORRECTED RENDER START',site,flush=True)
    with (WORK/f'render_v2_{site}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([str(PY),str(WORK/'render_scenes.py'),'--config',str(WORK/f'pandaset_{site}.yaml'),
                               'mode','trajectory'],cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(site+' failed')
    print('CORRECTED RENDER DONE',site,flush=True)

for site in ['034','109','110']:run(site)
