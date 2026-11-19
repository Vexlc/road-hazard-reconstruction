"""Render completed reconstructions while the third reconstruction trains."""
import subprocess
from finish_pipeline import WORK,OUT,REPO,PY,ENV

for site in ['034','109']:
    print('EARLY RENDER START',site,flush=True)
    with (WORK/f'render_{site}_early.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([str(PY),str(WORK/'render_scenes.py'),'--config',str(WORK/f'pandaset_{site}.yaml'),
                               'mode','trajectory'],cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{site}: render failed')
    print('EARLY RENDER DONE',site,flush=True)
