"""Place the nine assets at annotated v3 road anchors, keeping source scenes separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True,help="fu-xi project with StreetCrafter environment")
    ap.add_argument('--v3',type=Path,required=True,help="v3 output folder containing 本机运行配置.json")
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--mesh-cache',type=Path,required=True)
    ap.add_argument('--sites',nargs='+',default=['034','109','110'])
    args=ap.parse_args();project=args.project.resolve();work=args.work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    old=json.loads((args.v3/'本机运行配置.json').read_text())
    plan=json.loads(Path(old['plan']).read_text())
    slots={'034':[0,2,4],'109':[0,2,4],'110':[0,1,3]}
    plan={'category_order':['pothole','missing','raised'],'drive_start_by_site':plan['drive_start_by_site'],
          'slots_by_site':{site:[plan['slots_by_site'][site][i] for i in ids] for site,ids in slots.items()}}
    save(work/'plan.json',plan)
    shutil.copytree(Path(old['work'])/'road_masks',work/'road_masks',dirs_exist_ok=True)
    rows=json.loads((ROOT/'assets/asset_manifest.json').read_text())
    by_cat={k:sorted(a['id'] for a in rows if a['category']==k) for k in plan['category_order']}
    group_counts={len(v) for v in by_cat.values()}
    if len(group_counts)!=1 or 0 in group_counts:
        raise ValueError('Each group in this scene template uses one asset of each category')
    groups=[{'number':i+1,'assets':[by_cat[c][i] for c in plan['category_order']]} for i in range(next(iter(group_counts)))]
    save(work/'case_plan.json',{'schema':1,'groups':groups})
    out=ROOT/'scenes';out.mkdir(exist_ok=True)
    save(out/'placement_plan.json',dict(plan,groups=groups))
    adapter=Path(__file__).with_name('native_entry.py').resolve()
    job=dict(output=str(out),work=str(work),libraries=[str(ROOT/'assets')],plan=str(work/'plan.json'),
             configs=old['configs'],annotated_road_cache=old['annotated_road_cache'],road_occlusion_guard=True,
             road_guard_spacing_m=.018,material_contact=False,mesh_cache=str(args.mesh_cache.resolve()),
             image_fit_code=sha(adapter),image_labels={a['id']:sha((ROOT/'assets'/a['ply']).with_name('splat_material.npz')) for a in rows})
    save(work/'job.json',job)
    python=project/'work/envs/streetcrafter/bin/python'
    env=dict(os.environ,HAZARD_FACTORY_JOB=str(work/'job.json'),ROAD_BATCH_WORK=str(work),ROAD_PROJECT=str(project),
             CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',
             PYTHONPATH=':'.join(str(project/p) for p in ('work/EasyVolcap','work/asset_pipeline','work/road_hazard_pipeline',
                                'outputs/road_hazard_expansion/reproduction','work/street_crafter')))
    for site in args.sites:
        print('Migrating',site,flush=True)
        with (work/(site+'.log')).open('w') as stream:
            proc=subprocess.run([str(python),'-u',str(adapter),'--config',job['configs'][site],'mode','trajectory'],
                                cwd=project/'work/street_crafter',env=env,stdout=stream,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError('Failed '+site+'; see '+str(work/(site+'.log')))
        print('Reloaded',site,flush=True)


if __name__=='__main__':
    main()
