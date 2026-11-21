"""Build native PandaSet checkpoints in independent groups and render front-camera labels."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SITES=['034','109','110']
CATS=['pothole','missing','raised']
def groups():
    out=[dict(number=i,assets=[f'P{i:02}',f'M{i:02}',f'A{i:02}']) for i in range(4,36)]
    return out+[dict(number=36,assets=['P36','M36']),dict(number=37,assets=['P37','A36'])]
def save(p,val):
    tmp=p.with_suffix('.part');tmp.write_text(json.dumps(val,indent=1,ensure_ascii=False)+'\n');tmp.replace(p)
def run(project,v3,work,cache,group_ids,sites,mode):
    root=ROOT
    rows=json.loads((root/'assets/asset_manifest.json').read_text())
    lookup={m['id']:m for m in rows}
    gset=[g for g in groups() if g['number'] in group_ids]
    categories=[lookup[n]['category'] for n in gset[0]['assets']]
    if any([lookup[n]['category'] for n in g['assets']]!=categories for g in gset):raise ValueError('Use a separate native job for each slot layout')
    work=work/('groups_'+'_'.join(map(str,group_ids))); work.mkdir(parents=True,exist_ok=True)
    old=json.loads((v3/'本机运行配置.json').read_text())
    original=json.loads(Path(old['plan']).read_text());slots={'034':[0,2,4],'109':[0,2,4],'110':[0,1,3]}
    plan={'category_order':categories,'drive_start_by_site':original['drive_start_by_site'],
          'slots_by_site':{s:[original['slots_by_site'][s][sl[CATS.index(c)]] for c in categories] for s,sl in slots.items()}}
    save(work/'plan.json',plan); save(work/'case_plan.json',{'schema':1,'groups':gset})
    if not (work/'road_masks').exists():shutil.copytree(Path(old['work'])/'road_masks',work/'road_masks')
    out=root/'scenes';out.mkdir(exist_ok=True)
    adapter=Path(__file__).with_name('native_entry.py')
    job=dict(output=str(out),work=str(work),libraries=[str(root/'assets')],plan=str(work/'plan.json'),configs=old['configs'],
             annotated_road_cache=old['annotated_road_cache'],road_occlusion_guard=True,road_guard_spacing_m=.018,
             material_contact=False,mesh_cache=str(cache),case_groups=group_ids,image_fit_code=hashlib.sha256(adapter.read_bytes()).hexdigest(),
             image_labels={n:hashlib.sha256((root/'assets'/lookup[n]['ply']).with_name('splat_material.npz').read_bytes()).hexdigest() for g in gset for n in g['assets']})
    save(work/'job.json',job)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',HAZARD_FACTORY_JOB=str(work/'job.json'),ROAD_BATCH_WORK=str(work),ROAD_PROJECT=str(project),
             CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',
             PYTHONPATH=':'.join(str(project/p) for p in ['work/EasyVolcap','work/asset_pipeline','work/road_hazard_pipeline','outputs/road_hazard_expansion/reproduction','work/street_crafter']))
    script=adapter if mode=='native' else adapter.with_name('sensor_views.py')
    for site in sites:
        with (work/(mode+'_'+site+'.log')).open('w') as f:
            subprocess.run([str(project/'work/envs/streetcrafter/bin/python'),'-u',str(script),'--config',job['configs'][site],'mode','trajectory'],
                           cwd=project/'work/street_crafter',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(mode,site,group_ids,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','v3','work','cache']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--group-ids',type=int,nargs='+',required=True)
    p.add_argument('--sites',nargs='+',default=SITES);p.add_argument('--mode',choices=['native','sensor'],default='native')
    a=p.parse_args();run(a.project.resolve(),a.v3.resolve(),a.work.resolve(),a.cache.resolve(),a.group_ids,a.sites,a.mode)
