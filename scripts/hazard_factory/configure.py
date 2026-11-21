"""Create machine-local factory configs after restoring the v3 download groups."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from run import PROJECT, atomic_json, sha


def main(args):
    root=PROJECT;work=args.work.resolve();out=args.output.resolve()
    work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    template=json.loads((root/'catalog/factory-v3-scene-template.json').read_text(encoding='utf-8'))
    configs={}
    for site in ('034','109','110'):
        config=dict(template,parent_cfg=str(root/'work/street_crafter/configs/waymo_val_121.yaml'),
                    source_path=str(root/'tools/multiscene/data/processed'/site),
                    model_path=str(root/'outputs/道路病害_多场景迁入/重建场景'/site),
                    record_dir=str(work/'training_records'/site),exp_name='road_diversity_'+site)
        path=work/f'pandaset_{site}.yaml'
        path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8');configs[site]=str(path)
    plan=work/'plan.json';shutil.copy2(root/'catalog/factory-v3-plan.json',plan)
    groups=work/'case_plan.json'
    if not groups.exists():
        backup=out/'场景分组计划.json'
        shutil.copy2(backup if backup.exists() else root/'catalog/factory-v3-groups.json',groups)
    # Existing PBR input packages use reconstruction-inputs/pbr; the factory
    # fingerprint and mesh generator share the original road_hazard_sources path.
    source=root/'work/reconstruction-inputs/pbr';destination=root/'work/road_hazard_sources'
    if source.is_dir():
        for path in source.rglob('*.png'):
            target=destination/path.relative_to(source)
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    job={'output':str(out/'迁入场景'),'work':str(work),'plan':str(plan),
         'libraries':[str(root/'outputs/road_hazard_expansion'),str(out/'资产库')],
         'configs':configs,'transformers_path':str(root/'work/factory-extra-deps'),
         'road_occlusion_guard':True,'road_guard_spacing_m':.018,'mesh_cache':str(work/'meshes'),
         'annotated_road_cache':str(work/'annotated_road'),'visibility_stride':1,'visibility_width':640,
         'material_contact':True,'contact_occlusion_strength':.6}
    atomic_json(work/'migration_job.json',job)
    # Image-based placement consumes this copy alongside the v3 output.
    atomic_json(out/'本机运行配置.json',job)
    relocated=[]
    for folder in sorted((out/'迁入场景').glob('*_*')):
        site=folder.name.split('_')[0];config=folder/'render_config.yaml';complete=folder/'complete.json'
        if site not in configs or not config.exists() or not complete.exists():continue
        old=sha(config)
        config.write_text(yaml.safe_dump({'parent_cfg':configs[site],'model_path':str(folder),
                         'mode':'trajectory','loaded_iter':7000,'resolution':960},sort_keys=False),encoding='utf-8')
        done=json.loads(complete.read_text(encoding='utf-8'))
        done['files']['render_config.yaml']=sha(config)
        done.setdefault('relocation',{'original_render_config_sha256':old})
        done['relocation']['checkpoint_bytes_unchanged']=True
        atomic_json(complete,done);relocated.append(folder.name)
    atomic_json(work/'relocation.json',{'project':str(root),'relocated_scene_configs':relocated,
                'checkpoint_data_modified':False,
                'scope':'Only local paths are rewritten. Changed job/config paths invalidate migration-cache signatures; use fresh groups or rerender when migrating.'})
    print('Configured job:',work/'migration_job.json')
    print('Restored scene configs relocated:',len(relocated))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=PROJECT/'work/hazard_factory_v3')
    p.add_argument('--output',type=Path,default=PROJECT/'outputs/hazard_factory_v3')
    main(p.parse_args())
