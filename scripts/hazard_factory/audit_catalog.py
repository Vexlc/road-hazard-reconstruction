"""Audit selected file identities, diversity and capture-provenance limitations."""
import argparse
from collections import Counter
import json
from pathlib import Path

from run import atomic_json,sha


def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'))
    root=Path(job['output']).parent
    groups=json.loads((Path(job['work'])/'case_plan.json').read_text(encoding='utf-8'))['groups']
    models={}
    for library in map(Path,job['libraries']):
        for row in json.loads((library/'asset_manifest.json').read_text(encoding='utf-8')):
            models[row['id']]=(library,row)
    order=['stereo','rgbd','pothole','missing','raised'];rows=[]
    for position,category in enumerate(order):
        ids=[g['assets'][position] for g in groups]
        if len(ids)!=len(set(ids)):raise ValueError(f'Repeated selected record within {category}')
        records=[]
        for name in ids:
            folder,m=models[name]
            assert m['category']==category
            hashes={}
            for key in ('glb','ply','surface'):
                path=folder/m[key]
                assert path.is_file() and path.stat().st_size>0
                hashes[key]=sha(path)
                expected=m.get('quality',{}).get(key+'_sha256',m.get(key+'_sha256'))
                if expected:assert hashes[key]==expected,(name,key)
            records.append({'id':name,'width_m':m['width_m'],'length_m':m['length_m'],
                            'depth_m':m.get('depth_p95_m',m.get('depth_design_m')),
                            'family':m.get('family'),'state':m.get('state'),'material':m['material'],
                            'source_dataset':m.get('source_dataset'),'source_frame':m.get('source_frame'),
                            'source_rgb_sha256':m.get('source_rgb_hash'),'geometry_source':m.get('geometry_source'),
                            'limits':m.get('limitations',[]),'hashes':hashes})
        for key in ('glb','ply'):
            if len({r['hashes'][key] for r in records})!=len(records):
                raise ValueError(f'Exact duplicate {key} payload in {category}')
        rows.append({'category':category,'unique_asset_records':len(records),
            'native_placements':len(records)*len(job['configs']),
            'family_counts':dict(Counter(r['family'] for r in records)),
            'state_counts':dict(Counter(r['state'] for r in records if r['state'])),
            'material_counts':dict(Counter(r['material'] for r in records)),
            'source_dataset_groups':dict(Counter(str(r['source_dataset']) for r in records if r['source_dataset'] is not None)),
            'physical_defect_deduplication':'not_verified' if category in ('stereo','rgbd') else 'synthetic_designs',
            'records':records})
    report={'passed':True,'unique_asset_records':sum(r['unique_asset_records'] for r in rows),
            'categories':rows,'groups':len(groups),'sites':list(job['configs']),
            'native_placements':sum(r['native_placements'] for r in rows),
            'all_selected_payloads_exist_and_hashes_match':True,
            'no_exact_glb_or_ply_duplicates_within_category':True,
            'caveat':'Distinct files and source frames do not establish distinct physical potholes. Capture deduplication and nominal RGB-D scale remain unverified.',
            'source_manifest_sha256':{str(Path(p)/'asset_manifest.json'):sha(Path(p)/'asset_manifest.json') for p in job['libraries']},
            'group_plan_sha256':sha(Path(job['work'])/'case_plan.json')}
    atomic_json(root/'资产覆盖与多样性核对.json',report)
    print('CATALOG VERIFIED',report['unique_asset_records'],'records;',report['native_placements'],'placements',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    main(p.parse_args())
