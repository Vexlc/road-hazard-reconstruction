"""Append-only five-category scene groups; seed/index sorting never renumbers old groups."""
import json
from pathlib import Path
from run import atomic_json,lock


def build(job,plan,models,factory):
    path=Path(job['work'])/'case_plan.json'
    with lock(path.with_suffix('.lock')):
        saved=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'schema':1,'groups':[]}
        if saved['schema']!=1:raise ValueError('unsupported scene group plan version')
        categories=plan['category_order']
        assigned=set()
        for group in saved['groups']:
            for category,name in zip(categories,group['assets']):
                if name not in models or models[name]['category']!=category:
                    raise ValueError(f'previously assigned asset missing or reclassified: {name}')
                if category in ('pothole','missing','raised'):assigned.add(name)
        pending={c:sorted([m for m in factory if m['category']==c and m['id'] not in assigned],
                         key=lambda m:(m['stream_seed'],m['stream_index'],m['id']))
                 for c in ('pothole','missing','raised')}
        sizes=[len(v) for v in pending.values()]
        if len(set(sizes))!=1:raise ValueError('append a complete balanced procedural triplet before joint migration')
        changed=False
        for index in range(sizes[0]):
            number=len(saved['groups'])+1
            assets=[plan['selected_assets'][c][(number-1)%len(plan['selected_assets'][c])]
                    if c in ('stereo','rgbd') else pending[c][index]['id'] for c in categories]
            saved['groups'].append({'number':number,'assets':assets});changed=True
        if changed or not path.exists():atomic_json(path,saved)
    return saved['groups']
