"""Verify real append results against previously sealed native checkpoint hashes."""
import argparse
import json
from pathlib import Path
from run import atomic_json,sha


def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'));root=Path(job['output']).parent
    before=json.loads(args.before.read_text(encoding='utf-8'))
    for group in ('checkpoints','completions'):
        for path,digest in before[group].items():
            if sha(root/path)!=digest:raise ValueError(f'old artifact changed after append: {path}')
    old=before['groups']['groups'];current=json.loads((Path(job['work'])/'case_plan.json').read_text(encoding='utf-8'))['groups']
    if current[:len(old)]!=old:raise ValueError('previous scene groups renumbered or changed assets')
    count=len(list(Path(job['output']).glob('*/trained_model/iteration_7000.pth')))
    expected=len(current)*len(job['configs'])
    if count!=expected:raise ValueError(f'incomplete appended scenes: {count} != {expected}')
    for number in range(len(old)+1,len(current)+1):
        for site in job['configs']:
            path=Path(job['output'])/f'{site}_{number:02d}'
            if not (path/'complete.json').is_file():raise ValueError(f'uncommitted append: {path}')
    report={'passed':True,'sealed_old_checkpoints':len(before['checkpoints']),
            'old_checkpoints_byte_unchanged':True,'old_completion_records_byte_unchanged':True,
            'old_groups_unchanged':len(old),'new_groups':len(current)-len(old),
            'new_native_scene_versions':count-len(before['checkpoints']),
            'total_native_scene_versions':count,'scope':'Content hashes, group identity, and committed append results; full rendering checks are separate.'}
    atomic_json(args.output or root/'追加迁入稳定性验证.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--output',type=Path);main(p.parse_args())
