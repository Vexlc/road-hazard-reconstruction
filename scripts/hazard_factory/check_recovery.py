"""Exercise actual restart, corrupt-file repair, append and deterministic rebuild."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from recipes import recipe
from run import PROJECT, HERE, atomic_json, sha


def main(args):
    output, work = args.output.resolve(), args.work.resolve()
    launcher = [sys.executable,str(HERE/'run.py'),'--output',str(output),'--work',str(work)]
    manifest = lambda:json.loads((output/'asset_manifest.json').read_text(encoding='utf-8'))
    original = manifest()
    assert len(original) == 12
    original_ids = {m['id'] for m in original}
    with (work/'recovery_test.log').open('w',encoding='utf-8') as log:
        def execute(extra):
            subprocess.run(launcher+extra,stdout=log,stderr=subprocess.STDOUT,check=True)
            return json.loads((work/'last_run.json').read_text(encoding='utf-8'))
        cached = execute(['--count','12'])
        assert all(x['status']=='cached' for x in cached['assets'])
        baseline = {m['id']:m['quality'] for m in original}
        damaged = sorted(original,key=lambda m:m['stream_index'])[:2]
        (output/damaged[0]['preview']).write_bytes(b'truncated validation fixture')
        (output/damaged[1]['ply']).unlink()
        repaired = execute(['--count','12'])
        assert sum(x['status']=='built' for x in repaired['assets'])==2
        assert len(manifest())==12, 'quarantined copies leaked into registry'
        for m in manifest():
            for key in ('geometry_sha256','ply_sha256','glb_sha256'):
                assert m['quality'][key]==baseline[m['id']][key], (m['id'],key)
        # Interrupt the entire real process group after mesh output, before publication.
        before = set((output/'.staging').iterdir())
        child = subprocess.Popen(launcher+['--start','12','--count','3'],
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.monotonic()+45
        stopped = False
        while time.monotonic()<deadline and child.poll() is None:
            stages = set((output/'.staging').iterdir())-before
            if any(list(p.rglob('asset.glb')) for p in stages):
                os.killpg(child.pid,signal.SIGKILL); child.wait(); stopped=True;break
            time.sleep(.10)
        if not stopped:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise AssertionError('did not observe the intended partial-build interruption')
        assert {m['id'] for m in manifest()}==original_ids
        resumed = execute(['--start','12','--count','3'])
        assert len(manifest())==15 and original_ids.issubset({m['id'] for m in manifest()})
        cached_append = execute(['--start','12','--count','3'])
        assert all(x['status']=='cached' for x in cached_append['assets'])
    sys.path.insert(0,str(PROJECT/'outputs/road_hazard_expansion/reproduction'))
    from variant_catalog import variants
    catalog=list(variants())
    ids={recipe(20261120,i,catalog)['id'] for i in range(3000)}
    assert len(ids)==3000
    assert recipe(20261120,701,catalog)==recipe(20261120,701,catalog)
    assert recipe(20261120,701,catalog)['id']!=recipe(20261121,701,catalog)['id']
    report={'passed':True,'initial_assets':12,'final_assets':15,'cache_reuse_verified':12,
            'corrupted_preview_rebuilt':True,'deleted_ply_rebuilt':True,
            'rebuild_exact_geometry_ply_and_glb_hashes':True,'quarantine_excluded_from_registry':True,
            'real_mid_build_process_group_interrupted':True,'partial_build_not_published':True,
            'append_after_restart':3,'append_cache_verified':3,'recipe_ids_checked':3000,
            'recipe_ids_unique':True,'different_seed_distinct':True}
    atomic_json(output.parent/'生成稳定性验证.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    main(p.parse_args())
