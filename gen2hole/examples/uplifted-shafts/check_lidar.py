"""Replay the two measured LiDARs using the same A563 world placement as the cameras."""
import argparse,json,sys
from collections import Counter
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lane-damage'))
from photo_mesh import LIB,Surface,write_json
sys.path.insert(0,str(LIB.parent/'scripts'))
lidar=import_module('replay-prop-lidar')


def check(models,camera,raw,out):
    folder=models/'urayasu/series/assets/raised/A563';surface=Surface(folder/'asset.glb')
    records=[json.loads((camera/side/'A563/placement.json').read_text()) for side in ('front','back')]
    assert records[0]['anchor']==records[1]['anchor']
    frames=sorted({e['scan'] for record in records for e in record['exposures']})
    record=records[0].copy();anchor=record['anchor']
    record['world_xy_bounds_m']=lidar.world_bounds(anchor,[surface.low,surface.high]).tolist()
    record['frames']=[{'lidar_frame':f,'visible_px':True} for f in frames]
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        d=Path(temp);source=d/'placements/A563';source.mkdir(parents=True)
        write_json(source/'placement.json',record)
        stage=d/'packets';stage.mkdir()
        lidar.replay(SimpleNamespace(models=folder.parent,views=source.parent,raw=raw),stage)
        counts=Counter();differences=[]
        for f in sorted((stage/'A563').glob('*.npz')):
            with np.load(f) as q:
                hit=q['hit_part']>0
                counts.update(q['hit_part'][hit].astype(int).tolist())
                diff=q['range_before_m']-q['range_after_m']
                assert np.all(diff[hit]>.003) and np.all(diff[~hit]==0)
                differences.extend(diff[hit].tolist())
        assert differences and counts[9]>0 and counts[103]>0
        out.mkdir(exist_ok=True)
        import shutil
        shutil.copytree(stage,out,dirs_exist_ok=True)
        write_json(out/'parts.json',dict(changed_beams=len(differences),semantic_hits=dict(counts),
            shortening_median_m=float(np.median(differences)),shortening_p95_m=float(np.percentile(differences,95))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('models','camera','raw','out'):p.add_argument(key,type=Path)
    check(**vars(p.parse_args()))
