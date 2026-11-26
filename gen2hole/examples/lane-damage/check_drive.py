"""Verify the 18 road-camera clips against the original exposure and the delivered mesh."""
import argparse
from importlib import import_module
from pathlib import Path
import sys
import numpy as np
from photo_mesh import LIB
from road_geometry import Surface, read_json, save_json, sha256

sys.path.insert(0,str(LIB.parent/'scripts'))
exposure=import_module('check-pavement-pass').exposure


def check(models,raw,views,out):
    rows={}
    for front in sorted((views/'front').glob('*/placement.json')):
        key=front.parent.name
        (glb,)=models.glob(f'*/series/assets/*/{key}/asset.glb')
        mesh=Surface(glb);digest=sha256(glb);rows[key]={};anchors=[]
        for side in ('front','back'):
            folder=views/side/key
            record=read_json(folder/'placement.json')
            shot=exposure(folder,raw,mesh,digest)
            rows[key][side]=dict(frames=len(record['exposures']),representative_frame=shot['frame'],
                pixels=int(np.count_nonzero(shot['parts'])),max_ray_error_m=shot['error_m'],
                glb_sha256=digest)
            anchors.append(shot['matrix'])
        assert np.array_equal(*anchors)
    assert len(rows)==9
    save_json(out,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('models','raw','views','out'):p.add_argument(key,type=Path)
    check(**vars(p.parse_args()))
