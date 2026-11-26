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


def check(models,raw,views,out,ids=None):
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
    expected=set(ids or ("P546","P547","P548","P549","P550","P551","M552","A553","P554"))
    assert set(rows)==expected, (set(rows),expected)
    save_json(out,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('models','raw','views','out'):p.add_argument(key,type=Path)
    p.add_argument('--ids',nargs='+',help='Expected asset IDs for an independently reconstructed series')
    check(**vars(p.parse_args()))
