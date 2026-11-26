"""Render the two large road failures from an approaching 1.45 m camera."""
import argparse
from importlib import import_module
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image
from photo_mesh import LIB, normalize
from road_geometry import read_json
from mesh_transfer import RoadMesh

sys.path.insert(0, str(LIB.parent/'scripts'))
view = import_module('check-verge-chamber')


def film(folder, out):
    meta = read_json(folder/'metadata.json')
    if folder.name == 'P556':
        near,far = [np.asarray(meta[k])[50] for k in ('near_lip_xyz_m','far_lip_xyz_m')]
    else:
        corners=np.array(meta['panels'][0]['corner_xyz_m'])
        near,far=corners[2:].mean(0),corners[:2].mean(0)
    forward=normalize((far-near)[:2])
    mesh=RoadMesh(folder)
    out.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(out/(folder.name+'.mp4')),(768,432),fps=12,
                                       codec='libx264',output_params=['-crf','18'])
    writer.send(None)
    try:
        for i,t in enumerate(np.linspace(0,1,42)):
            # Ease to rest just ahead of the nearest surviving lip.
            distance=.85+4.4*(1-t)**2
            eye=np.r_[near[:2]-forward*distance,1.45]
            target=np.r_[near[:2]+forward*(2.2 if folder.name=='P556' else 1.),-.09]
            rays,_=view.view_rays(eye,target,630.,768,432)
            rgb,parts,_=view.render(mesh,eye,rays)
            if i==41:
                assert np.any(parts>0)
                Image.fromarray(rgb).save(out/(folder.name+'.jpg'),quality=94)
            writer.send(np.ascontiguousarray(rgb))
    finally:
        writer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('photos',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args()
    for key in ('P555','P556'):
        film(args.photos/'series/assets/pothole'/key,args.out)
