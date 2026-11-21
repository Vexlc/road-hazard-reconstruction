"""Reconstruct selected library assets from the packaged RGB, fits and relative depths."""
import argparse
import hashlib
import json
from pathlib import Path

import chambers
import pavement
from mesh_io import Mesh, ROOT, intact_pavement, source
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('keys', nargs='*')
    args=parser.parse_args()
    layout=json.loads((ROOT/'calibration/layout.json').read_text())
    for i,(key,cfg) in enumerate(layout.items()):
        if args.keys and key not in args.keys:
            continue
        mesh=Mesh()
        grain=intact_pavement(source(key),key,cfg)
        args.cache.mkdir(exist_ok=True,parents=True)
        Image.fromarray(grain).save(args.cache/(cfg['id']+'_ground.jpg'),quality=96,subsampling=0)
        field, fit = (pavement.build(mesh,key,cfg,grain) if cfg['category']=='pothole'
                      else chambers.build(mesh,key,cfg,layout,grain))
        meta=dict(id=cfg['id'],category=cfg['category'],label=cfg['label'],family=cfg.get('family',cfg.get('shape')),
                  source_image='references/'+key+'.png',source_sha256=hashlib.sha256((ROOT/'references'/(key+'.png')).read_bytes()).hexdigest(),
                  recipe_version='image-fit-v1',scale_kind='designed_metric',seed=311720+i,**fit)
        dest=ROOT/'assets'/cfg['category']/cfg['id']
        mesh.save(dest,args.cache,meta,field)
        print(cfg['id'],meta['lowest_z_m'],meta['highest_z_m'],sum(meta['geometry_parts'].values()),flush=True)


if __name__=='__main__':
    main()
