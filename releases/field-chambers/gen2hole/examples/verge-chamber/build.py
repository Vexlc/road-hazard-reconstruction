"""Build the open grass-verge utility chamber observed by Richard Webb."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
GEN=HERE.parents[1]
sys.path[:0]=[str(GEN),str(GEN/'recipes/common'),str(GEN/'recipes/series'),str(HERE.parent/'road-distress'),
              str(GEN/'downstream/sensors')]
from build_chamber import flatten_scene
from export_splats import make_one
from mesh_io import Mesh, write_json
from road_geometry import read_json, sha256, Surface
from projection import Projection
from surfaces import case, materials, photographed, tube, shaft_wall

CASES={'blue_case':'Blue instrument enclosure','pale_case':'Pale junction housing','white_case':'White rounded enclosure'}
TUBES={'metal_spindle':'Exposed metal spindle','pipe_and_coupling':'Pipe coupling','pipe_elbow':'Pipe elbow',
       'yellow_wire':'Yellow insulated lead','grey_wire':'Grey insulated lead',
       'dry_stem':'Crossing dry stem','right_stem':'Right-hand dry stem'}


def build(source,run,mesh_only=False):
    s=read_json(HERE/'survey.json')
    key=s['id'];photo=source/s['file']
    if sha256(photo)!=s['sha256']:raise ValueError('The original survey image has changed')
    mesh=Mesh();cam=Projection(s)
    rgb=np.asarray(Image.open(photo).convert('RGB'))
    mat=materials(mesh,rgb,s)
    ground=[[0,0],[639,0],[639,479],[0,479]]
    outer=s['concrete_extent_px']; seat=s['outer_seat_px']; lip=s['rim_inner_top'];floor=s['floor_visible']
    depth=-s['floor_depth_m']
    photographed(mesh,cam,'Grass and soil beside chamber',[ground,outer],[0.,0.],mat['frame'],True)
    photographed(mesh,cam,'Concrete frame on verge',[outer,seat],[0.,0.],mat['frame'])
    photographed(mesh,cam,'Cast ring and cover seat',[seat,lip],[0.,-s['seat_drop_m']],mat['frame'])
    shaft_wall(mesh,cam,s,np.asarray(mesh.materials[mat['frame']][0]))
    photographed(mesh,cam,'Leaf litter in chamber',[floor],[depth],mat['floor'])
    for k,name in CASES.items():
        # An independent side material avoids reprinting the label on the hidden face.
        sample={'blue_case':(170,277),'pale_case':(348,245),'white_case':(184,348)}[k]
        x,y=sample
        side=mesh.material(name+' hidden side',np.tile(np.median(rgb[y-2:y+3,x-2:x+3],axis=(0,1))[None,None,:],(8,8,1))*.82,.81,0,0)
        case(mesh,cam,name,s['objects'][k],depth,mat['equipment'],side)
    for k,name in TUBES.items():
        tube(mesh,cam,name,s['pipes'][k],mat['stems'] if k.endswith('stem') else mat['equipment'])

    series=run/'series';dest=series/'assets/missing'/key
    refs=series/'references';refs.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(photo,refs/(key+'.jpg'))
    span=np.ceil(np.max(np.abs(cam.world(ground,0)[:,:2]),axis=0)*2000)/1000
    shape=(385,545)
    y,x=np.meshgrid(np.linspace(span[1]/2,-span[1]/2,shape[0]),np.linspace(-span[0]/2,span[0]/2,shape[1]),indexing='ij')
    pos=np.stack([x,y,np.zeros_like(x)],axis=-1)
    uv=cam.pixels(pos)
    margin=np.minimum(uv,cam.size-1-uv).min(axis=2)
    alpha=np.clip(margin/13,0,1)
    field=dict(width=span[0],length=span[1],height=np.zeros(shape),alpha=alpha)
    meta=dict(id=key,label=s['label'],category='missing',family='verge-chamber',seed=627073,
              source=s['source'],source_image='references/'+key+'.jpg',source_sha256=s['sha256'],
              source_author=s['author'],source_license=s['license'],surface_dimensions_m=span.tolist(),depth_p95_m=-depth,
              scale_basis='Single photograph; opening width, heights and hidden surfaces inferred.',
              host='vegetated roadside verge',source_view_center_m=cam.eye.tolist(),
              wall_pair_fit_residual_m=cam.fit_errors_m.tolist())
    mesh.save(dest,series/'mesh_cache',meta,field);flatten_scene(dest/'asset.glb')
    native=Surface(dest/'asset.glb')
    pos[:,:,2]=.06;dist,part,_=native.cast(pos,np.broadcast_to([0.,0.,-1.],pos.shape))
    field['height']=np.where(np.isfinite(dist),.06-dist,0.)
    field['opening_mask']=(part!=0)&(field['height'] < -.027)
    field['installation_mask']=np.isin(part,[5,6,8,*range(165,174)])
    field['defect_mask']=field['opening_mask']
    np.savez_compressed(dest/'surface.npz',**{k:np.asarray(v,'f4') for k,v in field.items()})
    row=read_json(dest/'metadata.json')
    if not mesh_only:
        row=make_one(dest,series/'mesh_cache',point_limit=820000,kernel_factors=(.75,.14),
                     part_importance={**{n:12 for n in CASES.values()},**{n:18 for n in TUBES.values()},
                                      'Leaf litter in chamber':6,'Grass and soil beside chamber':5,'Inspection shaft':5})
    for role,filename in [('glb','asset.glb'),('metadata','metadata.json'),('surface','surface.npz')]+([] if mesh_only else [('ply','asset.ply')]):
        row[role]=f'missing/{key}/{filename}'
    write_json(series/'assets/asset_manifest.json',[row]);write_json(run/'run.json',dict(editions=['series'],assets=[key]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','run'):parser.add_argument(arg,type=Path)
    parser.add_argument('--mesh-only',action='store_true')
    build(**vars(parser.parse_args()))
