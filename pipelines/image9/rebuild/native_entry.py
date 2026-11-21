"""StreetCrafter entry: use the v3 merger with image-part labels and all-geometry camera fitting."""
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch


JOB=json.loads(Path(os.environ['HAZARD_FACTORY_JOB']).read_text())
PROJECT=Path(os.environ['ROAD_PROJECT'])
sys.path.insert(0,str(PROJECT/'scripts/hazard_factory'))
import migrate
from migrate import rb


@lru_cache(maxsize=12)
def support(model_id):
    with np.load(Path(JOB['mesh_cache'])/(model_id+'.npz')) as data:
        names=data['part_names']
        labels=np.array([('Fracture' in n or 'Water' in n or n.startswith('Lid') or 'Cast ring' in n or 'collar' in n) for n in names])
        xyz=data['xyz'][np.unique(data['faces'][labels[data['face_part']]])]
    return xyz


def camera(anchor, model, angle=0):
    pts=support(model['id'])
    cat=model['category']
    initial=-.38 if model['state']=='side_hinge' else .18
    az=angle+(1.93 if model['state']=='end_hinge' else initial)
    direction=np.array([np.sin(az),-np.cos(az),1.12 if cat=='pothole' else 1.17])
    away=anchor.basis@direction;away/=np.linalg.norm(away)
    forward=-away
    right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    orient=np.stack([right,up,forward],1)
    projected=(pts@anchor.basis.T)@orient
    center=(projected.min(0)+projected.max(0))*.5
    q=projected-center
    distance=max(float((abs(q[:,0])*874/394-q[:,2]).max()),float((abs(q[:,1])*874/211-q[:,2]).max()),.5)
    target=anchor.center+orient@center
    return rb.camera_at(anchor.reference,target+away*distance,target,(960,540),874.)


@torch.no_grad()
def fit_image_asset(anchor, model):
    verts,field=rb.read_asset(model)
    local=verts[:,:3]
    with np.load(Path(model['ply']).with_name('splat_material.npz')) as data:
        road=data['road']
        ao=data['ambient_visibility'].astype('f4')/255.
    if len(road)!=len(local):raise ValueError('Splat/material order mismatch')
    dc,rest=rb.split_gaussian_coefficients(verts,anchor.basis)
    src=(dc*rb.C0+.5).clamp(0,1)
    photo_road=road&(ao>.76)&(verts[:,15]>.2)
    if photo_road.sum()<100:raise ValueError('Need a visible reference road apron')
    src_road=torch.median(src[photo_road],dim=0).values
    color=rb.tensor(anchor.target_color)
    luma=rb.tensor([.2126,.7152,.0722])
    src_linear=rb.linear(src_road)
    target=rb.linear(color)
    gain=((target@luma)/(src_linear@luma).clamp_min(.009)).clamp(.12,5.5)
    balance=((target+.008)/(src_linear+.008)/gain).clamp(.82,1.20)
    corrected=rb.srgb((rb.linear(src)*gain*balance).clamp(0,1))
    near=(field['defect_mask']>.5)
    pixel=float(field['width'])/(near.shape[1]-1)
    distance=distance_transform_edt(~near)*pixel
    d=rb.sample_field(dict(field, distance=distance),'distance',local[road])
    fall=rb.tensor((1-rb.smooth(d/.13))[:,None])
    host=rb.tensor(anchor.color(local[road]))
    # Preserve fine near-lip texture and match the colour of the surrounding target
    # road, including contact shadows below an elevated plate. The face, seating
    # ledge, collar, lining, water and underside retain their source colours.
    detail=(src[road]/src_road.clamp_min(.10)).clamp(.46,1.75)
    intrinsic=rb.srgb((rb.linear(host)*(.47+.53*rb.tensor(ao[road,None]))).clamp(0,1))
    corrected[road]=(intrinsic*(1+fall*.72*(detail-1))).clamp(0,1)
    rest=rest*gain.pow(.25).clamp(.72,1.4)
    rest[road]*=fall[:,None]*.55
    state={'xyz':rb.tensor(local@anchor.basis.T+anchor.center),'feature_dc':((corrected-.5)/rb.C0)[:,None].contiguous(),
           'feature_rest':rest.contiguous(),'scaling':rb.tensor(verts[:,16:19]),
           'rotation':rb.quat_mul(anchor.quaternion,rb.tensor(verts[:,19:23])),'opacity':rb.tensor(verts[:,15:16]),
           'semantic':torch.empty(len(verts),0,device='cuda')}
    anchor.record['image_material_fit']={'road_splats':int(road.sum()),'appearance_gain_linear':float(gain),
                                         'balance_rgb':balance.cpu().tolist(),'host_reference_srgb':color.cpu().tolist()}
    return state,anchor.apron(field),anchor.parents_to_remove(field),float(gain)


if __name__=='__main__':
    rb.Anchor.fit_asset=fit_image_asset
    migrate.camera=camera
    migrate.main()
