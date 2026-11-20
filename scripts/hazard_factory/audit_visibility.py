"""Diagnose underground shaft visibility outside its road-plane aperture.

This is a conservative geometry/depth diagnostic, not a photorealism score.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull
import torch

import migrate as m
from run import atomic_json


def portal_support(cam,center,basis,model):
    """Projected opening incl. real frame-height variation and one AA pixel.

    At driving distances a whole opening can be only two pixels high; a pixel
    center outside the aperture can still legitimately integrate its interior.
    """
    theta=np.linspace(0,2*np.pi,129)
    exponent=model['exponent']
    xy=np.c_[np.sign(np.cos(theta))*abs(np.cos(theta))**(2/exponent)*model['width_m']*.5,
             np.sign(np.sin(theta))*abs(np.sin(theta))**(2/exponent)*model['length_m']*.5]*1.35
    local=np.concatenate([np.c_[xy,np.full(len(xy),z)] for z in (-.06,.06)])
    world=local@basis.T+center
    w=cam.get_extrinsic(world2cam=True).cpu().numpy();k=cam.K.cpu().numpy()
    q=world@w[:3,:3].T+w[:3,3]
    if np.all(q[:,2]<=0):return np.zeros((cam.image_height,cam.image_width),bool)
    q=q[q[:,2]>.05]
    if len(q)<3:return np.zeros((cam.image_height,cam.image_width),bool)
    uv=q[:,:2]/q[:,2:3]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    hull=ConvexHull(uv)
    image=Image.new('L',(cam.image_width,cam.image_height))
    ImageDraw.Draw(image).polygon([tuple(p) for p in uv[hull.vertices]],fill=255)
    return binary_dilation(np.array(image)>0,structure=np.ones((3,3),bool))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    job=json.loads(Path(os.environ['HAZARD_FACTORY_JOB']).read_text(encoding='utf-8'))
    root=Path(job['output']).parent;site=Path(m.rb.cfg.source_path).name
    plan=json.loads(Path(job['plan']).read_text(encoding='utf-8'))
    models={}
    for folder in map(Path,job['libraries']):
        models.update({x['id']:x for x in json.loads((folder/'asset_manifest.json').read_text(encoding='utf-8'))})
    scene=m.rb.create_scene();renderer=m.rb.StreetGaussianRenderer()
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
    width=int(job.get('visibility_width',640))
    m.rb.WIDTH,m.rb.HEIGHT=width,width*9//16;m.rb.resize_cameras(cameras)
    dest=Path(job.get('visibility_output',str(root/'可见性诊断')));dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted(Path(job['output']).glob(f'{site}_*/scene.json')):
        case=json.loads(path.read_text(encoding='utf-8'))
        if job.get('case_groups') and int(case['id'].split('_')[1]) not in job['case_groups']:continue
        state=torch.load(path.parent/'trained_model/iteration_7000.pth',map_location='cuda',weights_only=False)
        scene.gaussians.load_state_dict(state)
        original={k:getattr(scene.gaussians.background,a) for k,a in m.rb.ATTRS.items()}
        debug=[]
        for d in case['placements']:
            if d['category'] not in ('missing','raised'):continue
            start=d['merged_asset_offset'];stop=start+d['asset_points']
            height=(original['xyz'][start:stop]-m.rb.tensor(d['center_world']))@m.rb.tensor(np.array(d['basis_columns'])[:,2])
            indices=torch.nonzero(height<-.075).flatten()+start
            alternate=dict(original)
            dc=original['feature_dc'].clone();rest=original['feature_rest'].clone()
            dc[indices]=m.rb.tensor((np.array([1.,.04,.04])-.5)/m.rb.C0)[None]
            rest[indices]=0
            alternate['feature_dc']=torch.nn.Parameter(dc,requires_grad=False)
            alternate['feature_rest']=torch.nn.Parameter(rest,requires_grad=False)
            debug.append((d,alternate))
        start=plan['drive_start_by_site'][site]
        worst=None
        stride=int(job.get('visibility_stride',1))
        for frame in range(start,start+40,stride):
            m.rb.set_background(scene.gaussians.background,original)
            cam=cameras[frame];result=renderer.render(cam,scene.gaussians)
            picture=m.rb.numpy_image(result)
            depth=result['depth'][0].cpu().numpy()
            h,w=depth.shape;yy,xx=np.mgrid[:h,:w];k=cam.K.cpu().numpy()
            rays=np.stack([(xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones_like(xx)],axis=-1)
            pose=np.linalg.inv(cam.get_extrinsic(world2cam=True).cpu().numpy())
            hit=(rays*depth[...,None])@pose[:3,:3].T+pose[:3,3]
            union=np.zeros((h,w),bool);per_asset=[]
            for d,alternate in debug:
                model=models[d['asset_id']]
                if d['category'] not in ('missing','raised'):continue
                center=np.array(d['center_world']);basis=np.array(d['basis_columns'])
                local=(hit-center)@basis;eye=(pose[:3,3]-center)@basis
                t=-eye[2]/(local[...,2]-eye[2])
                road=eye[:2]+t[...,None]*(local[...,:2]-eye[:2])
                exponent=model['exponent'];scale=np.array([model['width_m'],model['length_m']])*.5
                cross=(np.sum(abs(road/scale)**exponent,axis=-1))**(1/exponent)
                inside=(np.sum(abs(local[...,:2]/scale)**exponent,axis=-1))**(1/exponent)
                depth_mask=((local[...,2]<-.075)&(local[...,2]>-model['depth_design_m']-.10)
                      &(inside<1.12)&(cross>1.35)&(t>0)&(t<1))
                # Depth is an expectation over blended splats and can miss visible walls.
                # Change only underground SH colors in memory, keeping optical weights
                # identical. A color response outside the aperture proves a contribution.
                direction=(rays@pose[:3,:3].T)@basis
                plane_t=-eye[2]/np.where(abs(direction[...,2])>1e-9,direction[...,2],1e-9)
                plane_xy=eye[:2]+plane_t[...,None]*direction[...,:2]
                plane_radius=np.sum(abs(plane_xy/scale)**exponent,axis=-1)**(1/exponent)
                outside=(plane_radius>1.35)&(plane_t>0)
                m.rb.set_background(scene.gaussians.background,alternate)
                alternate_picture=m.rb.numpy_image(renderer.render(cam,scene.gaussians))
                delta=np.max(abs(alternate_picture.astype('i2')-picture.astype('i2')),axis=2)
                raw_mask=(delta>8)&outside
                portal=portal_support(cam,center,basis,model)
                mask=raw_mask&~portal
                count=int(mask.sum());union|=mask
                per_asset.append({'asset_id':d['asset_id'],'possible_shaft_leak_pixels':count,
                                  'depth_diagnostic_pixels':int(depth_mask.sum()),
                                  'raw_outside_plane_response_pixels':int(raw_mask.sum()),
                                  'pixels_within_projected_portal_support':int((raw_mask&portal).sum()),
                                  'maximum_outside_color_response':int(delta[outside&~portal].max()) if (outside&~portal).any() else 0})
            record={'case':case['id'],'frame':frame,'resolution':[w,h],
                    'possible_shaft_leak_pixels':int(union.sum()),'assets':per_asset}
            records.append(record)
            if worst is None or record['possible_shaft_leak_pixels']>worst[0]:
                marked=picture.copy()
                marked[union]=(marked[union]*.3+np.array([255,30,50])*.7).astype('u1')
                worst=(record['possible_shaft_leak_pixels'],frame,picture,marked)
        Image.fromarray(np.concatenate([worst[2],worst[3]],axis=1)).save(dest/f'{case["id"]}_最差采样帧{worst[1]:02d}.jpg',quality=94)
        print('VISIBILITY',case['id'],'max suspect pixels',worst[0],flush=True)
        m.rb.set_background(scene.gaussians.background,original)
        del state,debug
    atomic_json(dest/f'{site}_可见性诊断.json',{'site':site,'records':records,
                'method':'underground-only SH color probe, unchanged geometry/opacity; response >8/255 outside aperture and its projected portal support',
                'portal_support':'1.35 aperture radius; +/-6cm frame-height envelope; 1 pixel AA dilation; raw plane-only responses retained separately',
                'sampling':f'{len(range(0,40,stride))} original driving frames per checkpoint; {width}x{width*9//16}',
                'review_required':any(r['possible_shaft_leak_pixels']>5 for r in records),
                'caveat':'Conservative diagnostic; review marked original renders before interpreting as a defect.'})


if __name__=='__main__':main()
