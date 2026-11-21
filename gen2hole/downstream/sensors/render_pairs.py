"""Render front and reverse views from the existing road checkpoints, aligned with physical part/depth maps."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from gsplat.rendering import rasterization, spherical_harmonics

from camera_geometry import choices, crop_box, camera_hits, foreground, outline_points, track
from prepare_scans import write_npz
from sunlight import cast_shadows
from road_geometry import Surface, load_inventory, save_json, sha256


def gather(checkpoint, placements, assets):
    state=torch.load(checkpoint,weights_only=False,map_location='cpu',mmap=True)['background']
    groups=[];models=[];points=[]
    for p in placements:
        root=assets[p['asset_id']][0]
        with np.load(root/'splat_material.npz') as f:
            road=f['road'].astype(bool);ao=f['ambient_visibility'].astype('f4')/255.
        i=p['merged_asset_offset'];j=i+p['asset_points']
        assert len(road)==j-i
        state_gpu={k:state[k][i:j].cuda() for k in ['xyz','rotation','scaling','opacity','feature_dc','feature_rest']}
        groups.append(dict(xyz=state_gpu['xyz'],quat=torch.nn.functional.normalize(state_gpu['rotation'],dim=-1),
            scale=state_gpu['scaling'].exp(),opacity=state_gpu['opacity'].sigmoid().flatten(),
            features=torch.cat([state_gpu['feature_dc'],state_gpu['feature_rest']],dim=1),
            solid=torch.from_numpy((~road).astype('f4')).cuda(),shadow=torch.from_numpy((road*.68*(1-ao)).astype('f4')).cuda()))
        with np.load(root/'surface.npz') as z:field={k:z[k] for k in ('width','length','defect_mask')}
        models.append((p,Surface(root/'asset.glb'),field))
        points.append(outline_points(root))
    splats={k:torch.cat([g[k] for g in groups]) for k in groups[0]}
    return splats,models,points


def render(splats,pose,K,rect,src,depth,part,which,protected,models,site,camera):
    l,t,r,b=rect
    def gpu(a):return torch.as_tensor(a,dtype=torch.float32,device='cuda').contiguous()
    extr=gpu(np.linalg.inv(pose))[None]
    kk=K.copy();kk[0,2]-=l;kk[1,2]-=t
    dirs=splats['xyz']-gpu(pose[:3,3])
    color=(spherical_harmonics(1,dirs,splats['features'])+.5).clamp(0,1)
    w=splats['solid'][:,None];s=splats['shadow'][:,None]
    channels=torch.cat((color*w,w,color*s,s),1)
    rendered,_,_=rasterization(splats['xyz'],splats['quat'],splats['scale'],splats['opacity'],channels,
         extr,gpu(kk)[None],r-l,b-t,near_plane=.03,far_plane=250.,rasterize_mode='antialiased',packed=True)
    im=rendered[0].cpu().numpy()
    interior=(part>1)
    # Preserve a one-pixel antialias fringe against the mesh silhouette. Keep the
    # independently weighted road-contact contribution where the actual apron exists.
    if interior.any():
        dist=cv2.distanceTransform((~interior).astype('u1'),cv2.DIST_L2,3)
        edge=np.clip(1.55-dist,0,1)
    else:edge=np.zeros_like(depth)
    solid=np.where(protected,0.,edge).astype('f4')
    apron=(~protected)&(part!=0)
    alpha=np.clip(solid*im[:,:,3]+apron*im[:,:,7],0,1)
    rgb=solid[:,:,None]*im[:,:,:3]+apron[:,:,None]*im[:,:,4:7]
    gains=[]
    for i,(p,_,_) in enumerate(models,1):
        road=(part==1)&(which==i)&~protected
        gain=np.ones(3)
        if road.sum()>90:
            sample=src[road].astype('f4')/255
            light=sample@np.array([.2126,.7152,.0722])
            usable=(light>=np.quantile(light,.16))&(light<=np.quantile(light,.73))
            target=np.median(sample[usable],axis=0)
            ref=np.asarray(p.get('image_material_fit',{}).get('host_reference_srgb',np.asarray(p['source_road_rgb'])/255.))
            logratio=np.log(np.maximum(target,.04)/np.maximum(ref,.04))
            gain=np.exp(.70*logratio.mean()+.38*(logratio-logratio.mean())).clip(.72,1.35)
            rgb[which==i]*=gain
        gains.append(gain.tolist())
    rgb=np.minimum(rgb,alpha[:,:,None])
    sun,shade=cast_shadows(site,camera,models,pose,K,rect,part,which,src,protected)
    light=1-sun[:,:,None]*(1-shade)
    out=np.rint(np.clip(((1-alpha[:,:,None])*(src/255.)+rgb)*light,0,1)*255).astype('u1')
    out[protected]=src[protected]
    return out,alpha,gains,sun


@torch.no_grad()
def run(parent,out,raw,proc,first,last):
    assets,scenes,_=load_inventory(parent)
    dest=out/'camera';dest.mkdir(parents=True,exist_ok=True)
    for case,(source,info) in sorted(scenes.items(),key=lambda e:(int(e[0].split('_')[1]),e[0])):
        if not first<=int(case.split('_')[1])<=last:continue
        folder=dest/case;folder.mkdir(exist_ok=True)
        site=info['site'];splats,models,points=gather(source.parent/'trained_model/iteration_7000.pth',info['placements'],assets)
        rows=[]
        for num,p in enumerate(info['placements'],1):
            aid=p['asset_id']
            for name in ('front_camera','back_camera'):
                ranked=choices(str(raw),site,p,points[num-1],[name]);best=None
                times,poses,K,size=track(str(raw),site,name)
                for _,_,frame,extent in ranked[:6]:
                    rect=crop_box(extent,size);pose=poses[frame]
                    depth,part,inst,norm=camera_hits(pose,K,rect,models)
                    dynamic,static,ego=foreground(str(raw),str(proc),out,site,name,frame,rect,depth)
                    protected=dynamic|static|ego
                    m=(inst==num)&(part>1);visible=m&~protected
                    n=int(visible.sum());total=int(m.sum())
                    if n<35:continue
                    if best is None or n>best[0]:
                        best=(n,frame,rect,depth,part,inst,norm,dynamic,static,ego,total)
                    if total>=35 and n/total>.93:break
                if best is None:
                    rows.append(dict(asset_id=aid,camera=name,available=False));continue
                n,frame,rect,depth,part,inst,norm,dynamic,static,ego,total=best
                protected=dynamic|static|ego;pose=poses[frame]
                with Image.open(raw/site/'camera'/name/f'{frame:02d}.jpg') as f:
                    src=np.asarray(f.convert('RGB').crop(rect))
                rgb,alpha,gains,sun=render(splats,pose,K,rect,src,depth,part,inst,protected,models,site,name)
                keep=(part>1)&~protected
                phys_part=np.where(keep,part,0).astype('u1')
                phys_which=np.where(keep,inst,0).astype('u1')
                metric=np.where(keep,depth,np.nan).astype('f4')
                norm[~keep]=0
                target=phys_which==num
                ys,xs=np.where(target)
                stem=aid+'_'+name.split('_')[0]
                Image.fromarray(src).save(folder/(stem+'_before.webp'),lossless=True,method=2)
                Image.fromarray(rgb).save(folder/(stem+'_after.webp'),lossless=True,method=2)
                write_npz(folder/(stem+'_geometry.npz'),depth_z_m=metric,part=phys_part,instance=phys_which,
                    world_normal=norm.astype('f2'),sun_shadow=np.rint(sun*255).astype('u1'),composite_alpha=np.rint(alpha*255).astype('u1'),
                    protected=(dynamic.astype('u1')+2*static.astype('u1')+4*ego.astype('u1')))
                record=dict(asset_id=aid,instance=num,camera=name,available=True,site=site,case=case,
                    frame=frame,camera_timestamp=float(times[frame]),calibration_size=list(size),crop_xyxy=rect,
                    xyxy_target_in_crop=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                    K_full=K.tolist(),camera_to_world=pose.tolist(),srgb_gain_by_instance=gains,visible_target_pixels=n,
                    occluded_target_pixels=total-n,cast_shadow_pixels=int((sun>.20).sum()),rendered_target_pixels=int((target&(alpha>.48)).sum()),
                    mean_target_rgb_difference=float(abs(rgb.astype('f4')-src).mean(axis=2)[target].mean()),
                    before_stem=stem+'_before.webp',after_stem=stem+'_after.webp',geometry=stem+'_geometry.npz')
                rows.append(record)
        save_json(folder/'views.json',rows)
        print(case,'paired crops',len([v for v in rows if v['available']]),flush=True)
        del splats,models,points
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['parent','out','raw','proc']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--first',type=int,default=1);p.add_argument('--last',type=int,default=37)
    a=p.parse_args();run(a.parent,a.out,a.raw,a.proc,a.first,a.last)
