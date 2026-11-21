"""Calibrated ROI selection, ray/triangle visibility and sensor-derived foreground masks."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_returns import mask_at
from prepare_scans import camera_track
from road_geometry import glb_triangles, read_json

CAMERAS=('front_camera','front_left_camera','front_right_camera','left_camera','right_camera','back_camera')


@lru_cache(maxsize=18)
def track(raw,site,camera):
    src=Path(raw)/site
    times,poses=camera_track(src,camera)
    k=read_json(src/'camera'/camera/'intrinsics.json')
    K=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]])
    with Image.open(src/'camera'/camera/'00.jpg') as im:size=im.size
    return times,poses,K,size


def outline_points(path):
    samples=[]
    for code,v,f in glb_triangles(path/'asset.glb'):
        if code in (1,6,7):continue
        points=v
        if code==5:points=points[points[:,2]>-.025]
        if len(points):
            step=max(1,len(points)//140)
            extremes=np.r_[points.argmin(0),points.argmax(0)]
            samples.append(np.concatenate([points[::step],points[extremes]]))
    return np.concatenate(samples)


@lru_cache(maxsize=1)
def roof_profiles():
    return read_json(Path(__file__).with_name('ego_back_profile.json'))


def roof_edge(site, x):
    profile=roof_profiles()[site]
    return np.interp(x,profile['x'],profile['first_protected_y'])


def choices(raw,site,placement,points,names):
    """Rank complete in-frame silhouettes using real camera poses; retain a few alternatives."""
    world=points@np.asarray(placement['basis_columns']).T+placement['center_world']
    candidates=[]
    for camera in names:
        _,poses,K,(W,H)=track(raw,site,camera)
        cam=np.einsum('fpi,fij->fpj',world[None]-poses[:,None,:3,3],poses[:,:3,:3])
        ok=(cam[:,:,2]>.5).all(1)
        if not ok.any():continue
        ids=np.flatnonzero(ok);p=cam[ids];xy=p[:,:,:2]/p[:,:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
        lo,hi=xy.min(1),xy.max(1)
        sizes=hi-lo
        retained=(lo[:,0]>=10)&(hi[:,0]<W-10)&(lo[:,1]>=10)&(hi[:,1]<H-10)&(sizes[:,0]>9)&(sizes[:,1]>4)
        if camera=='back_camera':
            xs=(lo[:,0,None]+(hi[:,0]-lo[:,0])[:,None]*np.linspace(0,1,9))
            retained &= hi[:,1] < roof_edge(site,xs).min(axis=1)-3
        for n in np.flatnonzero(retained):
            a=lo[n];b=hi[n];score=float(np.prod(sizes[n]))
            candidates.append((score,camera,int(ids[n]),[float(a[0]),float(a[1]),float(b[0]),float(b[1])]))
    return sorted(candidates,reverse=True)


def crop_box(bounds,size):
    w,h=size;x0,y0,x1,y1=bounds
    aspect=1.68
    width=int(min(w,h*aspect,max(276,(x1-x0)*1.72,(y1-y0)*1.78*aspect)))
    height=int(width/aspect)
    cx,cy=(x0+x1)/2,(y0+y1)/2
    left=int(np.clip(round(cx-width/2),0,w-width));top=int(np.clip(round(cy-height/2),0,h-height))
    return [left,top,left+width,top+height]


def camera_hits(pose,K,rect,models):
    l,t,r,b=rect
    xx,yy=np.meshgrid(np.arange(l,r,dtype='f4')+.5,np.arange(t,b,dtype='f4')+.5)
    cam=np.stack(((xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)),axis=-1).reshape(-1,3)
    world=cam@pose[:3,:3].T
    depth=np.full(len(cam),np.inf,'f4');parts=np.zeros(len(cam),'u1');which=np.zeros_like(parts)
    normals=np.zeros((len(cam),3),'f4')
    for n,(p,mesh,field) in enumerate(models,1):
        B=np.array(p['basis_columns']);o=(pose[:3,3]-p['center_world'])@B
        v=(world@B).astype('f4')
        d,part,norm=mesh.cast(np.broadcast_to(o,v.shape),v)
        # Since directions have camera-z == 1, Embree t_hit is optical-axis depth.
        finite=np.isfinite(d);near=finite & (d>.05)&(d<depth)
        order=np.flatnonzero(near)
        if not len(order):continue
        z=o[2]+d[order]*v[order,2]
        buried=(part[order]>=2)&(z<-.0015)
        if buried.any():
            idx=order[buried];den=v[idx,2]
            good=(o[2]>.01)&(den<-1e-5)
            scale=np.divide(-o[2],den,out=np.zeros_like(den),where=good)
            entry=o[:2]+v[idx,:2]*scale[:,None]
            good &= mask_at(field['defect_mask'],float(field['width']),float(field['length']),entry)
            near[idx[~good]]=False
        ix=np.flatnonzero(near)
        depth[ix],parts[ix],which[ix]=d[ix],part[ix],n
        normals[ix]=norm[ix]@B.T
    shape=xx.shape
    return depth.reshape(shape),parts.reshape(shape),which.reshape(shape),normals.reshape((*shape,3))


@lru_cache(maxsize=6)
def static_scene_points(out,site,frame):
    with np.load(Path(out)/'base'/site/f'{frame:03d}.npz') as a:
        sem=a['semantic']
        keep=np.isin(sem,[5,6,11,12,34,35,36,37,38,39,40,41,42])
        return a['xyz'][keep]


def foreground(raw,proc,out,site,camera,frame,rect,depth):
    l,t,r,b=rect
    _,poses,K,(W,H)=track(raw,site,camera);pose=poses[frame]
    with Image.open(Path(proc)/site/'dynamic_masks/all'/f'{frame:03d}_{CAMERAS.index(camera)}.png') as im:
        dynamic=np.asarray(im.convert('L').resize((W,H),Image.Resampling.NEAREST).crop(rect))>0
    points=static_scene_points(str(out),site,frame)
    cam=(points-pose[:3,3])@pose[:3,:3]
    cam=cam[cam[:,2]>.05]
    xy=np.floor(cam[:,:2]/cam[:,2,None]*[K[0,0],K[1,1]]+[K[0,2]-l,K[1,2]-t]).astype('i4')
    keep=(xy[:,0]>=0)&(xy[:,0]<r-l)&(xy[:,1]>=0)&(xy[:,1]<b-t)
    z=np.full(depth.shape,1e5,'f4')
    np.minimum.at(z,(xy[keep,1],xy[keep,0]),cam[keep,2].astype('f4'))
    z=cv2.erode(z,np.ones((5,5),'u1'))
    static=np.isfinite(depth)&(z<depth-.12)
    ego=np.zeros(depth.shape,bool)
    if camera=='back_camera':
        ego=np.arange(t,b)[:,None]>=roof_edge(site,np.arange(l,r))[None,:]
    return dynamic,static,ego
