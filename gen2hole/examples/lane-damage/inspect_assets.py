"""Render each delivered Gaussian from two views and check its native mesh against source observations."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
import cv2,torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from photo_mesh import ROOT,LIB,grid,xyz,RoadCamera
sys.path.insert(0,str(LIB.parent/'scripts'))
from atlas_gaussians import load,gpu
from gsplat.rendering import rasterization
from road_geometry import Surface,read_json,save_json,sha256,glb_triangles
from importlib import import_module
view_rays=import_module('check-verge-chamber').view_rays
HERE=Path(__file__).resolve().parent


def scan_fit(folder,mesh,meta,run):
    with np.load(run/'evaluation'/folder.name/'survey.npz') as p:
        labels=p['defect'].astype(bool);hold=p['reserved'].astype(bool)
        measured=(p['xyz']-meta['source_origin_xyz'])@np.array(meta['source_local_basis_columns'])
    inside=measured[labels]
    dims=sorted(cv2.minAreaRect(inside[:,:2].astype('f4'))[1],reverse=True)
    points=measured[hold];r,part,_=mesh.cast(xyz(points[:,:2],1.),np.broadcast_to([0,0,-1.],points.shape))
    use=labels[hold]; errors=(1-r[use]-points[use,2])*1000
    if not np.isfinite(errors).all():raise ValueError('Missing survey return '+folder.name)
    return dict(annotated_point_extent_m=dims,reference_depth_p95_m=float(np.percentile(-inside[:,2],95)),
                held_out_pit_points=len(errors),mae_mm=float(abs(errors).mean()),p90_mm=float(np.percentile(abs(errors),90)),
                annotated_points=len(inside),heldout_pit_hits=int((part[use]==2).sum()))


def mask_iou(s,mesh,meta,key):
    w,h=s['size'];y,x=np.indices((h,w));px=np.stack([x,y],axis=-1)
    cam=RoadCamera(s);r=xyz((px-[cam.cx,cam.cy])/cam.f,1.)@cam.R;_,part,_=mesh.cast(np.broadcast_to(cam.eye,r.shape),r)
    def compare(poly,codes):
        truth=np.zeros(part.shape,'u1');cv2.fillPoly(truth,[np.int32(poly)],1)
        mask=np.isin(part,codes);return float((mask&truth.astype(bool)).sum()/(mask|truth.astype(bool)).sum())
    result=dict(rim_iou=compare(s['rim'],[2,4]),sample_parts=[int(part[y,x]) for x,y in s['sample_checks']])
    if key=='P551':result['water_iou']=compare(s['water'],[4])
    return result


def field_check(folder,surf,meta,spec,run):
    key=folder.name
    if key in ('P546','P547','P548','P549','P550'):return scan_fit(folder,surf,meta,run)
    s=spec[key]
    if key in ('P551','P554'):return mask_iou(s,surf,meta,key)
    if key=='M552':
        R=np.array(meta['source_camera_rotation']);K=np.array(meta['source_camera_intrinsics']);eye=np.array(meta['source_camera_eye_m'])
        px=np.array(s['sample_checks']);r=np.c_[(px-K[:2,2])/np.diag(K)[:2],np.ones(len(px))]@R
        _,part,_=surf.cast(np.broadcast_to(eye,r.shape),r)
        d,hits,_=surf.cast(np.array([[0.,0.,1.]]),np.array([[0.,0.,-1.]]))
        assert hits[0]==7 and abs(d[0]-1-meta['shaft_depth_model_m'])<.001
        return dict(visible_source_parts=part.tolist(),center_first_hit='shaft bottom',depth_model_m=meta['shaft_depth_model_m'],
                    far_corner_reprojection_px=meta['far_corner_residual_px'],rim_component_rms_px=meta['rim_rms_px'])
    if key=='A553':
        topology=import_module('check-perforated-covers').topology
        pieces=[p for p in glb_triangles(folder/'asset.glb') if p[0] in [9,10,11]]
        solids=[]
        for side in (-1,1):
            subset=[]
            for label,v,f in pieces:
                selected=f[(v[f,0].mean(1)*side)>0]
                used,new=np.unique(selected,return_inverse=True)
                if len(used):subset.append((label,v[used],new.reshape(-1,3)))
            solids.append(topology(subset))
        assert all(r['closed_oriented_mesh'] and r['through_holes']==0 for r in solids)
        return dict(connected_solids=2,components=solids)
    if key=='P556':
        near,far=[np.array(meta[k]) for k in ('near_lip_xyz_m','far_lip_xyz_m')]
        u=np.arange(10,91,10);t=np.linspace(.05,.95,15)[:,None,None]
        sample=((1-t)*near[u]+t*far[u]).reshape(-1,3)
        start=xyz(sample[:,:2],1.);ray=np.broadcast_to([0.,0.,-1.],start.shape)
        dist,part,_=surf.cast(start,ray)
        assert np.all(part==2) and np.all(dist>1.2)
        app=xyz(meta['approach_ray_starts_xy_m'],1.)
        dd,pp,_=surf.cast(app,np.broadcast_to([0.,0.,-1.],app.shape))
        assert np.all(pp==13) and np.allclose(dd,1.)
        return dict(modeled_gap_m=float(np.linalg.norm(near[50]-far[50])),bed_first_hits=len(part),
                    tested_bed_depth_m=[float(dist.min()-1),float(dist.max()-1)],pavement_first_hits=pp.tolist())
    return dict(panels=len(meta['panels']),modeled_high_to_low_m=float(surf.high[2]-min(min(q[2] for q in row['corner_xyz_m']) for row in meta['panels'])))


@torch.inference_mode()
def inspect(models,out,evidence_check=field_check,fit_tall=False):
    out.mkdir(parents=True,exist_ok=True);records={}
    spec=read_json(HERE/'photos.json')
    for path in sorted(models.glob('*/series/assets/*/*/metadata.json'),key=lambda q:int(q.parent.name[1:])):
        folder=path.parent;key=folder.name;meta=read_json(path);run=folder.parents[3];glb=folder/'asset.glb'
        surf=Surface(glb); row=dict(id=key,label=meta['label'],glb_sha256=sha256(glb),ply_sha256=sha256(folder/'asset.ply'),
                                   gaussians=meta['gaussians'],triangles=sum(meta['geometry_parts'].values()),evidence=evidence_check(folder,surf,meta,spec,run))
        with np.load(folder/'surface.npz') as field:
            mask=field['defect_mask']>.5
            if not mask.any():raise ValueError(key)
            row['mesh_depth_p95_m']=float(np.percentile(np.maximum(-field['height'][mask],0),95))
            row['mesh_height_p95_m']=float(np.percentile(np.maximum(field['height'][mask],0),95))
            yy,xx=np.nonzero(mask);extent=np.array([field['width'],field['length']]);shape=np.array(mask.shape[::-1])-1
            lo=(np.array([xx.min(),yy.max()])/shape-.5)*extent*[1,-1]
            hi=(np.array([xx.max(),yy.min()])/shape-.5)*extent*[1,-1]
            target=np.r_[(hi+lo)*.5,0.];span=max(hi-lo);row['installed_xy_aabb_m']=(hi-lo).tolist()
        state=load(dict(id=key,gaussians=row['gaussians'],ply=(folder/'asset.ply').resolve()))
        row['view']={}
        for name,offset in [('near',[.26,-.89,.88]),('opposite',[-.56,.80,.60])]:
            if key=='M552':offset=[-.48,-.37,1.04] if name=='near' else [.45,.24,.9]
            if key in ('P554','P555','P556'):
                offset=[.72,-.23,.49] if name=='near' else [-.68,.38,.67]
            if fit_tall:
                target[2]=meta['highest_z_m']*.47
            eye=target+np.array(offset)*span*(1.55 if fit_tall else 1.18)
            w,h,fx=720,520,718
            r,B=view_rays(eye,target,fx,w,h);view=np.eye(4);view[:3,:3]=B;view[:3,3]=-B@eye
            K=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1.]])
            image,alpha,_=rasterization(**state,viewmats=gpu(view)[None],Ks=gpu(K)[None],width=w,height=h,
                near_plane=.001,far_plane=60.,sh_degree=1,rasterize_mode='antialiased',render_mode='RGB+ED',backgrounds=gpu([.22,.24,.23])[None])
            rgba=image[0].cpu().numpy();alpha=alpha[0,:,:,0].cpu().numpy()
            d,part,_=surf.cast(np.broadcast_to(eye,r.shape),r)
            core=(distance_transform_edt(part>1)>=3)&np.isfinite(d)
            row['view'][name]=dict(core_px=int(core.sum()),opacity_p10=float(np.percentile(alpha[core],10)),
                                   mesh_splat_depth_p90_mm=float(1000*np.percentile(abs(rgba[:,:,3][core]-d[core]),90)))
            Image.fromarray(np.uint8(rgba[:,:,:3].clip(0,1)*255+.5)).save(out/(key+'-'+name+'.jpg'),quality=94)
        records[key]=row;save_json(out/'quality.json',records);print(key,flush=True)
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('out',type=Path)
    inspect(**vars(p.parse_args()))
