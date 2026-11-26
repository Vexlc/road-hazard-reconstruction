"""Evaluate source rays, occlusion order and novel viewpoints on the delivered verge chamber."""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import map_coordinates
from mesh_transfer import RoadMesh
from road_geometry import read_json,save_json,sha256,Surface

BASE=Path(__file__).resolve().parents[1]/'gen2hole/examples/verge-chamber'
sys.path.insert(0,str(BASE))
from projection import Projection


def render(mesh,eye,directions):
    ray=np.concatenate([np.broadcast_to(eye,directions.shape),directions],axis=-1)
    hits=mesh.scene.cast_rays(o3d.core.Tensor(ray.astype('f4')),nthreads=2)
    gid=hits['geometry_ids'].numpy(); face=hits['primitive_ids'].numpy(); weights=hits['primitive_uvs'].numpy()
    part=np.zeros(gid.shape,'u1');rgb=np.zeros((*gid.shape,3),'u1')+61
    for i,(code,tri,uv,tex) in mesh.surfaces.items():
        mask=gid==i
        if not mask.any():continue
        pair=weights[mask]
        b=np.column_stack([1-pair.sum(1),pair])
        coords=(uv[tri[face[mask]]]*b[:,:,None]).sum(1)*tex.shape[1::-1]-.5
        rgb[mask]=np.stack([map_coordinates(channel,coords[:,::-1].T,order=1,mode='nearest') for channel in tex.transpose(2,0,1)],axis=1)
        part[mask]=code
    return rgb,part,hits['t_hit'].numpy()


def view_rays(eye,target,fx,w,h):
    forward=target-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    basis=np.array([right,np.cross(forward,right),forward])
    yy,xx=np.indices((h,w))
    rays=np.stack([(xx+.5-w/2)/fx,(yy+.5-h/2)/fx,np.ones_like(xx)],axis=-1)@basis
    return rays,basis


def check(folder,photo,out,report):
    import torch
    from gsplat.rendering import rasterization
    from atlas_gaussians import load,gpu
    s=read_json(BASE/'survey.json');cam=Projection(s);source=np.asarray(Image.open(photo).convert('RGB'))
    mesh=RoadMesh(folder);out.mkdir(exist_ok=True,parents=True)
    yy,xx=np.indices(source.shape[:2]);pixels=np.stack([xx,yy],axis=-1)
    r=cam.world(pixels,0)-cam.eye
    rgb,part,d=render(mesh,cam.eye,r)
    Image.fromarray(rgb).save(out/'source-view.jpg',quality=95)
    err=abs(rgb.astype(int)-source).mean(2)
    witnesses=read_json(BASE/'witnesses.json')
    checks=[]
    models={}
    for expected,uv in witnesses['visible']:
        x,y=uv
        checks.append(dict(uv=uv,expected=expected,first=int(part[y,x]),error_rgb=float(err[y,x])))
    layers=[]
    for front,under,uv in witnesses['crossings']:
        if front not in models:models[front]=Surface(folder/'asset.glb',exclude={front})
        x,y=uv;ri=r[y,x][None,:];eye=cam.eye[None,:]
        farther,behind,_=models[front].cast(eye,ri)
        layers.append(dict(uv=uv,expected=[front,under],measured=[int(part[y,x]),int(behind[0])],
                           gap_mm=float((farther[0]-d[y,x])*np.linalg.norm(ri)*1000)))
    covered=part>0;stats={}
    for code in np.unique(part[covered]):
        samples=err[part==code]
        stats[str(code)]=dict(pixels=len(samples),mean_rgb_error=float(samples.mean()),p90_rgb_error=float(np.percentile(samples,90)))
    np.savez_compressed(out/'source-rays.npz',part=part,range_m=d*np.linalg.norm(r,axis=2),rgb_error=err.astype('f2'))
    native=read_json(folder/'metadata.json')
    state=load(dict(id=native['id'],gaussians=native['gaussians'],ply=(folder/'asset.ply').resolve()))
    views={}
    for name,eye in [('west',[-.68,-.22,.45]),('east',[.64,.32,.40]),('low',[.26,-.92,.19])]:
        eye=np.array(eye);target=np.array([0.,0.,-.065]);w,h,fx=800,560,890
        rays,basis=view_rays(eye,target,fx,w,h)
        rgb,labels,depth=render(mesh,eye,rays)
        view=np.eye(4);view[:3,:3]=basis;view[:3,3]=-basis@eye
        K=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]])
        with torch.inference_mode():
            img,alpha,_=rasterization(**state,viewmats=gpu(view)[None],Ks=gpu(K)[None],width=w,height=h,sh_degree=1,
                         near_plane=.001,far_plane=5.,rasterize_mode='antialiased',render_mode='RGB+ED',backgrounds=gpu([.24,.24,.24])[None])
        rgba=img[0].cpu().numpy();alpha=alpha[0,:,:,0].cpu().numpy()
        Image.fromarray(rgb).save(out/(name+'-mesh.jpg'),quality=94)
        Image.fromarray(np.uint8(rgba[:,:,:3].clip(0,1)*255+.5)).save(out/(name+'-splat.jpg'),quality=94)
        np.savez_compressed(out/(name+'.npz'),part=labels,mesh_z_m=depth,splat_z_m=rgba[:,:,3].astype('f2'),opacity=alpha.astype('f2'))
        bypart={}
        for code in np.unique(labels[labels>0]):
            mask=labels==code
            core=cv2.erode(mask.astype('u1'),np.ones((3,3),'u1')).astype(bool)
            entry=dict(visible_px=int(mask.sum()),core_px=int(core.sum()))
            if core.any():
                entry.update(alpha_p10=float(np.percentile(alpha[core],10)),depth_mae_mm=float(1000*abs(rgba[:,:,3]-depth)[core].mean()))
            bypart[str(code)]=entry
        views[name]=bypart
    save_json(report,dict(glb_sha256=sha256(folder/'asset.glb'),ply_sha256=sha256(folder/'asset.ply'),
              reference_sha256=sha256(photo),source_stats=stats,visible_witnesses=checks,occlusion_pairs=layers,
              wall_pair_fit_mm=(cam.fit_errors_m*1000).tolist(),views=views))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('folder','photo','out','report'):p.add_argument(n,type=Path)
    check(**vars(p.parse_args()))
