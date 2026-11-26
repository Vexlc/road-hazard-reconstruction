"""Compare apparent steel coverage in the delivered Gaussian PLY and the perforated mesh."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from gsplat.rendering import rasterization
from atlas_gaussians import load,gpu
from road_geometry import Surface,read_json,save_json,sha256

@torch.inference_mode()
def check(folder,sections,dest):
    meta=read_json(folder/"metadata.json");glb=folder/"asset.glb";ply=folder/"asset.ply"
    mesh=Surface(glb);dest.mkdir(exist_ok=True,parents=True)
    state=load(dict(id=meta["id"],gaussians=meta["gaussians"],ply=ply.resolve()))
    with np.load(folder/"splat_material.npz") as data:
        roles=data["part_names"][data["part"]]
    metal=torch.as_tensor(np.isin(roles,["Photographed corroded chequer plate","Rust perforation cut edge","Lid underside","Lid rolled edge"]),device="cuda")
    lid={k:v[metal] for k,v in state.items()}
    info={}
    with np.load(sections/"overhead.npz") as q:source=q["source_id"]
    h,w=source.shape;size=np.asarray(meta["plate_dimensions_m"])
    for name,eye,target,lens in [
        ("normal", [0,0,1.6], [0,0,0], 2550),
        ("near", [.04,-.42,.24], [0,-.055,0], 1080),
        ("reverse", [-.40,.16,.19], [0,-.055,0], 955),
    ]:
        eye,target=np.array(eye),np.array(target)
        forward=target-eye;forward/=np.linalg.norm(forward)
        up=np.array([0,1,0] if name=="normal" else [0,0,1])
        right=np.cross(forward,up);right/=np.linalg.norm(right)
        basis=np.array([right,np.cross(forward,right),forward])
        view=np.eye(4);view[:3,:3]=basis;view[:3,3]=-basis@eye
        width,height=1040,768
        K=np.array([[lens,0,width/2],[0,lens,height/2],[0,0,1]])
        def render(model):
            img,alpha,_=rasterization(**model,viewmats=gpu(view)[None],Ks=gpu(K)[None],width=width,height=height,
                        sh_degree=1,near_plane=.001,far_plane=6.,packed=False,rasterize_mode="antialiased",render_mode="RGB+ED",backgrounds=gpu([.39,.41,.41])[None])
            return img[0].cpu().numpy(),alpha[0,:,:,0].cpu().numpy()
        rgb,_=render(state)
        _,alpha=render(lid)
        yy,xx=np.indices((height,width))
        ray=np.stack([(xx+.5-width/2)/lens,(yy+.5-height/2)/lens,np.ones_like(xx)],-1)@basis
        origin=np.broadcast_to(eye,ray.shape);_,part,_=mesh.cast(origin,ray)
        cross=eye+ray*(-eye[2]/ray[:,:,2])[:,:,None]
        ij=(cross[:,:,:2]/size*[1,-1]+.5)*[w-1,h-1]
        interior=(abs(cross[:,:,:2])<(size/2-.004)).all(2)
        opening=interior&np.isin(part,[6,7,16]);solid=interior&np.isin(part,[163,164,10])
        clear=opening&(distance_transform_edt(opening)>=1.8)
        opaque=solid&(distance_transform_edt(solid)>=2.)
        col,row=np.rint(ij).astype(int).transpose(2,0,1)
        ids=source[row.clip(0,h-1),col.clip(0,w-1)]
        tested=[];resolved=[]
        for i in range(1,int(source.max())+1):
            m=opening&(ids==i)
            if m.any():
                tested.append(i)
                if np.min(alpha[m])<.5:resolved.append(i)
        Image.fromarray(np.uint8(rgb[:,:,:3].clip(0,1)*255+.5)).save(dest/(name+".jpg"),quality=95)
        np.savez_compressed(dest/(name+".npz"),mesh_part=part,plate_opacity=alpha.astype("f2"),source_id=ids)
        info[name]=dict(open_core_px=int(clear.sum()),false_closed_fraction=float((alpha[clear]>=.5).mean()),
                        steel_core_px=int(opaque.sum()),false_open_fraction=float((alpha[opaque]<.5).mean()),
                        p90_alpha_in_opening=float(np.percentile(alpha[clear],90)),
                        openings_tested=tested,openings_resolved=resolved,packet_sha256=sha256(dest/(name+".npz")))
    return dict(glb_sha256=sha256(glb),ply_sha256=sha256(ply),views=info)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("folder","sections","dest","report"):p.add_argument(k,type=Path)
    a=p.parse_args()
    save_json(a.report,check(a.folder,a.sections,a.dest))
