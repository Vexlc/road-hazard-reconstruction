"""Jointly place and render the full road-hazard library in native PandaSet scenes."""
import copy
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.ndimage import binary_erosion, map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

from create_scene import create_scene
from export_reconstruction import export_splat
from gaussian_assets import C0, ROOT, linear, srgb, tensor
from gaussian_road import sample_field
from polish_road_hazard import ATTRS, set_background
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer

LIBRARY = ROOT / "outputs/road_hazard_expansion"
OUTPUT = ROOT / "outputs/road_hazard_joint_scenes"
WORK = Path(os.environ["ROAD_BATCH_WORK"])
WIDTH, HEIGHT = 1600, 900


def unit(a):
    return a / np.linalg.norm(a)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def camera_at(reference, position, target, resolution, focal):
    result = copy.copy(reference)
    result.world_view_transform = reference.world_view_transform.clone()
    forward = unit(target-position)
    right = unit(np.cross(forward, [0.,0.,1.]))
    pose = np.eye(4)
    pose[:3,:3] = np.stack([right,np.cross(forward,right),forward],axis=1)
    pose[:3,3] = position
    w,h = resolution
    result.image_width, result.image_height = w,h
    result.set_intrinsic(tensor([[focal,0,w*.5],[0,focal,h*.5],[0,0,1]]))
    result.set_extrinsic(tensor(pose))
    return result


def numpy_image(result):
    rgb = result["rgb"]
    assert torch.isfinite(rgb).all()
    return np.uint8(rgb.permute(1,2,0).clamp(0,1).cpu().numpy()*255+.5)


def load_vertices(path, fields):
    a = PlyData.read(path,mmap="r")["vertex"].data
    return np.stack([a[k] for k in fields],axis=1)


def grid_from_axis(values):
    x,y = np.meshgrid(values,values)
    return np.c_[x.ravel(),y.ravel(),np.zeros(x.size,dtype="f4")]


def split_gaussian_coefficients(verts, basis):
    dc = tensor(verts[:,3:6])
    rest = tensor(verts[:,6:15].reshape(-1,3,3).transpose(0,2,1))
    linear_xyz = torch.stack([-rest[:,2],-rest[:,0],rest[:,1]],axis=1)
    coeff = torch.einsum("ij,njc->nic",tensor(basis),linear_xyz)
    return dc, torch.stack([-coeff[:,1],coeff[:,2],-coeff[:,0]],axis=1)


def quat_mul(q, p):
    w,v = q[:1],q[1:]
    a,b = p[:,:1],p[:,1:]
    scalar = w*a - (b*v).sum(-1,keepdim=True)
    vector = (w*b + a*v + torch.linalg.cross(v.expand_as(b),b))
    return torch.cat([scalar,vector],dim=1)


def read_asset(row):
    fields = ["x","y","z"]+[f"f_dc_{i}" for i in range(3)]+[f"f_rest_{i}" for i in range(9)]
    fields += ["opacity"]+[f"scale_{i}" for i in range(3)]+[f"rot_{i}" for i in range(4)]
    verts = load_vertices(LIBRARY/row["ply"],fields)
    with np.load(LIBRARY/row["surface"]) as f:
        field = dict(f)
    assert row["gaussians"] == len(verts)
    return verts,field


class Anchor:
    grid_half = 3.50
    color_grid_n = 601

    def __init__(self, record, centers, cameras, world, spread, teachers):
        self.record = record
        self.center = np.array(record["center_world"])
        self.basis = np.array(record["basis_columns"])
        distances = [(i,np.linalg.norm(c.camera_center.cpu().numpy()[:2]-self.center[:2]))
                     for i,c in enumerate(cameras)]
        # Keep the actors near the time the source vehicle passed this position.
        self.reference = cameras[min(distances,key=lambda p:abs(p[1]-4.0)
                                 if 39<=p[0]<=68 else 1e5)[0]]
        axis = np.linspace(-self.grid_half,self.grid_half,self.color_grid_n,dtype="f4")
        samples = grid_from_axis(axis) @ self.basis.T + self.center
        self.colors, self.coverage = bake_color(samples,teachers)
        self.colors = self.colors.reshape(self.color_grid_n,self.color_grid_n,3)
        core = grid_from_axis(np.arange(-1.44,1.45,.0078,dtype="f4"))
        wide = grid_from_axis(np.arange(-3.4,3.415,.022,dtype="f4"))
        wide = wide[np.max(np.abs(wide[:,:2]),axis=1)>1.434]
        flat = np.concatenate([core,wide])
        spacing = np.r_[np.full(len(core),.0078),np.full(len(wide),.022)]
        xy_world = (flat@self.basis.T+self.center)[:,:2]
        here = np.linalg.norm(flat[:,:2],axis=1)
        nearest_dist = np.min(np.linalg.norm(xy_world[:,None]-centers[None],axis=2),axis=1)
        subset = (here<3.35) & (here < nearest_dist+.016)
        self.flat = flat[subset]
        self.spacing = spacing[subset]
        self.apron_base_color = self.color(self.flat)
        self.apron_alpha = .99*smooth((3.35-here[subset])/.68)
        world_flat = (world-self.center)@self.basis
        mask = (np.max(np.abs(world_flat[:,:2]),axis=1)<4.25) & (np.abs(world_flat[:,2])<.42)
        self.parent_ids = np.flatnonzero(mask)
        self.parent_local = world_flat[mask]
        self.parent_radius = np.minimum(3.*spread[mask]+.04,.91)
        inner = self.color(np.array([[x,y,0] for x in [-.55,-.29,.21,.49] for y in [-.43,-.18,.12,.45]]))
        self.target_color = np.median(inner,axis=0)
        self.quaternion = tensor(Rotation.from_matrix(self.basis).as_quat()[[3,0,1,2]])

    def color(self, local):
        xy = ((local[:,:2]/self.grid_half+1)*.5*(self.color_grid_n-1)).T
        return np.stack([map_coordinates(self.colors[:,:,i],xy[::-1],order=1,mode="nearest")
                         for i in range(3)],axis=1)

    def apron(self, surface):
        alpha = sample_field(surface,"alpha",self.flat)
        height = sample_field(surface,"height",self.flat)
        # A continuous host-colored surface covers the tails of removed Gaussians.
        mix = self.apron_alpha*(1-smooth((alpha-.20)/.69))
        keep = mix>.012
        local = self.flat[keep].copy()
        local[:,2] = np.where(alpha[keep]>.003, height[keep]-.0018*alpha[keep],0)
        n = len(local)
        return {
            "xyz":tensor(local@self.basis.T+self.center),
            "feature_dc":tensor((self.apron_base_color[keep]-.5)/C0)[:,None],
            "feature_rest":torch.zeros(n,3,3,device="cuda"),
            "scaling":tensor(np.log(np.c_[self.spacing[keep]*.74,self.spacing[keep]*.74,np.full(n,.00052)])),
            "rotation":self.quaternion[None].expand(n,4),
            "opacity":torch.logit(tensor(mix[keep,None]).clamp(.00001,.99999)),
            "semantic":torch.empty(n,0,device="cuda"),
        }

    def parents_to_remove(self, surface):
        mask = surface["alpha"]>.03
        r,c = np.where(mask & ~binary_erosion(mask))
        edge = np.c_[(c/(mask.shape[1]-1)-.5)*float(surface["width"]),
                    (.5-r/(mask.shape[0]-1))*float(surface["length"])]
        distance = cKDTree(edge).query(self.parent_local[:,:2],workers=2)[0]
        distance[sample_field(surface,"alpha",self.parent_local)>.03] = 0
        return self.parent_ids[distance<self.parent_radius]

    def fit_asset(self, row):
        verts,surface = read_asset(row)
        local = verts[:,:3]
        opacity = tensor(verts[:,15:16])
        dc,rest = split_gaussian_coefficients(verts,self.basis)
        # Blend the patch margins into the observed host surface while keeping true
        # depths, lid material, and the central source RGB/color contrast.
        r = np.linalg.norm(local[:,:2],axis=1)
        local_z = local[:,2]
        if row["category"] in {"rgbd","stereo"}:
            local_sample = WORK.parent/"expanded_road_assets"/"samples"/(row["id"]+".npz")
            if local_sample.is_file():
                with np.load(local_sample) as seed:
                    blending = seed["road_mask"]
            else:
                blending = smooth((local_z+.016)/.021)
        elif row["category"] in {"missing","raised"}:
            blending = smooth((r-max(row["width_m"],row["length_m"])*.69)/.21)
        else:
            blending = smooth((local_z+.013)/.019)
        mix = blending*smooth((local_z+.021)/.016)*smooth((.022-local_z)/.012)
        if row.get("family")=="waterlogged":
            mix *= smooth((local_z+.018)/.012)
        flat = (mix>.67)&(verts[:,15]>-1.0)
        source = np.median(verts[flat,3:6],axis=0)*C0+.5 if flat.sum()>32 else np.median(verts[:,3:6],axis=0)*C0+.5
        weights = tensor([.2126,.7152,.0722])
        exposure = ((linear(tensor(self.target_color))@weights)/
                    (linear(tensor(np.clip(source,0,1)))@weights).clamp_min(.018)).clamp(.48,2.50)
        corrected = srgb((linear((dc*C0+.5).clamp(0,1)) * exposure).clamp(0,1))
        # Make photographed daylight and modeled road in shade fall into the same exposure.
        use = mix>.025
        sample = corrected.clone()
        if np.any(use):
            use_index=np.flatnonzero(use)
            sample[use_index] = tensor(self.color(local[use]))
        w=tensor(mix[:,None])
        new_dc=torch.lerp(corrected,sample,w)
        dyn_scale = (exposure**.32).clamp(.8,1.28)
        state = {
            "xyz":tensor(local@self.basis.T+self.center),
            "feature_dc":((new_dc-.5)/C0)[:,None].contiguous(),
            "feature_rest":(rest*(1-w[:,None])*dyn_scale).contiguous(),
            "scaling":tensor(verts[:,16:19]),
            "rotation":quat_mul(self.quaternion,tensor(verts[:,19:23])),
            "opacity":opacity,
            "semantic":torch.empty(len(verts),0,device="cuda"),
        }
        return state,self.apron(surface),self.parents_to_remove(surface),float(exposure)


@torch.no_grad()
def bake_color(points, teachers):
    xyz=tensor(points)
    best=torch.full((len(xyz),),-1.,device="cuda")
    colors=torch.zeros(len(xyz),3,device="cuda")
    for w,k,im,depth in teachers:
        q=xyz@w[:3,:3].T+w[:3,3]
        z=q[:,2]
        u=q[:,0]*k[0,0]/z+k[0,2]
        v=q[:,1]*k[1,1]/z+k[1,2]
        h, width=depth.shape[-2:]
        grid=torch.stack([2*(u+.5)/width-1,2*(v+.5)/h-1],1)[None,None]
        obs=F.grid_sample(depth,grid,align_corners=False)[0,0,0]
        col=F.grid_sample(im,grid,align_corners=False)[0,:,0].T
        err=(obs-z).abs()
        vis=(z>2)&(u>5)&(u<width-5)&(v>6)&(v<h-6)&(err<.60)
        score=torch.where(vis,1/z-.12*err,torch.full_like(z,-1))
        take=score>best
        best[take]=score[take]
        colors[take]=col[take]
    missing=(best<0).cpu().numpy()
    rgb=colors.cpu().numpy()
    if missing.any():
        seen=~missing
        assert seen.sum()>50
        indices=cKDTree(points[seen,:2]).query(points[missing,:2],workers=3)[1]
        rgb[missing]=rgb[seen][indices]
    return rgb,float(1-missing.mean())


def near_camera(anchor,row,width=880,height=506,angle=0.):
    e=max(row["width_m"],row["length_m"])
    lid=row["category"] in {"missing","raised"}
    side=row.get("state")=="removed_side"
    if lid:
        distance,rise = (2.35,1.85) if side else (1.25,1.12)
    else:
        # True scale is retained. The small measured samples use a nearer inspection camera.
        distance,rise = max(.55,e*1.48),max(.63,e*1.32)
    target=anchor.center+anchor.basis[:,2]*(.14 if lid else -.006)
    if side:
        target=target+anchor.basis[:,0]*.4
    offset=anchor.basis[:,1]*(-distance*np.cos(angle))+anchor.basis[:,0]*(distance*np.sin(angle))
    position=anchor.center+offset+anchor.basis[:,2]*rise
    return camera_at(anchor.reference,position,target,(width,height),width*.91)


def writer(path,fps,quality=8):
    return imageio.get_writer(path,fps=fps,codec="libx264",quality=quality,
                             macro_block_size=2,output_params=["-movflags","+faststart"])


def resize_cameras(cameras):
    for cam in cameras:
        k=cam.K.clone()
        k[0]*=WIDTH/cam.image_width
        k[1]*=HEIGHT/cam.image_height
        cam.image_width,cam.image_height=WIDTH,HEIGHT
        cam.set_intrinsic(k)


def save_checkpoint(g,cameras,site,out,full=False):
    checkpoint=g.save_state_dict(is_final=True)
    checkpoint["iter"]=7001
    model=out/"trained_model"
    model.mkdir(exist_ok=True)
    path=model/"iteration_7000.pth"
    tmp=model/"iteration_7000.tmp"
    torch.save(checkpoint,tmp)
    tmp.replace(path)
    if full:
        export_splat(g,cameras[38],out/"scene_frame038.ply")
    (out/"render_config.yaml").write_text(
        f"parent_cfg: {ROOT}/work/street_crafter/configs/pandaset_{site}_warmup.yaml\n"
        f"model_path: {out}\nmode: trajectory\nloaded_iter: 7000\nresolution: {WIDTH}\n"
    )
    return path.stat().st_size


@torch.no_grad()
def render_batch(scene,renderer,cameras,source,manifest,anchors,batch,out):
    g=scene.gaussians
    bg=g.background
    keep=torch.ones(len(source["xyz"]),dtype=torch.bool,device="cuda")
    staged,details,shots,background_shots=[],[],[],[]
    for i,name in enumerate(batch["assets"]):
        row=manifest[name]
        a=anchors[i]
        asset,apron,cut,exposure=a.fit_asset(row)
        keep[tensor(cut).long()]=False
        staged.extend([asset,apron])
        shots.append(near_camera(a,row,angle=.33 if row["category"]=="raised" else -.11))
        details.append({"asset_id":name,"category":row["category"],"width_m":row["width_m"],
                        "length_m":row["length_m"],"depth_m":row.get("depth_p95_m",row.get("depth_design_m")),
                        "size_multiplier":1.,"exposure_gain":exposure,
                        "asset_points":len(asset["xyz"]),"apron_points":len(apron["xyz"]),"cut_points":len(cut),
                        "host_color_coverage":a.coverage,**a.record})
    set_background(bg,source)
    for cam in shots:
        original=numpy_image(renderer.render(cam,g))
        background_shots.append(original)
    merged={k:torch.nn.Parameter(torch.cat([source[k][keep]]+[s[k] for s in staged]),requires_grad=False)
            for k in ATTRS}
    del staged
    set_background(bg,merged)
    for index,(cam,detail,before) in enumerate(zip(shots,details,background_shots),1):
        picture=numpy_image(renderer.render(cam,g))
        ny,nx=picture.shape[:2]
        diff=np.abs(before.astype("i2")-picture.astype("i2")).mean(axis=2)
        detail["contrast_pixels_near_center"]=int((diff[ny//4:3*ny//4,nx//4:3*nx//4]>8).sum())
        filename=f"close_{index:02d}.jpg"
        detail["preview_file"]=filename
        Image.fromarray(picture).save(out/filename,quality=91)
        if batch["showcase"]:
            Image.fromarray(before).save(out/f"original_{index:02d}.jpg",quality=91)
    for frame in (37,44):
        Image.fromarray(numpy_image(renderer.render(cameras[frame],g))).save(out/f"drive_{frame}.jpg",quality=93)
    if batch["showcase"]:
        with writer(out/"drive.mp4",10,8) as video:
            for cam in cameras[17:67]:
                video.append_data(numpy_image(renderer.render(cam,g)))
        with writer(out/"inspect.mp4",24,8) as video:
            for i,name in enumerate(batch["assets"]):
                row=manifest[name]
                a=anchors[i]
                for angle in np.linspace(-.42,.95 if row["category"]=="raised" else .28,40):
                    cam=near_camera(a,row,1280,720,float(angle))
                    video.append_data(numpy_image(renderer.render(cam,g)))
    size=save_checkpoint(g,cameras,batch["site"],out,batch["showcase"])
    set_background(bg,source)
    summary={**batch,"camera_method":"joint anisotropic splat rendering with original dynamic actors",
             "checkpoint_bytes":size,"removed_background_points":int((~keep).sum()),
             "composited_background_points":len(merged["xyz"]),"placements":details}
    (out/"scene.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")


@torch.no_grad()
def main():
    site=Path(cfg.source_path).name
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((WORK/"plan.json").read_text())
    models={r["id"]:r for r in json.loads((LIBRARY/"asset_manifest.json").read_text())}
    scene=create_scene()
    renderer=StreetGaussianRenderer()
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta["frame"])
    resize_cameras(cameras)
    bg=scene.gaussians.background
    source={k:getattr(bg,v) for k,v in ATTRS.items()}
    teacher=[]
    for f in range(18,70,2):
        cam=cameras[f]
        result=renderer.render(cam,scene.gaussians)
        teacher.append((cam.get_extrinsic(world2cam=True).clone(),cam.K.clone(),
                        result["rgb"][None].clone(),result["depth"][None].clone()))
    for frame in (37,44):
        original=numpy_image(renderer.render(cameras[frame],scene.gaussians))
        Image.fromarray(original).save(OUTPUT/f"original_{site}_{frame}.jpg",quality=94)
    recs=plan["slots_by_site"][site]
    centers=np.array([a["center_world"][:2] for a in recs])
    world=source["xyz"].cpu().numpy()
    spread=source["scaling"].exp().max(1).values.cpu().numpy()
    anchors=[Anchor(rec,centers,cameras,world,spread,teacher) for rec in recs]
    print("READY",site,[round(a.coverage,3) for a in anchors],flush=True)
    del teacher,result
    requested=set(os.environ.get("ROAD_ONLY_BATCHES","").split(","))-{""}
    for batch in plan["scenes"]:
        if batch["site"]!=site or (requested and batch["id"] not in requested):
            continue
        out=OUTPUT / ("scene_"+batch["id"])
        if (out/"scene.json").is_file():
            continue
        out.mkdir(exist_ok=True)
        tick=time.monotonic()
        render_batch(scene,renderer,cameras,source,models,anchors,batch,out)
        print("DONE",batch["id"],len(batch["assets"]),round(time.monotonic()-tick,2),flush=True)


if __name__=="__main__":
    main()
