"""Hybrid mesh/3DGS insertion with metric road fitting and calibrated multi-frame views."""
import os,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from PIL import Image
from plyfile import PlyData
import imageio.v2 as imageio
from scipy.ndimage import map_coordinates
from create_scene import create_scene
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'outputs/road_assets'

def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device='cuda').contiguous()
def unit(a):return a/torch.linalg.norm(a,dim=-1,keepdim=True).clamp_min(1e-8)
def linear(c):return torch.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
def srgb(c):return torch.where(c<=.0031308,c*12.92,1.055*c.clamp_min(0)**(1/2.4)-.055)

def fit_road(source, camera, target=None, radius=5):
    center=camera.get_extrinsic().cpu().numpy()[:3,3]
    c2w=camera.get_extrinsic().cpu().numpy()
    if target is None:target=center+c2w[:3,2]*9
    xyz=[]
    for frame in range(30,51,2):
        p=PlyData.read(source/'lidar_forward/background'/f'{frame:03d}.ply')['vertex']
        p=np.stack([p[k] for k in ['x','y','z']],-1)
        take=(np.linalg.norm(p[:,:2]-target[:2],axis=1)<radius)&(p[:,2]<center[2]-1.1)&(p[:,2]>center[2]-3.2)
        xyz.append(p[take])
    xyz=np.concatenate(xyz);A=np.c_[xyz[:,:2],np.ones(len(xyz))];z=xyz[:,2]
    # Robustly reject sidewalk points and low vehicle body returns.
    b=np.linalg.lstsq(A,z,rcond=None)[0]
    for _ in range(12):
        residual=z-A@b;med=np.median(residual);scale=max(.012,np.median(abs(residual-med))*1.4826)
        keep=abs(residual-med)<min(.10,2.8*scale)
        b=np.linalg.lstsq(A[keep],z[keep],rcond=None)[0]
    normal=np.r_[-b[:2],1];normal/=np.linalg.norm(normal)
    right=c2w[:3,0];right-=normal*np.dot(right,normal);right/=np.linalg.norm(right)
    forward=np.cross(normal,right)
    return b,np.stack([right,forward,normal],1),{'points':len(xyz),'inliers':int(keep.sum()),'rms_m':float(np.sqrt(np.mean((z[keep]-A[keep]@b)**2)))}

class Asset:
    def __init__(self,name,center,basis):
        self.name=name;self.center=np.array(center);self.basis=basis
        d=ASSETS/name;a=np.load(d/'mesh.npz');surface=np.load(d/'surface.npz')
        self.surface=surface;self.width=float(surface['width']);self.length=float(surface['length'])
        self.v=tensor(a['vertices']@basis.T+center);self.f=torch.tensor(a['faces'],dtype=torch.int32,device='cuda').contiguous()
        self.uv=tensor(a['uv']);self.n=tensor(a['normals']@basis.T)
        def tex(file):
            im=Image.open(d/file)
            size=tuple(2**int(np.ceil(np.log2(v))) for v in im.size)
            if size!=im.size:im=im.resize(size,Image.Resampling.BILINEAR)
            im=np.asarray(im)
            im=im.astype('f4')/(65535 if im.dtype.itemsize==2 else 255)
            if im.ndim==2:im=im[...,None]
            return tensor(im)[None]
        self.color=tex('basecolor.png');self.normal=tex('normal.png');self.arm=tex('metallic_roughness.png')
        self.gain=tensor([1,1,1]);self.b=tensor(basis);self.tint=tensor([1,1,1]);self.neutral=.0
        self.ao=tex('ambient_occlusion.png') if (d/'ambient_occlusion.png').exists() else None

    def inside(self,world,pad=0):
        local=(world-self.center)@self.basis
        row=(.5-local[:,1]/self.length)*(self.surface['alpha'].shape[0]-1)
        col=(local[:,0]/self.width+.5)*(self.surface['alpha'].shape[1]-1)
        a=map_coordinates(self.surface['alpha'],[row,col],order=1,mode='constant',cval=0)
        return (a>.015)&(abs(local[:,2])<.35)

    def draw(self,ctx,camera,ss=2):
        H,W=camera.image_height,camera.image_width
        w2c=camera.get_extrinsic(world2cam=True);K=camera.K
        q=self.v@w2c[:3,:3].T+w2c[:3,3];z=q[:,2]
        clip=torch.stack([2*(q[:,0]*K[0,0]+z*K[0,2])/W-z,2*(q[:,1]*K[1,1]+z*K[1,2])/H-z,(1000.001/999.999)*z-2*.001*1000/999.999,z],1)
        rast,deriv=dr.rasterize(ctx,clip[None],self.f,resolution=[H*ss,W*ss])
        uv,duv=dr.interpolate(self.uv[None],rast,self.f,rast_db=deriv,diff_attrs='all');uv=uv.clone();uv[...,1]=1-uv[...,1];duv=duv.clone();duv[...,2:]*=-1
        color=dr.texture(self.color,uv,uv_da=duv,filter_mode='linear-mipmap-linear',boundary_mode='clamp')
        arm=dr.texture(self.arm,uv,filter_mode='linear',boundary_mode='clamp')
        ns=dr.texture(self.normal,uv,filter_mode='linear',boundary_mode='clamp')[...,:3]*2-1
        geometric,_=dr.interpolate(self.n[None],rast,self.f)
        ns=ns@self.b.T
        N=unit(.70*unit(geometric)+.30*unit(ns))
        # Fixed world lighting gives stable appearance as the camera moves.
        light=unit(tensor([-.35,-.25,1.]));flat=(self.b[:,2]*light).sum().clamp_min(.1)
        ndl=(N*light).sum(-1,keepdim=True).clamp_min(0)
        lit=(.74+.26*ndl)/(.74+.26*flat)
        position,_=dr.interpolate(self.v[None],rast,self.f)
        V=unit(camera.camera_center-position);half=unit(V+light);rough=arm[...,1:2].clamp(.25,1);metal=arm[...,2:3]
        raw=linear(color[...,:3]);luma=(raw*tensor([.2126,.7152,.0722])).sum(-1,keepdim=True)
        base=((1-self.neutral)*raw+self.neutral*luma*self.tint)*self.gain
        ao=dr.texture(self.ao,uv,filter_mode='linear',boundary_mode='clamp')[...,:1] if self.ao is not None else torch.ones_like(ndl)
        f0=.04*(1-metal)+base*metal
        nh=(N*half).sum(-1,keepdim=True).clamp(0,1)
        spec=(nh**(2/(rough**4)-2))*f0*.12
        height=((position-tensor(self.center))*self.b[:,2]).sum(-1,keepdim=True)
        cavity=(.60+.40*torch.exp(height.clamp(max=0)/.025)) if 'pothole' in self.name else 1.
        shaded=srgb((base*lit*(.40+.60*ao)*cavity+spec).clamp(0,1))
        alpha=color[...,3:4]*(rast[...,3:4]>0)
        rgba=torch.cat([shaded*alpha,alpha],-1)
        rgba=dr.antialias(rgba,rast,clip[None],self.f)
        depth,_=dr.interpolate(z[None,:,None].contiguous(),rast,self.f)
        depth=torch.where(alpha>.01,depth,torch.full_like(depth,1e6))
        if ss>1:
            rgba=F.avg_pool2d(rgba.permute(0,3,1,2),ss).permute(0,2,3,1)
            depth=-F.max_pool2d(-depth.permute(0,3,1,2),ss).permute(0,2,3,1)
        return rgba[0],depth[0,...,0]

    def calibrate(self,ctx,camera,rgb):
        rgba,dep=self.draw(ctx,camera)
        alpha=rgba[...,3];outer=(alpha>.05)&(alpha<.75)
        # Fit to this fixed reference road exposure once, never frame by frame.
        if outer.sum()<10:outer=alpha>.05
        texture_alpha=self.color[0,...,3]
        texture_mask=(texture_alpha>.05)&(texture_alpha<.85) if 'pothole' in self.name else texture_alpha>.8
        source=self.color[0,...,:3][texture_mask].median(0).values
        target=rgb.permute(1,2,0)[outer].median(0).values
        weight=tensor([.2126,.7152,.0722]);sl=(linear(source)*weight).sum();tl=(linear(target)*weight).sum()
        self.gain=(tl/sl.clamp_min(.02)).clamp(.20,3.).repeat(3)
        self.tint=(linear(target)/tl.clamp_min(.02)).clamp(.8,1.2)
        self.neutral=.90 if 'pothole' in self.name else .65
        return {'source_srgb':source.tolist(),'target_srgb':target.tolist(),'linear_gain':self.gain.tolist(),'neutral_color_fraction':self.neutral,'target_tint':self.tint.tolist()}

@torch.no_grad()
def main():
    scene=create_scene();renderer=StreetGaussianRenderer();ctx=dr.RasterizeCudaContext()
    sid=Path(cfg.source_path).name;label='A' if sid=='056' else 'B'
    out=ROOT/'outputs'/f'inserted_{label}_{sid}';out.mkdir(exist_ok=True)
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
    ref=cameras[40];b,basis,fit=fit_road(Path(cfg.source_path),ref);origin=ref.camera_center.cpu().numpy()
    assets=[]
    local_fits=[]
    for name,distance,side in [('manhole_scan_010',10.8,.65),('pothole_scan_road2',8.8,-.65)]:
        center=origin+basis[:,1]*distance+basis[:,0]*side
        local_b,local_basis,local_fit=fit_road(Path(cfg.source_path),ref,center,1.1)
        center[2]=local_b[0]*center[0]+local_b[1]*center[1]+local_b[2]
        assets.append(Asset(name,center,local_basis));local_fits.append(local_fit)
    rgb=renderer.render(ref,scene.gaussians)['rgb'].clamp(0,1)
    calibration=[a.calibrate(ctx,ref,rgb) for a in assets]
    original_opacity=scene.gaussians.background._opacity.clone()
    xyz=scene.gaussians.background._xyz.detach().cpu().numpy();cut=np.zeros(len(xyz),bool)
    for a in assets:cut|=a.inside(xyz)
    scene.gaussians.background._opacity[tensor(cut).bool()]=-20
    torch.save({'source_checkpoint':str(ROOT/'outputs'/f'scene_{label}_{sid}'/'checkpoint_7000.pth'),'background_opacity':scene.gaussians.background._opacity.cpu(),'removed_background_indices':np.flatnonzero(cut)},out/'road_cutout.pth')
    manifest={'scene':sid,'road_plane_z_ax_by_c':b.tolist(),'road_fit':fit,'removed_gaussians':int(cut.sum()),'method':'fixed metric scan meshes + road Gaussian cutout + depth tested PBR rasterization','placements':[{'asset':a.name,'center_world':a.center.tolist(),'basis_columns':a.basis.tolist(),'calibration':c,'local_road_fit':local_fit} for a,c,local_fit in zip(assets,calibration,local_fits)],'frames':[]}
    writer=imageio.get_writer(str(out/'inserted.mp4'),fps=10,codec='libx264',quality=9,macro_block_size=2)
    compare=imageio.get_writer(str(out/'before_after.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2)
    frames=out/'frames';frames.mkdir(exist_ok=True)
    for camera in cameras:
        frame=camera.meta['frame']
        original=np.array(Image.open(ROOT/'outputs'/f'scene_{label}_{sid}'/'frames'/f'{frame:03d}.jpg')).astype('f4')/255
        result=renderer.render(camera,scene.gaussians);depth=result['depth'][0];acc=result['acc'][0]
        base=tensor(original);coverage=0
        for a in sorted(assets,key=lambda a:np.linalg.norm(a.center-camera.camera_center.cpu().numpy()),reverse=True):
            rgba,d=a.draw(ctx,camera)
            # Cutout depth cannot make the original road hide the pit. Nearby cars
            # and static foreground geometry still occlude the inserted surface.
            visible=(depth>=d-.12)|(acc<.4)
            alpha=rgba[...,3]*visible
            base=base*(1-alpha[...,None])+rgba[...,:3]*visible[...,None]
            coverage+=int((alpha>.1).sum())
        image=np.uint8((base.clamp(0,1).cpu().numpy()*255).round())
        writer.append_data(image);compare.append_data(np.concatenate([np.uint8(original*255),image],1))
        Image.fromarray(image).save(frames/f'{frame:03d}.jpg',quality=96)
        if frame in [35,40,43,46,49]:Image.fromarray(np.concatenate([np.uint8(original*255),image],1)).save(out/f'comparison_{frame:03d}.jpg',quality=97)
        manifest['frames'].append({'frame':frame,'visible_asset_pixels':coverage})
        if frame%20==0:print(sid,frame,coverage,flush=True)
    writer.close();compare.close()
    # A camera-only lateral sweep verifies that both meshes have one fixed world
    # pose and retain their occlusion/parallax away from the recorded trajectory.
    original_pose=ref.get_extrinsic().clone();cut_opacity=scene.gaussians.background._opacity.clone()
    parallax=imageio.get_writer(str(out/'parallax_0.6m.mp4'),fps=12,codec='libx264',quality=9,macro_block_size=2)
    for i,offset in enumerate(.6*np.sin(np.linspace(-np.pi/2,3*np.pi/2,48))):
        pose=original_pose.clone();pose[:3,3]+=original_pose[:3,0]*float(offset);ref.set_extrinsic(pose)
        scene.gaussians.background._opacity.copy_(original_opacity)
        base=renderer.render(ref,scene.gaussians)['rgb'].permute(1,2,0).clamp(0,1)
        scene.gaussians.background._opacity.copy_(cut_opacity)
        result=renderer.render(ref,scene.gaussians);depth=result['depth'][0];acc=result['acc'][0]
        for a in sorted(assets,key=lambda a:np.linalg.norm(a.center-ref.camera_center.cpu().numpy()),reverse=True):
            rgba,d=a.draw(ctx,ref);visible=(depth>=d-.12)|(acc<.4);alpha=rgba[...,3]*visible
            base=base*(1-alpha[...,None])+rgba[...,:3]*visible[...,None]
        parallax.append_data(np.uint8((base.clamp(0,1).cpu().numpy()*255).round()))
    parallax.close();ref.set_extrinsic(original_pose)
    manifest['additional_validation']='48-frame calibrated camera sweep, +/-0.6 m lateral, fixed asset poses'
    (out/'placement.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',sid,fit,int(cut.sum()),flush=True)

if __name__=='__main__':main()
