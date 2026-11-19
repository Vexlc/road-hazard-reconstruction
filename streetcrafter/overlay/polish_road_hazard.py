"""Bake the host road appearance into Gaussian road surfaces and render at 1080p.

The final pixels always come from the joint Gaussian model. Camera projections
are used offline to transfer host-road appearance, not as output image overlays.
"""
from pathlib import Path
import os,sys,json,time
import torch,numpy as np
import torch.nn.functional as F
from PIL import Image
import imageio.v2 as imageio
from create_scene import create_scene
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from export_reconstruction import export_splat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/asset_pipeline'));sys.path.insert(0,str(ROOT/'work/road_hazard_pipeline'))
from gaussian_assets import tensor,unit,linear,srgb,C0,save_ply
from sample_hazards import load_surface
ATTRS={'xyz':'_xyz','feature_dc':'_features_dc','feature_rest':'_features_rest','scaling':'_scaling','rotation':'_rotation','opacity':'_opacity','semantic':'_semantic'}

def set_background(g,state):
    for key,attr in ATTRS.items():setattr(g,attr,state[key])

@torch.no_grad()
def bake(scene,renderer,cameras,positions):
    n=len(positions);best=torch.full((n,),-1.,device='cuda');colors=torch.zeros(n,3,device='cuda');H,W=cameras[0].image_height,cameras[0].image_width
    for frame in [20,24,28,32,35,38,40,42,44]:
        c=cameras[frame];r=renderer.render(c,scene.gaussians);w=c.get_extrinsic(world2cam=True);q=positions@w[:3,:3].T+w[:3,3];z=q[:,2]
        u=q[:,0]*c.K[0,0]/z+c.K[0,2];v=q[:,1]*c.K[1,1]/z+c.K[1,2];grid=torch.stack([2*(u+.5)/W-1,2*(v+.5)/H-1],1)[None,None]
        sample=F.grid_sample(r['rgb'][None],grid,align_corners=False)[0,:,0].T;depth=F.grid_sample(r['depth'][:,None],grid,align_corners=False)[0,0,0]
        valid=(z>.1)&(u>1)&(u<W-2)&(v>1)&(v<H-2)&((depth-z).abs()<.65)
        score=torch.where(valid,1/z,torch.full_like(z,-1));take=score>best;colors[take]=sample[take];best[take]=score[take]
    bad=best<0
    if bad.any():colors[bad]=colors[~bad].median(0).values
    return colors,float(bad.float().mean())

@torch.no_grad()
def main():
    started=time.time();scene=create_scene();renderer=StreetGaussianRenderer();out=Path(cfg.model_path);summary=json.loads((out/'summary.json').read_text());sid=summary['scene'];lab='A' if sid=='056' else 'B';asset=summary['assets'][0];name=asset['name'];placement=asset['placement']
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
    # Increase actual rasterization resolution rather than upscaling an image.
    for camera in cameras:
        scale=1920/camera.image_width;K=camera.K.clone();K[:2]*=scale;camera.image_width=1920;camera.image_height=1080;camera.set_intrinsic(K)
    g=scene.gaussians.background;merged={k:getattr(g,a) for k,a in ATTRS.items()}
    original=torch.load(ROOT/'outputs'/f'scene_{lab}_{sid}'/'checkpoint_7000.pth',map_location='cuda',weights_only=False)['background']
    source={k:original[k] for k in ATTRS};start=summary['original_background_gaussians_retained'];count=asset['gaussians']
    s=load_surface(name,np.asarray(placement['basis_columns']))
    photographed = bool(s.get('photo_appearance', False))
    if photographed:
        take = (s['road_mask'] > .01) & (abs(s['local_xyz'][:,2]) < .025)
    else:
        take = (s['road_mask'] > .5) & (abs(s['local_xyz'][:,2]) < .015) & (s['normals'][:,2] > .90)
    local_ids=np.flatnonzero(take);indices=tensor(local_ids+start).long();points=merged['xyz'][indices]
    set_background(g,source);color,missing=bake(scene,renderer,cameras,points);set_background(g,merged)
    b=tensor(placement['basis_columns']);N=unit(.7*tensor(s['normals'][take])@b.T+.3*(tensor(s['normal'][take])*2-1)@b.T)
    light=unit(tensor([-.35,-.25,1.]));flat=(b[:,2]*light).sum().clamp_min(.1);ndl=(N*light).sum(-1,keepdim=True).clamp_min(0)
    ao=tensor(s['ao'][take]);sun=tensor(s['sun_visibility'][take,None]);shade=(.74*(.035+.965*ao)+.26*ndl*sun)/(.74+.26*flat)
    # Keep very subtle scan grain, but transfer the host's larger colour/texture field.
    raw=linear(tensor(s['color'][take,:3]));lum=(raw*tensor([.2126,.7152,.0722])).sum(-1,keepdim=True);grain=(lum/lum.median().clamp_min(.01)).clamp(.7,1.3)
    baked=srgb((linear(color.clamp(0,1))*shade*(.95+.05*grain)).clamp(0,1)).clamp(0,1)
    if photographed:
        weight = tensor(s['road_mask'][take,None])
        merged['feature_dc'][indices,0] = (baked-.5)/C0*weight + merged['feature_dc'][indices,0]*(1-weight)
        merged['feature_rest'][indices] *= 1-weight[:,None]
    else:
        merged['feature_dc'][indices,0] = (baked-.5)/C0
        merged['feature_rest'][indices] = 0
    if name.startswith('stereo_pothole_'):
        pit = (s['road_mask'] < .8) & (s['local_xyz'][:,2] < -.002)
        pit_ids = tensor(np.flatnonzero(pit)+start).long()
        depth = tensor(-s['local_xyz'][pit,2,None])
        influence = tensor(1.-s['road_mask'][pit,None])
        facing = merged['feature_dc'][pit_ids,0]*C0+.5
        # The urban site is in open shade. Reduce the captured warm cast and
        # make the deepest part read as a recess without flattening its texture.
        luminance = facing @ tensor([.2126,.7152,.0722])
        site_tint = torch.tensor([.91,1.,1.12], device='cuda')
        facing = torch.lerp(facing,luminance[:,None]*site_tint,.47*influence)
        shade = 1.-.21*(1-torch.exp(-depth/.012))*influence
        merged['feature_dc'][pit_ids,0] = (facing*shade-.5)/C0
    summary['road_appearance_transfer']={'gaussians':len(indices),'method':'multi-view host-road texture projection baked into Gaussian SH0, retaining geometric AO and lid shadows','unobserved_fraction':missing,'reference_frames':[20,24,28,32,35,38,40,42,44]}
    del s
    writer=imageio.get_writer(str(out/'pure_3dgs.mp4'),fps=10,codec='libx264',quality=9,macro_block_size=2)
    compare=imageio.get_writer(str(out/'before_after.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2)
    frames=out/'frames';frames.mkdir(exist_ok=True)
    for c in cameras:
        set_background(g,source);before=renderer.render(c,scene.gaussians)['rgb'].clamp(0,1)
        set_background(g,merged);rgb=renderer.render(c,scene.gaussians)['rgb'].clamp(0,1)
        image=np.uint8(rgb.permute(1,2,0).cpu().numpy()*255+.5);old=np.uint8(before.permute(1,2,0).cpu().numpy()*255+.5);frame=c.meta['frame']
        writer.append_data(image);compare.append_data(np.concatenate([old,image],1));Image.fromarray(image).save(frames/f'{frame:03d}.jpg',quality=96)
        if frame in [35,40,42,43]:Image.fromarray(np.concatenate([old,image],1)).save(out/f'comparison_{frame:03d}.jpg',quality=96)
    writer.close();compare.close()
    export_splat(scene.gaussians,cameras[40],out/'scene_with_assets_frame040.ply')
    state={k:v[start:start+count] for k,v in merged.items()};save_ply(state,out/f'{name}_world.ply')
    checkpoint=scene.gaussians.save_state_dict(is_final=True);checkpoint['iter']=7001;torch.save(checkpoint,out/'trained_model/iteration_7000.pth')
    ref=renderer.render(cameras[40],scene.gaussians)['rgb'].clone();scene.gaussians.load_state_dict(torch.load(out/'trained_model/iteration_7000.pth',weights_only=False));again=renderer.render(cameras[40],scene.gaussians)['rgb']
    error=float((ref-again).abs().max());assert error<1e-6;summary['validation']['checkpoint_reload_max_rgb_error']=error
    c=cameras[40];pose=c.get_extrinsic().clone();sweep=imageio.get_writer(str(out/'parallax_0.6m.mp4'),fps=12,codec='libx264',quality=9,macro_block_size=2)
    for offset in .6*np.sin(np.linspace(-np.pi/2,3*np.pi/2,48)):
        newpose=pose.clone();newpose[:3,3]+=pose[:3,0]*float(offset);c.set_extrinsic(newpose)
        rgb=renderer.render(c,scene.gaussians)['rgb'].clamp(0,1);sweep.append_data(np.uint8(rgb.permute(1,2,0).cpu().numpy()*255+.5))
    sweep.close();c.set_extrinsic(pose);summary['render_resolution']=[1920,1080];summary['polish_seconds']=time.time()-started
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    config=out/'render_config.yaml';lines=[line for line in config.read_text().splitlines() if not line.startswith('resolution:')];config.write_text('\n'.join(lines+['resolution: 1920'])+'\n')
    print('POLISHED',name,len(indices),time.time()-started,flush=True)

if __name__=='__main__':main()
