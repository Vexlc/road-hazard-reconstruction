"""Resample road around the inserted assets so broad parent splats cannot fill pits."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from gaussian_assets import ROOT,tensor,C0

def surface_path(name):
    p=ROOT/'outputs/road_assets'/name/'surface.npz'
    return p if p.exists() else ROOT/'outputs/road_hazards'/name/'surface.npz'

def placement_surface(placement):
    path = placement.get('height_field')
    return np.load(path if path is not None else surface_path(placement['asset']))

def sample_field(surface,key,local):
    a=surface[key];h,w=a.shape
    row=(.5-local[:,1]/float(surface['length']))*(h-1);col=(local[:,0]/float(surface['width'])+.5)*(w-1)
    return ndimage.map_coordinates(a,[row,col],order=1,mode='constant',cval=0)

def road_cut_mask(g,placements):
    world=g._xyz.detach().cpu().numpy();sigma=g._scaling.detach().exp().cpu().numpy().max(1)
    cut=np.zeros(len(world),bool)
    for p in placements:
        s=placement_surface(p);b=np.array(p['basis_columns']);c=np.array(p['center_world']);local=(world-c)@b
        mask=s['alpha']>.03;edge=mask&~ndimage.binary_erosion(mask);r,x=np.where(edge)
        boundary=np.c_[(x/(mask.shape[1]-1)-.5)*float(s['width']),(.5-r/(mask.shape[0]-1))*float(s['length'])]
        distance=cKDTree(boundary).query(local[:,:2],workers=4)[0]
        distance[sample_field(s,'alpha',local)>.03]=0
        cut|=(distance<np.minimum(3.5*sigma+.04,2.5))&(abs(local[:,2])<.4)
    return cut

@torch.no_grad()
def make_road_apron(scene,renderer,cameras,placements):
    # A fine surface follows the two local road fits. A backing below each asset
    # follows its relief, so even residual transparency cannot reveal flat road.
    basis=np.array(placements[0]['basis_columns']);origin=np.array(placements[0]['center_world'])
    centers=np.stack([np.array(p['center_world']) for p in placements]);cc=(centers-origin)@basis
    captures = len(placements) == 1 and (placements[0].get('close_road_resampling', False)
        or placements[0]['asset'].startswith(('stereo_pothole_', 'photo_manhole_')))
    lo=cc[:,:2].min(0)-3.4;hi=cc[:,:2].max(0)+3.4;step=.016
    xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+step,step),np.arange(lo[1],hi[1]+step,step))
    flat=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    spacing=np.full(len(flat),step)
    if captures:
        # Replace the local coarse grid with a 4 mm grid, so close views do not
        # reveal the footprint seam or a sparse splat pattern.
        keep=np.max(np.abs(flat[:,:2]-cc[0,:2]),axis=1)>1.30
        fine_step=.004
        u=np.arange(-1.31,1.31+fine_step,fine_step)
        fx,fy=np.meshgrid(u+cc[0,0],u+cc[0,1])
        fine=np.c_[fx.ravel(),fy.ravel(),np.zeros(fx.size)]
        flat=np.concatenate([flat[keep],fine])
        spacing=np.r_[spacing[keep],np.full(len(fine),fine_step)]
    xyz=flat@basis.T+origin
    # Blend local ground planes over distance, rather than assuming a level road.
    estimates=[];weights=[]
    for p in placements:
        c=np.array(p['center_world']);n=np.array(p['basis_columns'])[:,2]
        estimates.append(c[2]-(n[0]*(xyz[:,0]-c[0])+n[1]*(xyz[:,1]-c[1]))/n[2])
        weights.append(1/(np.sum((xyz[:,:2]-c[:2])**2,1)+.2))
    weights=np.stack(weights);estimates=np.stack(estimates);xyz[:,2]=(weights*estimates).sum(0)/weights.sum(0)
    asset_alpha=np.zeros(len(xyz))
    for p in placements:
        c=np.array(p['center_world']);b=np.array(p['basis_columns']);s=placement_surface(p);local=(xyz-c)@b
        a=sample_field(s,'alpha',local);height=sample_field(s,'height',local)
        if captures:
            coverage=np.clip((a-.25)/.63,0,1)
            coverage=coverage**2*(3-2*coverage)
            backing=np.where(a>.005,height-.0015*a,0)
        else:
            coverage=a
            backing=np.where(a>.005,height-.025,0)
        asset_alpha=np.maximum(asset_alpha,coverage)
        xyz+=backing[:,None]*b[:,2]
    # Fade the apron well outside the removed parents, keeping the original scene
    # beyond this compact road area intact.
    nearest=np.min(np.linalg.norm(flat[:,:2,None]-cc[:,:2].T[None,:,:],axis=1),axis=1)
    alpha=np.clip((3.35-nearest)/.7,0,1)*.98*(1-asset_alpha)
    inside=alpha>.005;xyz=xyz[inside];alpha=alpha[inside];spacing=spacing[inside]
    positions=tensor(xyz);best=torch.full((len(xyz),),-1.,device='cuda');colors=torch.zeros(len(xyz),3,device='cuda')
    frames=[20,24,28,32,35,38,40,42,44]
    H,W=cameras[0].image_height,cameras[0].image_width
    for frame in frames:
        cam=cameras[frame];result=renderer.render(cam,scene.gaussians);rgb=result['rgb'];depth=result['depth'][0]
        w=cam.get_extrinsic(world2cam=True);q=positions@w[:3,:3].T+w[:3,3];z=q[:,2]
        u=q[:,0]*cam.K[0,0]/z+cam.K[0,2];v=q[:,1]*cam.K[1,1]/z+cam.K[1,2]
        grid=torch.stack([2*(u+.5)/W-1,2*(v+.5)/H-1],1)[None,None]
        sampled=F.grid_sample(rgb[None],grid,align_corners=False)[0,:,0].T
        observed=F.grid_sample(depth[None,None],grid,align_corners=False)[0,0,0]
        valid=(z>.2)&(u>1)&(u<W-2)&(v>1)&(v<H-2)&((observed-z).abs()<.6)
        score=torch.where(valid,1/z,torch.full_like(z,-1));take=score>best
        colors[take]=sampled[take];best[take]=score[take]
    missing=best<0
    if missing.any():
        good=np.flatnonzero((~missing).cpu().numpy());bad=np.flatnonzero(missing.cpu().numpy())
        nearest_good=cKDTree(xyz[good,:2]).query(xyz[bad,:2],workers=4)[1];colors[missing]=colors[tensor(good[nearest_good]).long()]
    if captures:
        texture=ROOT/'outputs/road_hazards/materials/asphalt/color.png'
        if texture.exists():
            gray=np.asarray(Image.open(texture).convert('L').resize((512,512)),dtype=np.float32)/255
            grain=gray-ndimage.gaussian_filter(gray,6,mode='wrap')
            local=flat[inside,:2]-cc[0,:2]
            pixel=(local*512/2.2)%512
            detail=ndimage.map_coordinates(grain,[pixel[:,1],pixel[:,0]],order=1,mode='wrap')
            feather=np.clip((1.29-np.max(np.abs(local),axis=1))/.47,0,1)
            gain=(1+1.3*np.clip(detail,-.13,.13)*feather)
            colors=(colors*tensor(gain[:,None])).clamp(0,1)
    # Fine splats remain thin along the local normal. Outside the small cutout,
    # the broad old road splats still supply any distant missing coverage.
    q=Rotation.from_matrix(basis).as_quat()[[3,0,1,2]];N=len(xyz)
    opacity=tensor(alpha[:,None]);sh=(colors-.5)/C0
    state={'xyz':positions,'feature_dc':sh[:,None].contiguous(),'feature_rest':torch.zeros(N,3,3,device='cuda'),'rotation':tensor(np.tile(q,(N,1))),'scaling':tensor(np.log(np.c_[spacing*.72,spacing*.72,np.full(N,.0004)])),'opacity':torch.logit(opacity.clamp(1e-5,1-1e-5)),'semantic':torch.empty(N,0,device='cuda')}
    info={'gaussians':N,'spacing_m':step,'teacher_frames':frames,'missing_teacher_fraction':float(missing.float().mean()),'source':'original scene Gaussian renders projected onto LiDAR fitted road; baked to Gaussian SH0','asset_backing_offset_m':.0015 if captures else .025}
    if captures:
        info['close_road_spacing_m']=fine_step
    return state,info
