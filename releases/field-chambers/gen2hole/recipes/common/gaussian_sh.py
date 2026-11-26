"""Degree-1 SH fit and PLY layout from the original road asset exporter."""
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement

C0 = .28209479177387814
C1 = .4886025119029199

def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device='cuda').contiguous()

def unit(a):return a/torch.linalg.norm(a,dim=-1,keepdim=True).clamp_min(1e-8)

def linear(c):return torch.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)

def srgb(c):return torch.where(c<=.0031308,c*12.92,1.055*c.clamp_min(0)**(1/2.4)-.055)

def sh_basis(d):return torch.stack([torch.full_like(d[...,0],C0),-C1*d[...,1],C1*d[...,2],-C1*d[...,0]],-1)

@torch.no_grad()
def make_gaussians(s,center=None,basis=None,calibration=None):
    center=np.zeros(3) if center is None else np.asarray(center)
    basis=np.eye(3) if basis is None else np.asarray(basis)
    cal=calibration or {'linear_gain':[1,1,1],'target_tint':[1,1,1],'neutral_color_fraction':0}
    xyz=tensor(s['local_xyz']@basis.T+center)
    matrices=basis[None]@s['rotation_matrices'];q=Rotation.from_matrix(matrices).as_quat()[:,[3,0,1,2]]
    count=len(xyz);coeff=torch.empty((count,4,3),device='cuda')
    # Full-sphere fitting avoids unstable SH extrapolation below the road.
    indices=np.arange(48);z=1-2*(indices+.5)/48;phi=indices*np.pi*(3-np.sqrt(5));r=np.sqrt(1-z*z)
    directions=tensor(np.stack([r*np.cos(phi),r*np.sin(phi),z],1));D=sh_basis(directions);inverse=torch.linalg.pinv(D)
    light=unit(tensor([-.35,-.25,1.]));flat=(tensor(basis[:,2])*light).sum().clamp_min(.1)
    residual=[]
    for start in range(0,count,16384):
        end=min(count,start+16384);ns=tensor(s['normal'][start:end,:3])*2-1
        N=unit(.70*tensor(s['normals'][start:end]@basis.T)+.30*unit(ns@tensor(basis.T)))
        raw=linear(tensor(s['color'][start:end,:3]));lum=(raw*tensor([.2126,.7152,.0722])).sum(-1,keepdim=True)
        neutral=cal['neutral_color_fraction']
        if 'road_mask' in s:neutral=neutral*tensor(s['road_mask'][start:end,None])
        base=((1-neutral)*raw+neutral*lum*tensor(cal['target_tint']))*tensor(cal['linear_gain'])
        arm=tensor(s['arm'][start:end]);rough=arm[:,1:2].clamp(.25,1);metal=arm[:,2:3]
        ao=tensor(s['ao'][start:end,:1]);ndl=(N*light).sum(-1,keepdim=True).clamp_min(0)
        lit=(.74+.26*ndl)/(.74+.26*flat)
        ao_factor=.40+.60*ao
        if s.get('geometric_ao',False):
            visibility=tensor(s['sun_visibility'][start:end,None]);lit=(.74*(.035+.965*ao)+.26*ndl*visibility)/(.74+.26*flat);ao_factor=1.
        f0=.04*(1-metal)+base*metal
        cavity=.60+.40*torch.exp(tensor(s['local_xyz'][start:end,2:3]).clamp(max=0)/.025) if 'pothole' in s['name'] and not s.get('geometric_ao',False) else 1.
        half=unit(-directions[:,None,:]+light)
        nh=(N[None]*half).sum(-1,keepdim=True).clamp(0,1)
        spec=nh**(2/(rough[None]**4)-2)*f0[None]*.12
        if s.get('geometric_ao',False):spec=spec*visibility[None]
        rgb=srgb((base[None]*lit[None]*(ao_factor[None] if torch.is_tensor(ao_factor) else ao_factor)*cavity+spec).clamp(0,1))
        sh=torch.einsum('kv,vnc->nkc',inverse,rgb-.5);coeff[start:end]=sh
        predicted=torch.einsum('vk,nkc->vnc',D,sh)+.5
        residual.append(float((predicted-rgb).abs().mean()))
    opacity=tensor(s['alpha'][:,None])*.98
    return {'xyz':xyz,'feature_dc':coeff[:,:1].contiguous(),'feature_rest':coeff[:,1:].contiguous(),'scaling':tensor(np.log(s['scales'])),'rotation':tensor(q),'opacity':torch.logit(opacity.clamp(1e-5,1-1e-5)),'semantic':torch.empty(count,0,device='cuda')},float(np.mean(residual))

def save_ply(state,path):
    xyz=state['xyz'].detach().cpu().numpy();sh=torch.cat([state['feature_dc'],state['feature_rest']],1).detach().cpu().numpy()
    arrays=[xyz,np.zeros_like(xyz),sh[:,0],sh[:,1:].transpose(0,2,1).reshape(len(xyz),-1)]
    arrays +=[state[k].detach().cpu().numpy() for k in ['opacity','scaling','rotation']]
    names=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(sh[:,1:].size//len(xyz))]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    values=np.concatenate(arrays,1).astype('f4');vertices=np.empty(len(xyz),dtype=[(k,'f4') for k in names])
    for i,k in enumerate(names):vertices[k]=values[:,i]
    Path(path).parent.mkdir(exist_ok=True,parents=True);PlyData([PlyElement.describe(vertices,'vertex')]).write(path)
