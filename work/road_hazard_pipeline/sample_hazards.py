"""Sample full 3D surfaces (including vertical walls and lid undersides), ray-bake AO."""
from pathlib import Path
import json,sys,time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import open3d as o3d

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/road_hazards';CACHE=ROOT/'work/road_hazard_samples';CACHE.mkdir(exist_ok=True)

def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-10)
def texture(path,uv):
    a=np.asarray(Image.open(path));a=a.astype('f4')/(65535 if a.dtype.itemsize==2 else 255)
    if a.ndim==2:a=a[...,None]
    u=uv[:,0]%1;v=uv[:,1]%1
    return np.stack([map_coordinates(a[...,k],[(1-v)*(a.shape[0]-1),u*(a.shape[1]-1)],order=1,mode='wrap') for k in range(a.shape[-1])],1)

def ray_scene(a):
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(a['vertices'].astype('f4')),o3d.core.Tensor(a['faces'].astype('u4')))
    return scene

def sun_visibility(scene,points,normals,direction):
    d=np.broadcast_to(unit(np.asarray(direction,dtype='f4')),points.shape)
    rays=np.concatenate([points+normals*.0006,d],1).astype('f4')
    return (~scene.test_occlusions(o3d.core.Tensor(rays),tfar=3,nthreads=8).numpy()).astype('f4')

def build(name):
    start=time.time();d=OUT/name;a=np.load(d/'mesh.npz');v=a['vertices'];f=a['faces'];tri=v[f];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];cross=np.cross(e1,e2);areas=np.linalg.norm(cross,axis=-1)*.5;normals=unit(cross)
    cdf=np.cumsum(areas,dtype='f8');area=float(cdf[-1]);count=max(150000,min(1100000,int(area*165000)))
    rng=np.random.default_rng(1781);ids=np.minimum(np.searchsorted(cdf,rng.random(count)*area),len(f)-1)
    u=np.sqrt(rng.random(count));v2=rng.random(count);bary=np.stack([1-u,u*(1-v2),u*v2],1).astype('f4')
    points=(tri[ids]*bary[...,None]).sum(1);N=normals[ids];uv=(a['uv'][f[ids]]*bary[...,None]).sum(1)
    du1=a['uv'][f[:,1]]-a['uv'][f[:,0]];du2=a['uv'][f[:,2]]-a['uv'][f[:,0]];det=du1[:,0]*du2[:,1]-du1[:,1]*du2[:,0]
    T=(e1*du2[:,1,None]-e2*du1[:,1,None])/np.where(abs(det[:,None])>1e-9,det[:,None],1)
    T=unit(T[ids]-N*np.sum(T[ids]*N,1,keepdims=True));bad=np.linalg.norm(T,axis=1)<.5
    if bad.any():
        axis=np.tile([0,0,1.],(bad.sum(),1));axis[abs(N[bad,2])>.9]=[0,1,0]
        T[bad]=unit(np.cross(axis,N[bad]))
    B=np.cross(N,T)
    color=np.ones((count,4),dtype='f4');normal=np.zeros((count,3),dtype='f4');arm=np.ones((count,3),dtype='f4');road_mask=np.zeros(count,dtype='f4')
    material=a['face_material'][ids];materials=json.loads((d/'materials.json').read_text())
    for index,m in enumerate(materials):
        take=material==index
        if not take.any():continue
        color[take,:3]=texture(m['color'],uv[take])[:,:3]
        ns=texture(m['normal'],uv[take])[:,:3]*2-1
        ns=unit(T[take]*ns[:,:1]+B[take]*ns[:,1:2]+N[take]*ns[:,2:3]);normal[take]=(ns+1)/2
        arm[take,1]=np.clip(texture(m['roughness'],uv[take])[:,0],.4,1);arm[take,2]=m['metallic']
        road_mask[take]=m['name'] in ['asphalt','concrete','fracture_asphalt','fracture_concrete']
    surface=np.load(d/'surface.npz');half=float(surface['width'])/2
    alpha=np.clip(np.minimum(half-abs(points[:,0]),half-abs(points[:,1]))/.075,0,1).astype('f4')
    scene=ray_scene(a);S=24;ao=np.empty(count,dtype='f4')
    # Deterministic cosine-weighted hemisphere samples; rotation decorrelates points.
    k=np.arange(S);r=np.sqrt((k+.5)/S);z=np.sqrt(1-r*r);phi=k*np.pi*(3-np.sqrt(5))
    for first in range(0,count,12000):
        last=min(count,first+12000);n=N[first:last];t=T[first:last];b=B[first:last]
        turn=rng.uniform(0,2*np.pi,(last-first,1));angles=phi[None]+turn
        directions=t[:,None]*((r[None]*np.cos(angles))[...,None])+b[:,None]*((r[None]*np.sin(angles))[...,None])+n[:,None]*z[None,:,None]
        origins=np.broadcast_to((points[first:last]+n*.0006)[:,None],directions.shape)
        rays=np.concatenate([origins,directions],-1).astype('f4')
        occluded=scene.test_occlusions(o3d.core.Tensor(rays),tfar=2.0,nthreads=8).numpy();ao[first:last]=1-occluded.mean(1)
    sun=sun_visibility(scene,points,N,[-.35,-.25,1])
    spacing=np.sqrt(area/count);scales=np.tile([spacing*.82,spacing*.82,.00035],(count,1)).astype('f4')
    matrices=np.stack([T,B,N],-1).astype('f4')
    np.savez_compressed(CACHE/(name+'.npz'),local_xyz=points.astype('f4'),normals=N.astype('f4'),rotation_matrices=matrices,scales=scales,alpha=alpha,uv=uv.astype('f4'),color=color,normal=normal,arm=arm,ao=ao[:,None],sun_visibility=sun,road_mask=road_mask)
    (d/'sampling.json').write_text(json.dumps({'surface_area_m2':area,'gaussians':count,'sampling_spacing_m':spacing,'ambient_occlusion_rays_per_point':S,'normal_sigma_m':.00035,'geometry':'full 3D mesh with shaft walls and lid undersides','seconds':time.time()-start},indent=2));print('SAMPLED',name,count,time.time()-start,flush=True)

def load_surface(name,basis=None):
    s=dict(np.load(CACHE/(name+'.npz')));s['name']=name;s['geometric_ao']=True
    if basis is not None and (OUT/name/'mesh.npz').is_file():
        a=np.load(OUT/name/'mesh.npz');scene=ray_scene(a);direction=np.array([-.35,-.25,1.])@np.asarray(basis)
        s['sun_visibility']=sun_visibility(scene,s['local_xyz'],s['normals'],direction)
    return s

if __name__=='__main__':
    names=sys.argv[1:] or [c['id'] for c in json.loads((OUT/'catalog.json').read_text())]
    for name in names:
        if not (CACHE/(name+'.npz')).exists():build(name)
