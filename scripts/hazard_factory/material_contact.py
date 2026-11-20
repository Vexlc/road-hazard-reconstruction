"""Preserve metal parts and transfer intrinsic geometric contact occlusion to host RGB."""
import hashlib
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from run import atomic_json,sha


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def original_blend(vertices,model):
    local=vertices[:,:3];r=np.linalg.norm(local[:,:2],axis=1);z=local[:,2]
    return (smooth((r-max(model['width_m'],model['length_m'])*.69)/.21)
            *smooth((z+.021)/.016)*smooth((.022-z)/.012))


def intrinsic(model,vertices,job):
    mesh_file=Path(job['mesh_cache'])/(model['id']+'.npz')
    cache=Path(job['work'])/'material_contact';cache.mkdir(exist_ok=True)
    key=hashlib.sha256((sha(mesh_file)+sha(Path(model['ply']))+sha(Path(__file__))).encode()).hexdigest()
    path=cache/(model['id']+'.npz');stamp=path.with_suffix('.json')
    if path.exists() and stamp.exists() and __import__('json').loads(stamp.read_text())['key']==key:
        with np.load(path) as data:return dict(data)
    with np.load(mesh_file) as mesh:
        scene=o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(mesh['xyz'].astype('f4')),
                            o3d.core.Tensor(mesh['faces'].astype('u4')))
        primitive=[];errors=[]
        for first in range(0,len(vertices),60000):
            closest=scene.compute_closest_points(o3d.core.Tensor(vertices[first:first+60000,:3].astype('f4')),nthreads=4)
            primitive.append(closest['primitive_ids'].numpy())
            errors.append(np.linalg.norm(closest['points'].numpy()-vertices[first:first+60000,:3],axis=1))
        primitive=np.concatenate(primitive)
        part=mesh['part_names'][mesh['face_part'][primitive]]
        # Float32 closest-point queries on the fine embossed lid can slide a few
        # tens of microns along a skinny triangle; keep a 0.1 mm geometric bound.
        if np.max(np.concatenate(errors))>1e-4:raise ValueError('canonical mesh does not match Gaussian geometry')
        road=part=='Road around the aperture'
        occluders=o3d.t.geometry.RaycastingScene(nthreads=4)
        nonroad=mesh['part_names'][mesh['face_part']]!='Road around the aperture'
        occluders.add_triangles(o3d.core.Tensor(mesh['xyz'].astype('f4')),
                              o3d.core.Tensor(mesh['faces'][nonroad].astype('u4')))
    matrices=Rotation.from_quat(vertices[:,[20,21,22,19]]).as_matrix()
    normals=matrices[:,:,2];tangent=matrices[:,:,0];bitangent=matrices[:,:,1]
    road&=normals[:,2]>.5
    mix=original_blend(vertices,model)*road
    ids=np.flatnonzero(mix>.025)
    rng=np.random.default_rng(model['seed']+57737)
    ao=np.ones(len(vertices),dtype='f4');count=32
    k=np.arange(count);radius=np.sqrt((k+.5)/count);up=np.sqrt(1-radius**2)
    phi=k*np.pi*(3-np.sqrt(5))
    for first in range(0,len(ids),10000):
        index=ids[first:first+10000]
        angle=phi+rng.uniform(0,2*np.pi,(len(index),1))
        direction=(tangent[index,None,:]*(radius*np.cos(angle))[...,None]
                 +bitangent[index,None,:]*(radius*np.sin(angle))[...,None]
                 +normals[index,None,:]*up[None,:,None])
        origin=np.broadcast_to((vertices[index,:3]+normals[index]*.0008)[:,None,:],direction.shape)
        rays=o3d.core.Tensor(np.concatenate([origin,direction],axis=-1).astype('f4'))
        ao[index]=1-occluders.test_occlusions(rays,tfar=.65,nthreads=4).numpy().mean(axis=1)
    result={'road':road,'ambient_visibility':ao}
    np.savez_compressed(path,**result)
    atomic_json(stamp,{'key':key,'method':'closest canonical mesh part; 32 cosine hemisphere rays; 0.65m range',
                       'mesh_sha256':sha(mesh_file),'road_points':int(road.sum())})
    return result


@torch.no_grad()
def apply(anchor,model,asset,exposure,job,rb):
    vertices,_=rb.read_asset(model)
    fields=intrinsic(model,vertices,job)
    old_mix=original_blend(vertices,model);mix=old_mix*fields['road']
    dc,rest=rb.split_gaussian_coefficients(vertices,anchor.basis)
    corrected=rb.srgb((rb.linear((dc*rb.C0+.5).clamp(0,1))*exposure).clamp(0,1))
    host=rb.tensor(anchor.color(vertices[:,:3]))
    visibility=fields['ambient_visibility']
    strength=float(job.get('contact_occlusion_strength',.60))
    if not 0<=strength<=1:raise ValueError('contact occlusion strength must be between 0 and 1')
    factor=1-strength*(1-visibility)
    shaded=rb.srgb((rb.linear(host)*rb.tensor(factor[:,None])).clamp(0,1))
    w=rb.tensor(mix[:,None])
    asset['feature_dc']=((torch.lerp(corrected,shaded,w)-.5)/rb.C0)[:,None].contiguous()
    asset['feature_rest']=(rest*(1-w[:,None])*np.clip(exposure**.32,.8,1.28)).contiguous()
    affected=(mix>.1)&(visibility<.95)
    info={'road_mesh_labeled_points':int(fields['road'].sum()),
          'restored_nonroad_material_points':int(((old_mix>.025)&~fields['road']).sum()),
          'host_road_contact_shaded_points':int(affected.sum()),
          'minimum_ambient_visibility':float(visibility[mix>.1].min()) if (mix>.1).any() else 1.,
          'ambient_occlusion_strength':strength,
          'method':'canonical mesh parts preserve metal; intrinsic 3D hemisphere visibility darkens only transferred host-road color'}
    return asset,info
