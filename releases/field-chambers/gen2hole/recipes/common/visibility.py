"""Visibility of the reconstructed triangle surface for SH appearance fitting."""
import numpy as np
import open3d as o3d

def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-10)

def ray_scene(a):
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(a['vertices'].astype('f4')),o3d.core.Tensor(a['faces'].astype('u4')))
    return scene

def sun_visibility(scene,points,normals,direction):
    d=np.broadcast_to(unit(np.asarray(direction,dtype='f4')),points.shape)
    rays=np.concatenate([points+normals*.0006,d],1).astype('f4')
    return (~scene.test_occlusions(o3d.core.Tensor(rays),tfar=3,nthreads=8).numpy()).astype('f4')

def ray_ambient(geometry, xyz, normal, tangent, bitangent, rng):
    import open3d as o3d
    rays_count = 12
    idx = np.arange(rays_count)
    sin_phi = np.sqrt((idx+.5)/rays_count)
    cos_phi = np.sqrt(1-sin_phi**2)
    base_theta = idx * np.pi*(3-np.sqrt(5))
    result = np.empty(len(xyz), "f4")
    for first in range(0, len(xyz), 16000):
        ids = slice(first, first+16000)
        xyzs = xyz[ids]
        theta = base_theta+rng.uniform(0, 2*np.pi, (len(xyzs), 1))
        d = (tangent[ids, None, :]*(np.cos(theta)*sin_phi)[..., None]
             + bitangent[ids, None, :]*(np.sin(theta)*sin_phi)[..., None]
             + normal[ids, None, :]*cos_phi[None, :, None])
        origin = np.broadcast_to((xyzs+normal[ids]*.00065)[:, None, :], d.shape)
        rays = o3d.core.Tensor(np.concatenate([origin, d], axis=-1).astype("f4"))
        result[ids] = 1 - geometry.test_occlusions(rays, tfar=1.35, nthreads=5).numpy().mean(axis=1)
    return result
