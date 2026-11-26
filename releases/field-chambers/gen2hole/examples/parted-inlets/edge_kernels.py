"""Keep splats parallel to a cover edge while limiting their footprint across the opening."""
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from road_geometry import glb_triangles,part_id


def boundaries(v,f):
    pts,idx=np.unique(v,axis=0,return_inverse=True)
    triangles=idx[f]
    edges=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    pairs,count=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    ends=pts[pairs[count==1]]
    length=np.linalg.norm(ends[:,1]-ends[:,0],axis=1)
    ends=ends[length>.00012]
    length=length[length>.00012]
    if not len(ends):return None
    # Midpoint queries are conservative; recheck candidate segments at exact distance.
    n=np.maximum(1,np.ceil(length/.003).astype(int))
    index=np.repeat(np.arange(len(ends)),n)
    start=np.repeat(np.cumsum(n)-n,n)
    s=(np.arange(len(index))-start+.5)/n[index]
    samples=ends[index,0]+(ends[index,1]-ends[index,0])*s[:,None]
    return ends,index,cKDTree(samples)


def align_edges(dest, part_codes=(147,148,149,150,151,152,11,14), *, falloff=2.0, minimum_ratio=.24):
    path=Path(dest)/'asset.ply'
    data=PlyData.read(path,mmap=False)
    p=data['vertex'].data
    with np.load(Path(dest)/'splat_material.npz') as side:
        parts=side['part'];names=side['part_names'].tolist()
    xyz=np.c_[p['x'],p['y'],p['z']]
    changed=0
    for name,v,f in glb_triangles(Path(dest)/'asset.glb',names=True):
        if part_id(name) not in part_codes:continue
        band=boundaries(v,f)
        if band is None:continue
        ends,index,tree=band
        ids=np.flatnonzero(parts==names.index(name))
        if not len(ids):continue
        sigma=np.exp(p['scale_0'][ids])
        near,nearest=tree.query(xyz[ids],workers=2,distance_upper_bound=float(sigma.max()*2.6+.002))
        keep=np.isfinite(near)
        ids,sigma,nearest=ids[keep],sigma[keep],nearest[keep]
        a,b=ends[index[nearest]].transpose(1,0,2)
        vector=b-a
        rel=xyz[ids]-a
        weight=np.clip((rel*vector).sum(1)/(vector*vector).sum(1),0,1)
        distance=np.linalg.norm(rel-vector*weight[:,None],axis=1)
        new_sigma=np.clip(distance/falloff,minimum_ratio*sigma,sigma)
        keep=new_sigma<.99*sigma
        ids,new_sigma,vector=ids[keep],new_sigma[keep],vector[keep]
        if not len(ids):continue
        rot=Rotation.from_quat(np.stack([p[f'rot_{j}'][ids] for j in (1,2,3,0)],1)).as_matrix()
        normal=rot[:,:,2]
        tangent=vector-normal*(vector*normal).sum(1,keepdims=True)
        norm=np.linalg.norm(tangent,axis=1)
        ok=norm>1e-5
        ids,new_sigma,normal=ids[ok],new_sigma[ok],normal[ok]
        if not len(ids):continue
        tangent=tangent[ok]/norm[ok,None]
        quat=Rotation.from_matrix(np.stack([tangent,np.cross(normal,tangent),normal],axis=2)).as_quat()
        for j,k in enumerate((3,0,1,2)):p[f'rot_{j}'][ids]=quat[:,k]
        p['scale_1'][ids]=np.log(new_sigma)
        changed+=len(ids)
    tmp=path.with_suffix('.partial')
    data.write(tmp)
    tmp.replace(path)
    return changed
