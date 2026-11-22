"""Surface-constrained degree-1 SH export using the original road project's renderer."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
PART_IMPORTANCE = {
    'Road adjoining the fracture': .7, 'Road around the aperture': .62,
    'Fracture lip and exposed bed': 3.1, 'Water surface': 2.,
    'Chipped concrete collar': 2.1, 'Inspection shaft': .85,
    'Cast ring and cover seat': 4.8, 'Shaft bottom': .42,
    'Lid face': 4.8, 'Lid fragment face': 4.8, 'Lid underside with casting ribs': 1.9, 'Lid rolled edge': 2.8,
}


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1.e-12)


def at_surface(img, local, width, length):
    return map_coordinates(img, ((.5-local[:,1]/length)*(img.shape[0]-1),
                                 (.5+local[:,0]/width)*(img.shape[1]-1)), order=1, mode='constant')


def make_one(dest, cache):
    from gaussian_sh import make_gaussians, save_ply
    from visibility import ray_ambient, ray_scene, sun_visibility
    meta=json.loads((dest/'metadata.json').read_text())
    with np.load(cache/(meta['id']+'.npz')) as data: m=dict(data)
    with np.load(dest/'surface.npz') as data: field=dict(data)
    v, f=m['xyz'], m['faces']
    tri=v[f]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=np.linalg.norm(cross,axis=1)*.5
    weights=np.array([PART_IMPORTANCE.get(str(k),2.4) for k in m['part_names']])[m['face_part']]
    road=m['road'].astype(bool); w=float(field['width']); l=float(field['length'])
    # The delivery mesh can carry a generous inspection apron. Allocate the native
    # point budget only to surfaces participating in the host cut.
    allowed=at_surface(field['alpha'],tri.mean(1),w,l)
    weights[road & (allowed<.001)]=0
    measure=np.cumsum(area*weights,dtype='f8')
    count=int(np.clip(measure[-1]*157000,260000,630000))
    draws=qmc.Sobol(3,scramble=True,seed=meta['seed']).random_base2(int(np.ceil(np.log2(count))))[:count]
    faces=np.searchsorted(measure,draws[:,0]*measure[-1],side='right')
    s=np.sqrt(draws[:,1]); bary=np.stack([1-s,s*(1-draws[:,2]),s*draws[:,2]],1).astype('f4')
    xyz=np.einsum('nvc,nv->nc',tri[faces],bary)
    opacity=np.where(road[faces],at_surface(field['alpha'],xyz,w,l),1.)
    keep=opacity>.008
    faces,xyz,bary,opacity=faces[keep],xyz[keep],bary[keep],opacity[keep]
    norm=unit(cross[faces])
    tan=unit(np.cross(np.broadcast_to([0,0,1.],norm.shape),norm))
    level=abs(norm[:,2])>.82
    tan[level]=unit(np.cross(np.broadcast_to([0.,1.,0.],norm[level].shape),norm[level]))
    bitan=np.cross(norm,tan)
    uv=np.einsum('nvc,nv->nc',m['uv'][f[faces]],bary)
    mat=m['material'][faces]
    rgb=np.empty((len(xyz),3),'f4')
    for i in np.unique(mat):
        ids=mat==i; tex=m['texture_'+str(i)]
        rc=[(1-uv[ids,1])*(tex.shape[0]-1),uv[ids,0]*(tex.shape[1]-1)]
        rgb[ids]=np.stack([map_coordinates(tex[:,:,k],rc,order=1,mode='nearest') for k in range(3)],1)/255.
    rng=np.random.default_rng(meta['seed']+8167)
    geo=ray_scene({'vertices':v,'faces':f})
    ao=ray_ambient(geo,xyz,norm,tan,bitan,rng).astype('f4')
    direct=sun_visibility(geo,xyz,norm,[-.35,-.25,1])
    spacing=np.sqrt(measure[-1]/(count*weights[faces]))
    names=m['part_names']; part=m['face_part'][faces]
    # Residual light in the photographed lining and the sky's diffuse bounce keep
    # the observed brick/concrete pattern legible in the low-order SH renderer.
    bounce=np.where(np.isin(part,np.flatnonzero(np.isin(names,['Inspection shaft','Shaft bottom']))),.17,.11)
    ambient=bounce+(1-bounce)*ao
    samples=dict(name=meta['id'],local_xyz=xyz,normals=norm,rotation_matrices=np.stack([tan,bitan,norm],-1),
                 scales=np.c_[.85*spacing,.85*spacing,np.full(len(spacing),.00039)],alpha=opacity,
                 normal=(norm+1)/2,color=rgb,arm=np.c_[np.ones(len(mat)),m['parameters'][mat]],ao=ambient[:,None],
                 sun_visibility=direct,road_mask=road[faces].astype('f4'),geometric_ao=True)
    state,err=make_gaussians(samples)
    save_ply(state,dest/'asset.ply')
    # Compact native material labels stay in splat order, including road points underneath a displaced cover.
    np.savez_compressed(dest/'splat_material.npz',road=road[faces],ambient_visibility=np.uint8(ao*255+.5),
                        part=part.astype('u1'),part_names=names)
    meta.update(gaussians=len(faces),sh_degree=1,appearance_sh_fit_mae=err,gaussian_sigma_median_m=float(np.median(spacing)*.85))
    (dest/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(meta['id'],len(faces),'splats',flush=True)
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('ids',nargs='*')
    args=parser.parse_args()
    catalog=ROOT/'assets'
    for file in sorted(catalog.glob('*/*/metadata.json')):
        if args.ids and file.parent.name not in args.ids:continue
        make_one(file.parent,args.cache)
    manifest=[]
    for file in sorted(catalog.glob('*/*/metadata.json')):
        m=json.loads(file.read_text()); d=file.parent.relative_to(catalog)
        if not (file.parent/'asset.ply').is_file():continue
        m.update(ply=str(d/'asset.ply'),glb=str(d/'asset.glb'),surface=str(d/'surface.npz'),metadata=str(d/'metadata.json'))
        manifest.append(m)
    tmp=catalog/'asset_manifest.tmp'
    tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(catalog/'asset_manifest.json')


if __name__=='__main__':
    main()
