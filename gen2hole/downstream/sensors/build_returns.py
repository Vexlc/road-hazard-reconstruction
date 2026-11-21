"""Replay the two recorded LiDARs against the reconstructed asset surfaces."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from prepare_scans import write_npz
from road_geometry import CLASSES, SITES, Surface, load_inventory, read_json, save_json, sha256


def mask_at(field, w, h, xy):
    rows,cols=field.shape
    u=(xy[:,0]/w+.5)*(cols-1)
    v=(.5-xy[:,1]/h)*(rows-1)
    inside=(u>=0)&(u<=cols-1)&(v>=0)&(v<=rows-1)
    col=np.rint(u[inside]).astype('i4');row=np.rint(v[inside]).astype('i4')
    a=np.zeros(len(xy),bool);a[inside]=field[row,col]>.45
    return a


def project_asset(mesh, field, rays, placement, slot_index):
    o,v,r,old=rays['o'],rays['v'],rays['r'],rays['xyz']
    t,part,_=mesh.cast(o,v)
    finite=np.isfinite(t)
    safe_t=np.where(finite,t,0)
    hit=o+v*safe_t[:,None]
    down=(v[:,2]<-1e-5)&(o[:,2]>.01)
    plane_t=np.divide(-o[:,2],v[:,2],out=np.zeros_like(r),where=down)
    entry=o[:,:2]+v[:,:2]*plane_t[:,None]
    w,h=float(field['width']),float(field['length'])
    cut_at_entry=down & mask_at(field['defect_mask'],w,h,entry)
    cut_at_old=mask_at(field['defect_mask'],w,h,old[:,:2])
    near_entry=np.linalg.norm(entry-old[:,:2],axis=1)<=.18
    near_road=rays['road']&(abs(old[:,2])<=max(.055,5*placement['rms_m']))
    exposed_ground=near_road & cut_at_entry & (cut_at_old | near_entry)
    physical=finite & (part>1) & (t>0)
    above=hit[:,2]>.0015
    earlier=(t < r-.003)
    # A raised solid may occlude an existing vehicle. An earlier measured occluder wins.
    # Below the fitted tangent plane a beam must first enter an actual road cut.
    replace=physical & ((above & earlier) | exposed_ground)
    # If the measured ground is cut and the ray escapes the physical mesh, retain its
    # original row/time with no synthetic point. Do not fabricate an emitted beam.
    lost=~finite & exposed_ground & cut_at_old
    choice=replace|lost
    out_t=t[choice].copy()
    out_t[~replace[choice]]=np.nan
    B=np.asarray(placement['basis_columns']);c=np.asarray(placement['center_world'])
    out_xyz=hit[choice].astype('f8')@B.T+c
    out_xyz[~replace[choice]]=np.nan
    local_z=hit[choice,2].copy();local_z[~replace[choice]]=np.nan
    return dict(frame=rays['frame'][choice],row=rays['row'][choice],
                range_before_m=r[choice],range_after_m=out_t,xyz=out_xyz,local_z_m=local_z,
                action=np.where(replace[choice],1,2).astype('u1'),sensor=rays['sensor'][choice],
                part=part[choice],instance=np.full(choice.sum(),slot_index,'u1'),
                hazard_class=np.full(choice.sum(),CLASSES[placement['category']],'u1'))


def merge(edits):
    arrays={k:np.concatenate([x[k] for x in edits]) for k in edits[0]}
    rank=np.nan_to_num(arrays['range_after_m'],nan=np.inf)
    order=np.lexsort((arrays['instance'],rank,arrays['row'],arrays['frame']))
    same=(np.diff(arrays['frame'][order].astype('i4'))==0)&(np.diff(arrays['row'][order])==0)
    keep=order[np.r_[True,~same]] if len(order) else order
    out={k:v[keep] for k,v in arrays.items()}
    offset=np.r_[0,np.bincount(out.pop('frame').astype('i4'),minlength=80).cumsum()].astype('i4')
    return dict(frame_offsets=offset,**out),int(same.sum())


def run(parent,out,cache,first,last):
    assets,scenes,_=load_inventory(parent)
    @lru_cache(maxsize=9)
    def rays(site,slot):
        with np.load(cache/site/(slot+'.npz')) as f:return {k:f[k] for k in f.files}
    dest=out/'cases';dest.mkdir(parents=True,exist_ok=True)
    @lru_cache(maxsize=3)
    def model(name):
        root,_=assets[name]
        with np.load(root/'surface.npz') as src:
            field={k:src[k] for k in ['width','length','defect_mask']}
        return Surface(root/'asset.glb'),field
    for case,(path,scene) in sorted(scenes.items(),key=lambda pair:(int(pair[0].split('_')[1]),pair[0])):
        if not first <= int(case.split('_')[1]) <= last:
            continue
        site=scene['site']
        edits=[]
        for n,pl in enumerate(scene['placements'],1):
            mesh,field=model(pl['asset_id'])
            edits.append(project_asset(mesh,field,rays(site,pl['slot']),pl,n))
        arr,dups=merge(edits);dst=dest/(case+'.npz');write_npz(dst,**arr)
        per_asset=[]
        for n,pl in enumerate(scene['placements'],1):
            wh=(arr['instance']==n); valid=wh&(arr['action']==1)
            z=arr['local_z_m'][valid];ranges=arr['range_after_m'][valid]-arr['range_before_m'][valid]
            root,_=assets[pl['asset_id']]
            per_asset.append(dict(asset_id=pl['asset_id'],instance=n,slot=pl['slot'],category=pl['category'],
                center_world=pl['center_world'],basis_columns=pl['basis_columns'],plane_rms_m=pl['rms_m'],
                changed=int(wh.sum()),hits=int(valid.sum()),lost=int((wh&~valid).sum()),
                by_sensor=np.bincount(arr['sensor'][valid],minlength=2).tolist(),
                part_counts=np.bincount(arr['part'][valid],minlength=13).tolist(),
                z_range_m=[float(z.min()),float(z.max())] if len(z) else [],
                range_change_m=[float(ranges.min()),float(ranges.max())] if len(ranges) else []))
        save_json(dst.with_suffix('.json'),dict(case=case,site=site,
            scene=str(path.relative_to(parent)),scene_sha256=sha256(path),delta_sha256=sha256(dst),
            frames=80,original_beams_with_multiple_edits=dups,assets=per_asset))
        print(case,'changed returns',len(arr['row']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['parent','out','cache']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--first',type=int,default=1);p.add_argument('--last',type=int,default=37)
    a=p.parse_args();run(a.parent,a.out,a.cache,a.first,a.last)
