"""Audit every raw scan/delta and select stratified first hits for an independent BVH."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from road_geometry import SITES,CLASSES,PARTS,load_inventory,read_json,save_json,sha256
from replay import load_scan


def audit(parent,out,raw):
    assets,scenes,_=load_inventory(parent);checks=out/'checks'
    selected=defaultdict(list)
    counts=Counter(); byclass=Counter();byd=Counter();bypart=Counter();zeros=0;worst=np.zeros(4)
    heights={cat:[np.inf,-np.inf] for cat in CLASSES}
    frames_touched=0; placements=0;both=0;nonroad_occluded=0
    for site in SITES:
        src=raw/site
        audit=read_json(out/'base'/site/'source.json')
        track=np.load(out/'base'/site/'sensor_track.npz')
        interp=[interp1d(track['ts']-track['ts'][0],p,axis=0,fill_value='extrapolate') for p in track['origins']]
        t0=track['ts'][0]
        cases={}
        for case,(path,info) in sorted(scenes.items()):
            if info['site']!=site:continue
            jf=out/'cases'/(case+'.json');df=jf.with_suffix('.npz')
            j=read_json(jf); assert sha256(df)==j['delta_sha256'] and sha256(path)==j['scene_sha256']
            with np.load(df) as f: z={k:f[k] for k in f.files}
            offsets=z['frame_offsets'];assert len(offsets)==81 and offsets[0]==0 and offsets[-1]==len(z['row']) and (np.diff(offsets)>=0).all()
            frames_touched+=int((np.diff(offsets)>0).sum()); counts['case_frames']+=80
            draws={}
            for a in j['assets']:
                n=a['instance'];pl=info['placements'][n-1];sel=(z['instance']==n)&(z['action']==1)
                ix=np.flatnonzero(sel);placements+=1;both+=int(len(np.unique(z['sensor'][sel]))==2)
                zeros+=int(len(ix)==0)
                if not len(ix):continue
                assert (z['hazard_class'][sel]==CLASSES[pl['category']]).all()
                byclass[pl['category']]+=len(ix)
                heights[pl['category']][0]=min(heights[pl['category']][0],float(z['local_z_m'][ix].min()))
                heights[pl['category']][1]=max(heights[pl['category']][1],float(z['local_z_m'][ix].max()))
                # Stratify by physical material, sensor and extremes of signed relief.
                chosen={int(ix[z['local_z_m'][ix].argmin()]),int(ix[z['local_z_m'][ix].argmax()])}
                for part in np.unique(z['part'][ix]):
                    for d in (0,1):
                        group=ix[(z['part'][ix]==part)&(z['sensor'][ix]==d)]
                        if len(group):
                            chosen.add(int(group[len(group)//2]))
                for idx in sorted(chosen):
                    f=int(np.searchsorted(offsets,idx,side='right')-1)
                    draws.setdefault(f,[]).append((idx,pl))
            cases[case]=(j,z,draws)
        for frame in range(80):
            bf=out/'base'/site/f'{frame:03d}.npz';entry=audit['frames'][frame]
            assert sha256(bf)==entry['base_sha256']
            cloud=pd.read_pickle(src/'lidar'/f'{frame:02d}.pkl.gz'); labels=pd.read_pickle(src/'annotations/semseg'/f'{frame:02d}.pkl.gz')
            with np.load(bf) as f:b={k:f[k] for k in f.files}
            for k,arr in [('xyz',cloud[['x','y','z']]),('intensity',cloud.i),('time',cloud.t),('sensor',cloud.d),('semantic',labels['class'])]:
                assert np.array_equal(b[k],np.asarray(arr))
            counts['raw_scans']+=1;counts['original_returns']+=len(cloud)
            for case,(j,z,draws) in cases.items():
                lo,hi=z['frame_offsets'][frame:frame+2]
                if lo==hi:continue
                ids=z['row'][lo:hi];assert np.all(np.diff(ids)>0) and ids.min()>=0 and ids.max()<len(cloud)
                sensor=b['sensor'][ids];assert np.array_equal(sensor,z['sensor'][lo:hi])
                origin=np.zeros((len(ids),3))
                for d in (0,1):
                    k=sensor==d;origin[k]=interp[d](b['time'][ids[k]]-t0)
                ov=b['xyz'][ids]-origin;before=np.linalg.norm(ov,axis=1)
                direction=ov/before[:,None]
                after=z['xyz'][lo:hi]-origin;dist=np.linalg.norm(after,axis=1)
                valid=z['action'][lo:hi]==1
                residual=np.linalg.norm(np.cross(direction,after),axis=1)
                local_error=0.
                for a in j['assets']:
                    k=(z['instance'][lo:hi]==a['instance'])&valid
                    if k.any():
                        zz=(z['xyz'][lo:hi][k]-a['center_world'])@np.array(a['basis_columns'])[:,2]
                        local_error=max(local_error,float(abs(zz-z['local_z_m'][lo:hi][k]).max()))
                if valid.any():
                    err=[residual[valid].max(),abs(before[valid]-z['range_before_m'][lo:hi][valid]).max(),
                         abs(dist[valid]-z['range_after_m'][lo:hi][valid]).max(),local_error]
                    worst=np.maximum(worst,err)
                others=~np.isin(b['semantic'][ids],audit['road_ids'])
                if others.any():
                    assert valid[others].all() and np.all(dist[others]<before[others]-.002)
                    nonroad_occluded+=int(others.sum())
                for dd,count in zip(*np.unique(sensor[valid],return_counts=True)):byd[str(dd)]+=int(count)
                for pp,count in zip(*np.unique(z['part'][lo:hi][valid],return_counts=True)):bypart[PARTS[pp]]+=int(count)
                counts['replacements']+=int(valid.sum());counts['no_return']+=int((~valid).sum())
                for idx,pl in draws.get(frame,[]):
                    k=idx-lo;B=np.array(pl['basis_columns']);c=pl['center_world']
                    selected[pl['asset_id']].append(dict(case=case,frame=frame,row=int(ids[k]),
                        o=((origin[k]-c)@B).tolist(),v=(direction[k]@B).tolist(),t=float(dist[k]),part=int(z['part'][idx])))
                if (frame in (4,33,60)) and case.endswith(('_01','_13','_37')):
                    full=load_scan(out,case,frame);changed=np.zeros(len(cloud),bool);changed[ids]=True
                    for k in ('xyz','intensity','semantic'):assert np.array_equal(full[k][~changed],b[k][~changed])
                    assert np.array_equal(full['time'],b['time']) and np.array_equal(full['sensor'],b['sensor'])
                    assert np.isnan(full['intensity'][changed]).all()
                    counts['roundtrip_nonempty_full_scans']+=1
        print('raw & ray checks',site,flush=True)
    assert max(worst)<.0002 and zeros==0 and len(selected)==len(assets)
    save_json(checks/'independent_samples.json',selected)
    result=dict(counts=dict(counts),sites=len(SITES),cases=len(scenes),asset_count=len(assets),placements=placements,
        placements_with_both_lidars=both,placements_without_returns=zeros,case_frames_with_edits=frames_touched,
        case_frames_without_edits=counts['case_frames']-frames_touched,by_category=dict(byclass),by_sensor=dict(byd),
        by_surface=dict(bypart),synthetic_local_height_range_m=heights,occluded_nonroad_source_returns=nonroad_occluded,
        max_lateral_ray_error_m=float(worst[0]),max_source_range_error_m=float(worst[1]),
        max_replay_range_error_m=float(worst[2]),max_local_height_error_m=float(worst[3]))
    save_json(checks/'scan_audit.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['parent','out','raw']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();audit(a.parent,a.out,a.raw)
