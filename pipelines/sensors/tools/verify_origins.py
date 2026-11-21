"""Test the front emitter translation against continuous beam lines at measured range steps."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation,Slerp
import yaml
from prepare_scans import camera_track,sensor_transform,RIG
from road_geometry import SITES,read_json,save_json


def run(raw,out):
    ext=yaml.safe_load(RIG.read_text());groups=[];breakdown={}
    for site in SITES:
        times,cam=camera_track(raw/site,'front_camera');main=cam@sensor_transform(ext,'front_camera');ts=times-times[0]
        rotation=Slerp(ts,Rotation.from_matrix(main[:,:3,:3]));position=interp1d(ts,main[:,:3,3],axis=0)
        before=len(groups)
        for frame in (9,20,49,69):
            cloud=pd.read_pickle(raw/site/'lidar'/f'{frame:02d}.pkl.gz')
            gt=cloud[cloud.d==1];t,idx,c=np.unique(gt.t.to_numpy(),return_inverse=True,return_counts=True)
            v=gt[['x','y','z']].to_numpy()-position(t-times[0])[idx]
            p=np.einsum('ni,nij->nj',v,rotation(t-times[0]).as_matrix()[idx])
            order=np.argsort(idx,kind='stable');starts=np.r_[0,c.cumsum()]
            for k in np.flatnonzero(c==180)[::2]:
                lines=p[order[starts[k]:starts[k+1]]].reshape(60,3,3)
                triples=np.stack([lines[:-2],lines[1:-1],lines[2:]],2).reshape(-1,3,3)
                distance=np.linalg.norm(triples,axis=2)
                selected=triples[(np.ptp(distance,axis=1)>1)&(distance.min(1)>5)]
                if len(selected):groups.append(selected)
        breakdown[site]=sum(len(g) for g in groups[before:])
    data=np.concatenate(groups);g=sensor_transform(ext,'front_gt')
    stats={}
    for name,offset in [('used_translation',g[:3,3]),('inverted_transform_translation',np.linalg.inv(g)[:3,3]),('shared_spinning_origin',np.zeros(3))]:
        q=data-offset;unit=q/np.linalg.norm(q,axis=2)[:,:,None]
        bend=np.linalg.norm(unit[:,2]+unit[:,0]-2*unit[:,1],axis=1)*(180/np.pi)
        stats[name]=dict(translation_m=offset.tolist(),median_angular_second_difference_deg=float(np.median(bend)),
                         p75_deg=float(np.percentile(bend,75)))
    assert stats['used_translation']['median_angular_second_difference_deg'] < .2*stats['inverted_transform_translation']['median_angular_second_difference_deg']
    checks={s:read_json(out/'base'/s/'source.json') for s in SITES}
    save_json(out/'checks/sensor_origins.json',dict(triplets=len(data),by_site=breakdown,method='Second difference of unit direction in full interleaved 180-return GT groups at scene range discontinuities',tested=stats,
        cross_camera_interpolated_main_origin_residual_m={s:c['cross_camera_origin_residual'] for s,c in checks.items()},
        largest_endpoint_extrapolation_s=max(v['max_extrapolation_s'] for c in checks.values() for v in c['frames'])))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for k in ['raw','out']:a.add_argument('--'+k,type=Path,required=True)
    p=a.parse_args();run(p.raw,p.out)
