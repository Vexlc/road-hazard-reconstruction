"""Independent road-labelled, equal-frame height references for both LiDAR sensors."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from annotated_road import read
from run import atomic_json


def compare(samples,cache,site,sensor):
    values=[]
    for frame in range(1,80,3):
        points=read(cache,site,frame,sensor)
        if len(points)<3:continue
        distance,ids=cKDTree(points[:,:2]).query(samples[:,:2],k=3,workers=2)
        visible=(distance[:,0]<.08)&(distance[:,2]<.16)
        height=np.median(points[ids,2],axis=1)
        height[~visible]=np.nan
        values.append(height)
    values=np.array(values)
    witnesses=np.isfinite(values).sum(0)
    good=witnesses>=3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        reference=np.nanmedian(values,axis=0)
        variability=np.nanmedian(abs(values-reference[None]),axis=0)
    error=abs(samples[good,2]-reference[good])
    return {'sensor':sensor,'reference_frames':list(range(1,80,3)),
            'sufficient_support':len(error)>=1000,
            'matched_samples':int(good.sum()),'unmatched_or_fewer_than_3_frames':int((~good).sum()),
            'median_absolute_height_error_m':float(np.median(error)) if len(error) else None,
            'p95_absolute_height_error_m':float(np.quantile(error,.95)) if len(error) else None,
            'maximum_absolute_height_error_m':float(error.max()) if len(error) else None,
            'median_reference_temporal_mad_m':float(np.median(variability[good])) if good.any() else None,
            'over_5cm':int((error>.05).sum()),'over_10cm':int((error>.1).sum())}


def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'));records=[]
    for site in job['configs']:
        folder=Path(job['output'])/f'{site}_01';meta=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
        state=torch.load(folder/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)
        bg=state['background'];n=meta['host_road_guard']['points']
        xyz=bg['xyz'][-n:-n//2].detach().numpy().copy();q=bg['rotation'][-n:-n//2].detach().numpy()
        xyz+=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()[:,:,2]*.002
        samples=xyz[np.linspace(0,len(xyz)-1,min(10000,len(xyz)),dtype=int)]
        results=[compare(samples,job['annotated_road_cache'],site,sensor) for sensor in (1,0)]
        records.append({'site':site,'sampled_surfels':len(samples),'sensors':results})
        print('ANNOTATED HEIGHT CHECK',site,[(r['sensor'],r['matched_samples'],r['p95_absolute_height_error_m']) for r in results],flush=True)
        del state,bg
    atomic_json(args.output,{'method':'Dataset-labelled Road/marking returns; median of per-frame 3-neighbour height estimates; at least 3 distinct held-out frames; sensors reported separately',
               'fit_frames':list(range(0,80,3)),'reference_frames':list(range(1,80,3)),
               'source_annotations':'PandaSet original semantics, index-aligned with raw LiDAR',
               'scope':'Unobserved regions stay unverified; cross-sensor calibration differences are not adjusted away.',
               'scenes':records})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
