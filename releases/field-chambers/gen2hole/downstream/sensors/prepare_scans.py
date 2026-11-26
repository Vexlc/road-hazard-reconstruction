"""Pack the measured scans once; gather the original beams that can cross each asset slot."""
import argparse
from pathlib import Path
import shutil
import struct
import json

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import yaml

from road_geometry import SITES, load_inventory, read_json, save_json, sha256

RIG = Path(__file__).with_name('pandaset_extrinsics.yaml')
CAMERAS = ('front_left_camera','front_right_camera','left_camera','right_camera','back_camera')
ROAD_NAMES = {'Road', 'Lane Line Marking', 'Stop Line Marking', 'Other Road Marking'}


def transform(p, q):
    matrix = np.eye(4)
    matrix[:3,:3] = Rotation.from_quat([q[k] for k in 'xyzw']).as_matrix()
    matrix[:3,3] = [p[k] for k in 'xyz']
    return matrix


def sensor_transform(extrinsics, name):
    e = extrinsics[name]['extrinsic']['transform']
    return transform(e['translation'], e['rotation'])


def camera_track(site, name):
    root = site/'camera'/name
    times = np.asarray(read_json(root/'timestamps.json'))
    poses = np.stack([transform(p['position'],p['heading']) for p in read_json(root/'poses.json')])
    return times, poses


def interpolate(ts, values, query):
    """Linear interpolation, with endpoint velocities for the short end-of-sequence tails."""
    i = np.searchsorted(ts, query, side='right').clip(1, len(ts)-1)
    w = ((query-ts[i-1])/(ts[i]-ts[i-1]))[:,None]
    return values[i-1] + w*(values[i]-values[i-1])


def calibrate(site):
    ext = yaml.safe_load(RIG.read_text())
    ts, front = camera_track(site, 'front_camera')
    main = front @ sensor_transform(ext,'front_camera')
    # The forward emitter lies below the main emitter. The direction of this
    # translation is also checked on range discontinuities in full GT scan lines.
    offset = sensor_transform(ext,'front_gt')[:3,3]
    gt_center = main[:,:3,3] + main[:,:3,:3] @ offset
    errors = {}
    for camera in CAMERAS:
        times, poses = camera_track(site, camera)
        derived = poses @ sensor_transform(ext,camera)
        aligned = interpolate(times,derived[:,:3,3],ts[1:-1])
        err = np.linalg.norm(aligned-main[1:-1,:3,3],axis=-1)
        errors[camera] = {'median_m':float(np.median(err)), 'p95_m':float(np.percentile(err,95))}
    return ts, np.stack((main[:,:3,3],gt_center)), errors


def write_npz(path, compressed=True, **kwargs):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as h:
        (np.savez_compressed if compressed else np.savez)(h,**kwargs)
    tmp.replace(path)


def asset_box(path):
    with path.open('rb') as h:
        h.seek(12); size, = struct.unpack('<I',h.read(4)); h.read(4)
        g = json.loads(h.read(size))
    lo, hi = [], []
    for mesh in g['meshes']:
        if mesh['name'].startswith('Road '):
            continue
        for prim in mesh['primitives']:
            a=g['accessors'][prim['attributes']['POSITION']]
            a0,a1=a['min'],a['max']
            lo.append([a0[0],-a1[2],a0[1]])
            hi.append([a1[0],-a0[2],a1[1]])
    return np.min(lo,axis=0),np.max(hi,axis=0)


def possible(rays_o, rays_d, box):
    """Conservative positive ray/slab overlap, including parallel rays on the boundary."""
    low,high = box
    parallel = abs(rays_d)<1e-10
    outside = (parallel & ((rays_o<low)|(rays_o>high))).any(axis=1)
    den = np.where(parallel,1,rays_d)
    a,b = (low-rays_o)/den,(high-rays_o)/den
    near = np.where(parallel,-np.inf,np.minimum(a,b)).max(axis=1)
    far = np.where(parallel,np.inf,np.maximum(a,b)).min(axis=1)
    return ~outside & (far >= np.maximum(near,0))


def prepare(raw, parent, out, cache, site):
    source = raw/site
    assets, scenes, anchors = load_inventory(parent)
    slots = {slot:(center,basis) for (s,slot),(center,basis) in anchors.items() if s==site}
    bounds = {slot:[] for slot in slots}
    boxes = {k:asset_box(p/'asset.glb') for k,(p,_) in assets.items()}
    for _,scene in scenes.values():
        if scene['site']==site:
            for p in scene['placements']:
                bounds[p['slot']].append(boxes[p['asset_id']])
    boxes={k:(np.min(np.stack(v)[:,0],axis=0)-.12,np.max(np.stack(v)[:,1],axis=0)+.12) for k,v in bounds.items()}
    base = out/'base'/site; base.mkdir(parents=True,exist_ok=True)
    cache = cache/site;cache.mkdir(parents=True,exist_ok=True)
    ts, centers, errors = calibrate(source)
    write_npz(base/'sensor_track.npz',ts=ts,origins=centers)
    classes = read_json(source/'annotations/semseg/classes.json')
    road = [int(c) for c,n in classes.items() if n in ROAD_NAMES]
    assert len(road)==4
    save_json(base/'classes.json',classes)
    shutil.copyfile(source/'LICENSE.txt',base/'LICENSE.txt')
    candidates={k:[] for k in slots}; rows=[]
    for frame in range(len(ts)):
        lf=source/'lidar'/f'{frame:02d}.pkl.gz'; sf=source/'annotations/semseg'/f'{frame:02d}.pkl.gz'
        cloud=pd.read_pickle(lf);label=pd.read_pickle(sf)
        assert cloud.index.equals(label.index) and np.array_equal(cloud.index,np.arange(len(cloud)))
        xyz=cloud[['x','y','z']].to_numpy();t=cloud['t'].to_numpy();sensor=cloud['d'].to_numpy(dtype='u1')
        assert set(np.unique(sensor))=={0,1}
        intensity=cloud['i'].to_numpy(dtype='u1'); assert np.array_equal(intensity,cloud['i'])
        sem=label['class'].to_numpy(dtype='u1'); assert np.isfinite(xyz).all()
        dest=base/f'{frame:03d}.npz'
        write_npz(dest,xyz=xyz,time=t,sensor=sensor,intensity=intensity,semantic=sem)
        origins=np.empty_like(xyz)
        for d in (0,1):
            keep=sensor==d
            origins[keep]=interpolate(ts,centers[d],t[keep])
        delta=xyz-origins;dist=np.linalg.norm(delta,axis=1);direction=delta/np.maximum(dist[:,None],1e-12)
        supported=(dist>.05)&np.isfinite(t)
        for slot,(center,basis) in slots.items():
            o=(origins-center)@basis; v=direction@basis
            ids=np.flatnonzero(supported & possible(o,v,boxes[slot]))
            candidates[slot].append(dict(frame=np.full(len(ids),frame,'u1'),row=ids.astype('i4'),
                o=o[ids].astype('f4'),v=v[ids].astype('f4'),r=dist[ids].astype('f4'),
                xyz=((xyz[ids]-center)@basis).astype('f4'),sensor=sensor[ids],road=np.isin(sem[ids],road)))
        rows.append(dict(frame=frame,points=len(xyz),sensor_counts=np.bincount(sensor,minlength=2).tolist(),
                         extrapolated_points=int(((t<ts[0])|(t>ts[-1])).sum()),
                         max_extrapolation_s=float(max(ts[0]-t.min(),t.max()-ts[-1],0)),
                         lidar_sha256=sha256(lf),semseg_sha256=sha256(sf),base_sha256=sha256(dest)))
        if frame%20==19: print(site,frame+1,'scans',flush=True)
    for slot,items in candidates.items():
        arrays={k:np.concatenate([f[k] for f in items]) for k in items[0]}
        write_npz(cache/(slot+'.npz'),compressed=False,**arrays)
    save_json(base/'source.json',dict(site=site,frames=rows,rig_sha256=sha256(RIG),
          cross_camera_origin_residual=errors,road_ids=road,source_url='https://github.com/scaleapi/pandaset-devkit',
          candidate_bounds={s:[a.tolist() for a in box] for s,box in boxes.items()}))
    print('READY',site,[(s,sum(len(a['row']) for a in v)) for s,v in candidates.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','parent','out','cache']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--site',choices=SITES,required=True)
    a=p.parse_args()
    prepare(a.raw,a.parent,a.out,a.cache,a.site)
