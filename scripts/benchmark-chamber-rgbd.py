"""Generate an independent oblique RGB-D calibration target for the chamber reconstruction CLI."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make(folder):
    folder.mkdir(parents=True,exist_ok=False)
    w,h=640,480
    fx,fy,cx,cy=540.,532.,319.5,239.5
    outer,inner,shaft_depth=.407,.325,.710
    eye=np.array([.12,-.88,1.72])
    forward=-eye/np.linalg.norm(eye)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    basis=np.column_stack((right,down,forward))
    y,x=np.indices((h,w))
    local=np.stack(((x-cx)/fx,(y-cy)/fy,np.ones((h,w))),axis=-1)
    direction=local@basis.T
    def at(t):
        return eye+direction*t[:,:,None]
    ground=-eye[2]/direction[:,:,2]
    ground_rad=np.linalg.norm(at(ground)[:,:,:2],axis=-1)
    opening=ground_rad<inner
    floor=(-shaft_depth-eye[2])/direction[:,:,2]
    bottom=opening&(np.linalg.norm(at(floor)[:,:,:2],axis=-1)<inner)
    a=(direction[:,:,:2]**2).sum(-1)
    b=2*(direction[:,:,:2]@eye[:2])
    c=eye[:2]@eye[:2]-inner**2
    wall=(-b+np.sqrt(np.maximum(0,b*b-4*a*c)))/(2*a)
    distance=np.where(opening,np.where(bottom,floor,wall),ground)
    rng=np.random.default_rng(147)
    color=np.full((h,w,3),[147,141,133],dtype=float)
    color[ground_rad<outer]=[79,76,70]
    color[opening]=[64,58,54]
    color[bottom]=[48,41,36]
    color+=rng.normal(0,5,(h,w,1))
    Image.fromarray(np.clip(color,0,255).astype('u1')).save(folder/'frame.png')
    optical=distance+rng.normal(0,.0018,(h,w))
    corrupt=rng.random((h,w))<.04
    optical[corrupt]+=rng.uniform(.10,.42,corrupt.sum())
    optical[rng.random((h,w))<.12]=0
    Image.fromarray(np.rint(optical*1000).astype('uint16')).save(folder/'z_mm.png')
    Image.fromarray(np.uint8(bottom)*255).save(folder/'bottom.png')
    def landmarks(radius):
        world=np.array([[-radius,0,0],[0,radius,0],[radius,0,0],[0,-radius,0]])
        camera=(world-eye)@basis
        return (camera[:,:2]/camera[:,2:]*[fx,fy]+[cx,cy]).tolist()
    ly,lx=np.indices((480,480))
    lid=np.full((480,480,3),146.,dtype=float)
    circle=np.hypot(lx-239.5,ly-239.5)<157
    tread=(np.sin(lx*.35)*np.sin(ly*.35)>0)*26.
    lid[circle]=(95+tread[circle])[:,None]
    Image.fromarray(lid.astype('u1')).save(folder/'lid.png')
    spec=dict(family='calibration-target',frame=dict(image='frame.png',landmarks_xy=landmarks(outer),
              aperture_landmarks_xy=landmarks(inner),pavement_patch_xyxy=[32,32,188,125]),
              lid=dict(image='lid.png',landmarks_xy=[[82.5,239.5],[239.5,82.5],[396.5,239.5],[239.5,396.5]]),
              rgbd=dict(depth='z_mm.png',bottom_mask='bottom.png',intrinsics_px=[fx,fy,cx,cy],metres_per_unit=.001,
                        source_kind='synthetic_benchmark'),lid_diameter_m=.684,lid_thickness_m=.028,
              states=[dict(id='M97',label='标定敞井',absent=True),dict(id='A97',label='标定掀盖',angle_deg=50,offset_m=[0,0])])
    (folder/'survey.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
    truth=dict(frame_diameter_m=2*outer,aperture_diameter_m=2*inner,shaft_depth_m=shaft_depth)
    (folder/'truth.json').write_text(json.dumps(truth,indent=2)+'\n')


def score(fixture,run):
    truth=json.loads((fixture/'truth.json').read_text())
    gauge=json.loads((run/'series/calibration/depth_measurements.json').read_text())
    err={k:abs(v-gauge['measured'][k]) for k,v in truth.items()}
    if max(err.values())>.006:
        raise ValueError(f'Calibrated fixture exceeds 6 mm tolerance: {err}')
    for path in (run/'series/assets').glob('*/*/metadata.json'):
        meta=json.loads(path.read_text())
        values=[meta['frame_width_m'],meta['width_m'],meta['depth_measured_m'],-meta['lowest_z_m']]
        if not np.allclose(values,[truth[k] for k in truth]+[truth['shaft_depth_m']],atol=.006):
            raise ValueError(path)
    return dict(source_kind=gauge['source_kind'],truth_m=truth,estimated_m=gauge['measured'],absolute_error_m=err,
                road_plane=gauge['road_plane'],observations=gauge['observations'],
                format_checks=json.loads((run/'series/checks.json').read_text()),limitations='Analytic synthetic fixture with optical-Z noise, dropout and outliers.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--score-run',type=Path)
    a=p.parse_args()
    if a.score_run:print(json.dumps(score(a.folder,a.score_run),indent=2))
    else:make(a.folder)
