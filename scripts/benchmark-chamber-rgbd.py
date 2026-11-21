"""Generate an independent oblique RGB-D calibration target for the chamber reconstruction CLI."""
import argparse
import json
from pathlib import Path
import runpy

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
    # Independently render a raised tread plane into a second pinhole camera.
    # No pose angle or translation from this fixture is supplied to the reconstruction.
    pose_eye=np.array([-.70,-.84,1.74])
    look=-pose_eye/np.linalg.norm(pose_eye)
    right=np.cross(look,[0,0,1]);right/=np.linalg.norm(right)
    camera_axes=np.column_stack((right,np.cross(look,right),look))
    pose_ray=local@camera_axes.T
    angle,yaw=np.radians([56,27])
    cos,sin=np.cos,np.sin
    rotation=np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]])@np.array(
        [[cos(angle),0,-sin(angle)],[0,1,0],[sin(angle),0,cos(angle)]])
    face_center=np.array([.13,-.072,.342*sin(angle)+.027])
    normal=rotation[:,2]
    plane_t=((face_center-pose_eye)@normal)/(pose_ray@normal)
    ray_hit=pose_eye+pose_ray*plane_t[:,:,None]
    face_local=(ray_hit-face_center)@rotation
    road_t=-pose_eye[2]/pose_ray[:,:,2]
    support=(plane_t>0)&(plane_t<road_t)&(np.hypot(face_local[:,:,0],face_local[:,:,1])<=.342)
    posed_depth=np.where(support,plane_t,road_t)+rng.normal(0,.002,(h,w))
    posed_depth[rng.random((h,w))<.1]=0
    corrupt=rng.random((h,w))<.035
    posed_depth[corrupt]+=.24
    frame_xy=pose_eye+pose_ray*road_t[:,:,None]
    r=np.hypot(frame_xy[:,:,0],frame_xy[:,:,1])
    rgb=np.full((h,w,3),[143,138,129],float)
    rgb[r<outer]=[84,80,74];rgb[r<inner]=[44,40,36]
    pattern=np.cos(face_local[:,:,0]*200)*np.cos(face_local[:,:,1]*170)
    rgb[support]=(132+26*pattern[support])[:,None]
    rgb+=rng.normal(0,3,(h,w,1))
    Image.fromarray(np.clip(rgb,0,255).astype('u1')).save(folder/'pose.png')
    Image.fromarray(np.rint(posed_depth*1000).astype('uint16')).save(folder/'pose_z_mm.png')
    Image.fromarray(support.astype('u1')*255).save(folder/'pose_top.png')
    cardinal=np.array([[-1,0,0],[0,1,0],[1,0,0],[0,-1,0]])
    def project_pose(points):
        v=(points-pose_eye)@camera_axes
        return (v[:,:2]/v[:,2:]*[fx,fy]+[cx,cy]).tolist()
    spec['pose_capture']=dict(image='pose.png',depth='pose_z_mm.png',lid_mask='pose_top.png',metres_per_unit=.001,
        intrinsics_px=[fx,fy,cx,cy],source_kind='synthetic_benchmark',frame_landmarks_xy=project_pose(outer*cardinal),
        lid_landmarks_xy=project_pose(face_center+.342*cardinal@rotation.T))
    spec['states'].append(dict(id='A98',label='深度观测盖板',pose_from_capture=True))
    (folder/'pose_truth.json').write_text(json.dumps(dict(center_m=face_center.tolist(),rotation=rotation.tolist(),
         angle_deg=56,diameter_m=.684),indent=2)+'\n')
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
    pose_summary=None
    if (fixture/'pose_truth.json').exists():
        expected=json.loads((fixture/'pose_truth.json').read_text())
        estimated=json.loads((run/'series/calibration/observed_pose.json').read_text())
        A=np.asarray(expected['rotation'])
        B=np.asarray(estimated['transform'])[:3,:3]
        rotation_error=float(np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1))))
        center_error=float(np.linalg.norm(np.array(expected['center_m'])-estimated['face_center_local_m']))
        mesh_parts=runpy.run_path(str(Path(__file__).with_name('check-chamber-family.py')))['mesh_parts']
        face,_=mesh_parts(run/'series/assets/raised/A98/asset.glb')['Lid face']
        local=(face-expected['center_m'])@A
        footprint_error=float(np.max(abs(np.ptp(local,axis=0)[:2]-expected['diameter_m'])))
        plane_residual=float(np.percentile(np.abs(local[:,2]),95))
        if rotation_error>.5 or center_error>.004 or footprint_error>.005 or plane_residual>.005:
            raise ValueError('Measured pose disagrees with independent fixture')
        pose_summary=dict(actual_angle_deg=expected['angle_deg'],recovered_angle_deg=estimated['angle_degrees'],
                          rotation_error_deg=rotation_error,face_center_error_m=center_error,
                          exported_glb_face_diameter_error_m=footprint_error,exported_glb_to_true_plane_p95_m=plane_residual)
    return dict(observed_pose=pose_summary,source_kind=gauge['source_kind'],truth_m=truth,estimated_m=gauge['measured'],absolute_error_m=err,
                road_plane=gauge['road_plane'],observations=gauge['observations'],
                format_checks=json.loads((run/'series/checks.json').read_text()),limitations='Analytic synthetic fixture with optical-Z noise, dropout and outliers.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--score-run',type=Path)
    a=p.parse_args()
    if a.score_run:print(json.dumps(score(a.folder,a.score_run),indent=2))
    else:make(a.folder)
