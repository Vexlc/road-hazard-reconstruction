"""Ray-cast the exported seated GLB into held-out photographs and against independent laser samples."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gen2hole'))
sys.path.insert(0,str(ROOT/'gen2hole/downstream/sensors'))
from fit_road_cover import camera_matrix, ellipse, read, save
from road_geometry import Surface, glb_triangles, sha256


def mask_overlap(mesh, row, camera, center, basis, shift, crop):
    x0,y0,x1,y1=crop
    scale=4
    y,x=np.indices(((y1-y0)*scale,(x1-x0)*scale),dtype='f4')/scale
    rays=np.stack([x+x0-shift[0],y+y0-shift[1],np.ones_like(x)],axis=-1) @ np.linalg.inv(camera[1]).T @ basis
    eye=(camera[0]-center) @ basis
    flat=rays.reshape(-1,3)
    distance,part_ids,_=mesh.cast(np.broadcast_to(eye,flat.shape),flat)
    # The GLB carries a finite pavement apron. Extend its surrounding road
    # plane to block grazing rays that would otherwise see the shaft from below.
    road_t=-eye[2]/flat[:,2]
    at_plane=eye+road_t[:,None]*flat
    outside=np.any((at_plane[:,:2]<mesh.low[:2]) | (at_plane[:,:2]>mesh.high[:2]),axis=1)
    part_ids[outside & (road_t>0) & (road_t<distance)]=1
    parts=part_ids.reshape(x.shape)
    scores={}
    outlines={}
    for key,detected in [('frame_edge',np.isin(parts,[5,9,10,11])),('lid_seam',parts==9),
                         ('apron_edge',np.isin(parts,[5,8,9,10,11]))]:
        if key not in row:
            continue
        source=np.zeros(parts.shape,'u1')
        outline=np.array(row[key]) if key=='apron_edge' else ellipse(row[key],720)
        polygon=np.int32(np.round((outline-[x0,y0])*scale))
        cv2.fillPoly(source,[polygon],1)
        annotation=source.astype(bool)
        scores[key]=float((detected & annotation).sum()/(detected | annotation).sum())
        # Ignore small internal voids at the recessed seat in the display outline.
        contours,_=cv2.findContours(detected.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        outlines[key]=max(contours,key=cv2.contourArea).reshape(-1,2)/scale
    return scores,outlines


def check(args):
    spec=read(args.capture)
    bundle=args.run/'series'
    fit=read(bundle/'calibration/capture-fit.json')
    seated=[s for s in spec['states'] if not s.get('absent') and s.get('angle_deg')==0 and s.get('offset_m')==[0,0]]
    if len(seated)!=1:
        raise ValueError('Exactly one seated reference state is required')
    glb=bundle/'assets/raised'/seated[0]['id']/'asset.glb'
    mesh=Surface(glb)
    center=np.array(fit['center_world_m'])
    basis=np.array(fit['basis_columns'])
    if not np.allclose(basis.T@basis,np.eye(3)) or np.linalg.det(basis)<.9999:
        raise ValueError('Invalid local-to-world registration')
    raw=args.raw/fit['sequence']
    for path,digest in fit['raw_sha256'].items():
        if sha256(raw/path)!=digest:
            raise ValueError('Changed capture input '+path)
    images={}
    out=args.run/'reprojection'
    out.mkdir(exist_ok=True)
    for row in spec['holdout']:
        f=row['frame']
        name=row.get('camera',spec['camera'])
        key=f'{name}_{f:02}'
        if key in fit['fit_frames']+fit['registration_fit_frames'] or key not in fit['heldout_frames']:
            raise ValueError('Invalid camera split '+key)
        camera_dir=raw/'camera'/name
        cam=camera_matrix(read(camera_dir/'poses.json')[f],read(camera_dir/'intrinsics.json'))
        cxy=ellipse(row['frame_edge'])
        boundary=np.asarray(row.get('apron_edge',cxy))
        lo=np.floor(boundary.min(0)-40).astype(int)
        hi=np.ceil(boundary.max(0)+40).astype(int)
        crop=(*lo,*hi)
        src=np.array(Image.open(camera_dir/f'{f:02}.jpg').crop(crop))
        images[key]={}
        face=center+fit['lid_face_above_road_m']*basis[:,2]
        coeff=fit['local_pixel_registration'].get(name,np.zeros((2,2)))
        delta=[1.,1./((face-cam[0])@cam[1][2])]@np.asarray(coeff)
        outlines=[]
        for method,shift in [('native_pose',np.zeros(2)),('locally_registered',delta)]:
            scores,contours=mask_overlap(mesh,row,cam,center,basis,shift,crop)
            images[key][method]=scores
            outlines.append([contours[k] for k in ('frame_edge','apron_edge') if k in contours])
        im=cv2.resize(src,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        Image.fromarray(im).save(out/(key+'-photo.png'))
        def trace(target,contour,rgb):
            cv2.polylines(target,[np.int32(np.round(contour*3*16))],True,rgb,2,cv2.LINE_AA,shift=4)
        trace(im,cxy-lo,(255,211,88))
        if 'apron_edge' in row:
            trace(im,boundary-lo,(255,211,88))
        for method,contours,rgb in zip(('native','registered'),outlines,[(245,108,142),(62,225,208)]):
            view=im.copy()
            for contour in contours:
                trace(view,contour,rgb)
            Image.fromarray(view).save(out/(key+'-'+method+'.png'))
    returns={}
    all_errors=[]
    apron_errors=[]
    for f in spec['holdout_scans']:
        if f in fit['fit_scans']:
            raise ValueError('Repeated training scan')
        file=f'{f:02}.pkl.gz'
        cloud=pd.read_pickle(raw/'lidar'/file)
        labels=pd.read_pickle(raw/'annotations/semseg'/file)
        if not cloud.index.equals(labels.index):
            raise ValueError('Label/scan mismatch')
        road=cloud[labels['class'].isin([7,8,9,10])]
        points=(road[['x','y','z']].to_numpy()-center)@basis
        if 'apron_radius_m' in fit:
            profile=fit['apron_radius_m']
            angle=np.arctan2(points[:,1],points[:,0])
            radius=np.linalg.norm(points[:,:2],axis=1)
            boundary=np.interp(angle,np.arange(len(profile))*2*np.pi/len(profile),profile,period=2*np.pi)
            # Leave a centimetre at both material boundaries.
            keep=(radius>fit['frame_diameter_m']/2+.01)&(radius<boundary-.01)&(abs(points[:,2])<.15)
            apron=points[keep]
            if len(apron):
                origins=apron.copy()
                origins[:,2]=.5
                distance,parts,_=mesh.cast(origins,np.broadcast_to([0.,0.,-1.],origins.shape))
                if not (parts==8).all():
                    raise ValueError('Repair sample missed the exported apron')
                apron_errors.extend(np.abs(.5-distance-apron[:,2]))
        keep=(np.linalg.norm(points[:,:2],axis=1)<.4*fit['lid_diameter_m']) & (abs(points[:,2])<.15)
        selected=points[keep]
        if not len(selected):
            returns[str(f)]=dict(points=0)
            continue
        origins=selected.copy()
        origins[:,2]=.5
        dz=np.broadcast_to([0.,0.,-1.],origins.shape)
        distance,parts,_=mesh.cast(origins,dz)
        if not (parts==9).all():
            raise ValueError('Interior samples must hit the exported lid face')
        error=np.abs(.5-distance-selected[:,2])
        all_errors.extend(error)
        returns[str(f)]=dict(points=len(selected),median_abs_error_m=float(np.median(error)),
                             p95_abs_error_m=float(np.percentile(error,95)))
    if not all_errors:
        raise ValueError('No held-out laser samples')
    report=dict(asset=seated[0]['id'],glb_sha256=sha256(glb),capture_spec_sha256=sha256(args.capture),
                model_input_sha256=sha256(bundle/'calibration/capture-fit.json'),camera_mask_iou=images,
                lidar=returns,heldout_lid_returns=len(all_errors),lidar_abs_error_median_m=float(np.median(all_errors)),
                lidar_abs_error_p95_m=float(np.percentile(all_errors,95)),frame_diameter_m=fit['frame_diameter_m'],
                lid_diameter_m=fit['lid_diameter_m'],lid_face_above_road_m=fit['lid_face_above_road_m'])
    if apron_errors:
        report.update(heldout_apron_returns=len(apron_errors),apron_median_abs_height_error_m=float(np.median(apron_errors)),
                      apron_p95_abs_height_error_m=float(np.percentile(apron_errors,95)))
    for state in spec['states']:
        if state.get('absent'):
            continue
        folder=bundle/'assets/raised'/state['id']
        if 'lid_support' not in read(folder/'metadata.json'):
            continue
        fixed=Surface(folder/'asset.glb',exclude=(9,10,11))
        vertices=next(v for part,v,_ in glb_triangles(folder/'asset.glb') if part==10)
        origin=vertices.copy()
        origin[:,2]=fixed.high[2]+1
        dist,part,_=fixed.cast(origin,np.broadcast_to([0.,0.,-1.],origin.shape))
        # The underside can extend over the open shaft; only the load-bearing
        # frame, repair and adjoining road determine first contact.
        on_support=np.isin(part,[1,5,8])
        gaps=(vertices[:,2]-origin[:,2]+dist)[on_support]
        if not len(gaps) or abs(gaps.min())>2e-5:
            raise ValueError('Displaced GLB failed independent first-contact check')
        report.setdefault('displaced_support',{})[state['id']]=dict(support_samples=len(gaps),
            minimum_clearance_m=float(gaps.min()),samples_within_1mm=int((gaps<.001).sum()))
    save(args.out,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','raw','capture','out'):
        p.add_argument('--'+name,type=Path,required=True)
    check(p.parse_args())
