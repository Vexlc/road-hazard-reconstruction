"""Fit a seated circular cover from calibrated PandaSet photographs and world-space laser returns."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


EDGES = ('frame_edge', 'lid_seam')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def camera_matrix(pose, intrinsics):
    rotation = Rotation.from_quat([pose['heading'][k] for k in 'xyzw']).as_matrix()
    center = np.array([pose['position'][k] for k in 'xyz'])
    k = intrinsics
    projection = np.array([[k['fx'],0,k['cx']], [0,k['fy'],k['cy']], [0,0,1.]]) @ rotation.T
    return center, projection


def project(world, camera):
    center, projection = camera
    q = (world-center) @ projection.T
    return q[..., :2] / q[..., 2:3]


def ellipse(params, samples=256):
    u,v,a,b,deg = params
    angle = np.deg2rad(deg)
    t = np.arange(samples)*(2*np.pi/samples)
    return np.c_[a*np.cos(t),b*np.sin(t)] @ np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]) + [u,v]


def intersections(pixels, camera, point, normal):
    center, projection = camera
    rays = np.c_[pixels,np.ones(len(pixels))] @ np.linalg.inv(projection).T
    dist = ((point-center) @ normal)/(rays @ normal)
    if np.min(dist) <= 0:
        raise ValueError('Annotated edge does not intersect the road in front of the camera')
    return center + dist[:,None]*rays


def fit_road(points):
    keep = points
    for threshold in (.08,.045,.025,.018):
        center = keep.mean(axis=0)
        normal = np.linalg.svd(keep-center,full_matrices=False)[2][-1]
        keep = points[np.abs((points-center) @ normal)<threshold]
        if len(keep)<30:
            raise ValueError('Insufficient laser support for the road plane')
    center = keep.mean(axis=0)
    normal = np.linalg.svd(keep-center,full_matrices=False)[2][-1]
    return center, normal*np.sign(normal[2]), len(keep)


def fit(spec, raw, out):
    site = raw/spec['sequence']
    def camera_dir(name):
        return site/'camera'/name
    camera_root = camera_dir(spec['camera'])
    fit_rows, test_rows = spec['fit'], spec['holdout']
    if any(row.get('camera',spec['camera'])!=spec['camera'] for row in fit_rows):
        raise ValueError('The primary camera supplies the geometry fit and texture. Use registration_fit for additional cameras.')
    training = [r['frame'] for r in fit_rows]
    test = [r['frame'] for r in test_rows]
    reg_rows=spec.get('registration_fit',[])
    label=lambda row: row.get('camera',spec['camera'])+'_'+str(row['frame']).zfill(2)
    labels=[label(row) for row in fit_rows+reg_rows+test_rows]
    if not training or not test or len(set(labels)) != len(labels):
        raise ValueError('Use disjoint fitting and held-out camera frames')
    if not spec['fit_scans'] or not spec['holdout_scans'] or set(spec['fit_scans']) & set(spec['holdout_scans']):
        raise ValueError('Use disjoint fitting and held-out laser scans')
    sources = set()
    def source(path):
        sources.add(path)
        return path
    poses = read(source(camera_root/'poses.json'))
    intr = read(source(camera_root/'intrinsics.json'))
    cameras = [camera_matrix(p,intr) for p in poses]
    tracks={spec['camera']:cameras}
    for name in {row.get('camera',spec['camera']) for row in reg_rows+test_rows}-set(tracks):
        path=camera_dir(name)
        k=read(source(path/'intrinsics.json'))
        tracks[name]=[camera_matrix(p,k) for p in read(source(path/'poses.json'))]
    def get_camera(row):
        return tracks[row.get('camera',spec['camera'])][row['frame']]
    def road_scan(frame):
        name = f'{frame:02}.pkl.gz'
        cloud = pd.read_pickle(source(site/'lidar'/name))
        seg = pd.read_pickle(source(site/'annotations/semseg'/name))
        if not cloud.index.equals(seg.index):
            raise ValueError('Label and beam indices disagree')
        return cloud.loc[seg['class'].isin([7,8,9,10]),['x','y','z','d']].to_numpy()
    fit_cloud = np.concatenate([road_scan(f) for f in spec['fit_scans']])
    primary = fit_rows[0]
    camera = cameras[primary['frame']]
    xyz = fit_cloud[:,:3]
    forward = (xyz-camera[0]) @ camera[1][2] > 0
    candidates = xyz[forward]
    u,v,a,b,_ = primary['frame_edge']
    pixels = project(candidates,camera)
    near = candidates[((pixels[:,0]-u)/a)**2+((pixels[:,1]-v)/b)**2 < .9]
    if len(near)<5:
        raise ValueError('No road laser support inside the first image ellipse')
    approx = np.median(near,axis=0)
    dist = np.linalg.norm(xyz[:,:2]-approx[:2],axis=1)
    # Ring outside the photographed fixture; excludes other elevations at the same XY.
    radius = np.linalg.norm(intersections(np.array([[u-a,v],[u+a,v]]), camera, approx, np.array([0,0,1]))[0]-approx)
    road = xyz[(dist>radius*1.4)&(dist<max(1.6,radius*3.2))&(np.abs(xyz[:,2]-approx[2])<.25)]
    origin, normal, n_road = fit_road(road)
    if normal[2] < .85:
        raise ValueError('Expected a road-like laser plane')
    axis = camera[1][0]-intr['cx']*camera[1][2]
    axis = axis-normal*(axis@normal)
    axis /= np.linalg.norm(axis)
    basis = np.column_stack([axis,np.cross(normal,axis),normal])
    plane_xy = (xyz-origin) @ basis[:,:2]

    def estimate(level):
        samples = [(cameras[row['frame']],i,ellipse(row[key],96)) for row in fit_rows for i,key in enumerate(EDGES)]
        def on_plane(cam,px):
            return (intersections(px,cam,origin+level*normal,normal)-origin) @ basis[:,:2]
        lifted = [on_plane(cam,px) for cam,i,px in samples]
        c0 = np.median(lifted[0],axis=0)
        r0 = [np.median(np.linalg.norm(arr-c0,axis=1)) for arr in lifted[:2]]
        # Convert the radial constraint to first-order pixel distance. Image noise
        # near the top/bottom of a foreshortened ellipse has higher metric uncertainty.
        terms = [(xy,on_plane(cam,px+[.1,0])-xy,on_plane(cam,px+[0,.1])-xy,i)
                 for (cam,i,px),xy in zip(samples,lifted)]
        def residual(p):
            errors=[]
            for xy,du,dv,i in terms:
                delta=xy-p[:2]
                distance=np.linalg.norm(delta,axis=1)
                gradient=delta/distance[:,None]
                scale=np.hypot((du*gradient).sum(1),(dv*gradient).sum(1))/.1
                errors.extend((distance-p[i+2])/scale)
            return errors
        solution=least_squares(residual,[*c0,*r0],loss='soft_l1',f_scale=1.)
        if not solution.success or not 0 < solution.x[3] < solution.x[2]:
            raise ValueError('Invalid fitted circle radii')
        return solution.x
    first = estimate(0.)
    heights = (xyz-origin) @ normal
    middle = (np.linalg.norm(plane_xy-first[:2],axis=1) < first[3]*.8) & (abs(heights)<.15)
    if middle.sum()<8:
        raise ValueError('Insufficient measured laser returns on lid')
    level = float(np.median(heights[middle]))
    cx,cy,outer,inner = estimate(level)
    center = origin+basis[:,:2]@[cx,cy]
    def grid_world(x,y):
        return center + x[...,None]*axis + y[...,None]*basis[:,1] + level*normal
    angle = np.linspace(0,2*np.pi,720,endpoint=False)
    def render_circle(r, cam):
        return project(grid_world(np.cos(angle)*r,np.sin(angle)*r),cam)
    # A local, image-space registration for this static target. A constant image
    # shift plus a term proportional to inverse depth models the first-order
    # effect of small calibration offsets. Always report native poses alongside it.
    def predictor(cam):
        return [1.,1./((center+level*normal-cam[0]) @ cam[1][2])]
    alignment={}
    for name in tracks:
        observations=[r for r in fit_rows+reg_rows if r.get('camera',spec['camera'])==name]
        if len(observations)<2:
            continue
        design=[]; shifts=[]
        for row in observations:
            cam=get_camera(row)
            loop=render_circle(outer,cam)
            design.append(predictor(cam))
            shifts.append(row['frame_edge'][:2]-(loop.min(0)+loop.max(0))/2)
        if np.linalg.matrix_rank(design)<2:
            raise ValueError('Local registration needs distinct camera distances')
        alignment[name]=np.linalg.lstsq(design,shifts,rcond=None)[0]
    def adjustment(name,cam):
        return predictor(cam)@alignment[name] if name in alignment else np.zeros(2)
    scores = {}
    for row in fit_rows+reg_rows+test_rows:
        f = row['frame']
        name=row.get('camera',spec['camera'])
        key_frame=label(row)
        cam=get_camera(row)
        img_file = source(camera_dir(name)/f'{f:02}.jpg')
        im = cv2.imread(str(img_file))
        scores[key_frame] = {}
        for key,r in zip(EDGES,[outer,inner]):
            if key not in row:
                continue
            annotation = ellipse(row[key],720)
            pred = render_circle(r,cam)
            corrected=pred+adjustment(name,cam)
            local_min = np.min([a.min(0) for a in (annotation,pred,corrected)],axis=0).astype(int)-2
            shape = np.ceil(np.max([a.max(0) for a in (annotation,pred,corrected)],axis=0)-local_min).astype(int)+3
            def mask(xy):
                m=np.zeros(shape[::-1]*4,np.uint8)
                cv2.fillPoly(m,[np.int32(np.round((xy-local_min)*4))],1)
                return m.astype(bool)
            scores[key_frame][key]={}
            aa=mask(annotation)
            for method,loop in [('native_pose',pred),('locally_registered',corrected)]:
                pp=mask(loop)
                distance=np.r_[cKDTree(annotation).query(loop)[0],cKDTree(loop).query(annotation)[0]]
                scores[key_frame][key][method]=dict(mask_iou=float((aa&pp).sum()/(aa|pp).sum()),
                    median_boundary_px=float(np.median(distance)),p95_boundary_px=float(np.percentile(distance,95)))
            if key=='frame_edge':
                box=tuple(np.round([annotation[:,0].min()-27,annotation[:,1].min()-27,annotation[:,0].max()+27,annotation[:,1].max()+27]).astype(int))
                crop=Image.fromarray(im[:,:,::-1]).crop(box)
                crop.save(out/(key_frame+'.png'))
                if row in test_rows:
                    for points,color in ((annotation,(249,194,72)),(pred,(234,107,142)),(corrected,(48,227,208))):
                        cv2.polylines(im,[np.int32(np.round(points*16))],True,color[::-1],1,cv2.LINE_AA,shift=4)
                    Image.fromarray(im[:,:,::-1]).crop(box).save(out/('holdout-'+key_frame+'.png'))
    # Warp only the two fitting photographs, with higher pixel-area views weighted more.
    pixels=480
    y,x=np.indices((pixels,pixels),dtype=float)
    ortho=grid_world((x-239.5)*outer/157,(239.5-y)*outer/157)
    maps=[]; weights=[]
    for f in training:
        cam=cameras[f]
        uv=(project(ortho,cam)+adjustment(spec['camera'],cam)).astype('f4')
        rgb=cv2.imread(str(camera_root/f'{f:02}.jpg'))[:,:,::-1]
        if not ((uv.min(axis=(0,1))>=0).all() and (uv.max(axis=(0,1))<[rgb.shape[1]-1,rgb.shape[0]-1]).all()):
            raise ValueError('The texture extends outside a fitting photograph')
        maps.append(cv2.remap(rgb,uv,None,cv2.INTER_LINEAR))
        weights.append(abs(cv2.contourArea(render_circle(outer,cam).astype('f4'))))
    Image.fromarray(np.uint8(np.round(np.average(maps,axis=0,weights=weights)))).save(out/'surface.png')
    checks={}
    for f in spec['holdout_scans']:
        cloud=road_scan(f)
        local=(cloud[:,:3]-center)@basis
        d=np.linalg.norm(local[:,:2],axis=1)
        face=(d<inner*.8)&(np.abs(local[:,2])<.15)
        pavement=(d>outer*1.4)&(d<1.6)&(np.abs(local[:,2])<.25)
        checks[str(f)]=dict(face_points=int(face.sum()),road_points=int(pavement.sum()),
            median_abs_face_z_error_m=float(np.median(abs(local[face,2]-level))) if face.any() else None,
            median_abs_road_z_error_m=float(np.median(abs(local[pavement,2]))) if pavement.any() else None,
            face_points_by_emitter={str(int(k)):int((cloud[face,3]==k).sum()) for k in np.unique(cloud[face,3])})
    hashes={str(p.relative_to(site)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    report=dict(sequence=spec['sequence'],camera=spec['camera'],fit_frames=[label(r) for r in fit_rows],registration_fit_frames=[label(r) for r in reg_rows],
                heldout_frames=[label(r) for r in test_rows],local_pixel_registration={k:v.tolist() for k,v in alignment.items()},
                fit_scans=spec['fit_scans'],fit_plane_inliers=n_road,fit_lid_returns=int(middle.sum()),heldout_scans=checks,
                fit_lid_p10_p90_height_m=np.percentile(heights[middle],[10,90]).tolist(),
                center_world_m=center.tolist(),basis_columns=basis.tolist(),frame_diameter_m=float(2*outer),
                lid_diameter_m=float(2*inner),lid_face_above_road_m=level,pixel_checks=scores,raw_sha256=hashes)
    save(out/'capture-fit.json',report)
    def landmarks(radius):
        return [[239.5-radius,239.5],[239.5,239.5-radius],[239.5+radius,239.5],[239.5,239.5+radius]]
    save(out/'survey.json',dict(family=spec['family'],credit='PandaSet, Scale AI, Inc. and Hesai Photonics Technology Co., Ltd. CC BY 4.0 and PandaSet Dataset Terms.',
        source_url='https://pandaset.org/',license_url='https://creativecommons.org/licenses/by/4.0/',
        modifications='Annotated, road-plane registered and blended fitting photographs. Geometry below the observed surface and alternative lid states are designed.',
        capture_record='capture-fit.json',capture_crop=label(primary)+'.png',frame=dict(image='surface.png',landmarks_xy=landmarks(157),pavement_patch_xyxy=[8,8,76,76]),
        lid=dict(image='surface.png',landmarks_xy=landmarks(157*inner/outer)),
        frame_diameter_m=2*outer,lid_diameter_m=2*inner,aperture_diameter_m=2*inner*.94,
        shaft_depth_m=spec['shaft_depth_design_m'],lid_thickness_m=spec['lid_thickness_design_m'],states=spec['states']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new output directory')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as temp:
        stage=Path(temp)/'capture'
        stage.mkdir()
        spec=read(args.capture)
        fit(spec,args.raw,stage)
        shutil.copyfile(args.capture,stage/'annotations.json')
        shutil.copyfile(args.raw/spec['sequence']/'LICENSE.txt',stage/'Dataset-Terms.txt')
        stage.rename(args.out)


if __name__=='__main__':
    main()
