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
    if not np.isfinite(params).all() or not 0 < b <= a:
        raise ValueError('Expected a finite pixel ellipse with positive ordered radii')
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


def apron_profile(rows, cameras, center, basis, correction, edge_height):
    """Intersect each labelled repair polygon with 144 radial lines in the road plane."""
    angles=np.arange(144)*(2*np.pi/144)
    directions=np.c_[np.cos(angles),np.sin(angles)]
    profiles=[]
    def cross(a,b):
        return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for row in rows:
        cam=cameras[row['frame']]
        vertices=np.asarray(row['apron_edge'],float)
        if vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<3 or not np.isfinite(vertices).all():
            raise ValueError('Expected a finite repair polygon with at least three pixel coordinates')
        px=vertices-correction(cam)
        verts=(intersections(px,cam,center+edge_height*basis[:,2],basis[:,2])-center)@basis[:,:2]
        edges=np.roll(verts,-1,axis=0)-verts
        denom=cross(directions[:,None,:],edges)
        with np.errstate(divide='ignore',invalid='ignore'):
            dist=cross(verts,edges)/denom
            fraction=cross(verts,directions[:,None,:])/denom
        good=(dist>0)&(fraction>=0)&(fraction<=1)
        radii=np.min(np.where(good,dist,np.inf),axis=1)
        if not np.isfinite(radii).all():
            raise ValueError('Repair boundary must surround the metal component')
        profiles.append(radii)
    return np.mean(profiles,axis=0)


def estimate_circles(rows, cameras, origin, basis, level, clearance):
    """Fit concentric metric outlines using the pixel uncertainty of each observed edge."""
    normal=basis[:,2]
    seam_observed=all('lid_seam' in row for row in rows)
    if any('lid_seam' in row for row in rows)!=seam_observed:
        raise ValueError('Mark the lid seam consistently across the geometry fitting photographs')
    measured_edges=EDGES if seam_observed else EDGES[:1]
    samples = [(cameras[row['frame']],i,ellipse(row[key],96)) for row in rows for i,key in enumerate(measured_edges)]
    def on_plane(cam,px):
        return (intersections(px,cam,origin+level*normal,normal)-origin) @ basis[:,:2]
    lifted = [on_plane(cam,px) for cam,i,px in samples]
    c0 = np.median(lifted[0],axis=0)
    r0 = [np.median(np.linalg.norm(arr-c0,axis=1)) for arr in lifted[:len(measured_edges)]]
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
    fitted=solution.x
    if not seam_observed:
        clearance=float(clearance)
        if not 0 < clearance < .03:
            raise ValueError('An unresolved lid seam needs a positive designed radial clearance below 30 mm')
        fitted=np.r_[fitted,fitted[2]-clearance]
    if not solution.success or not 0 < fitted[3] < fitted[2]:
        raise ValueError('Invalid fitted circle radii')
    return fitted


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
    minimum=radius*1.4
    if 'apron_edge' in primary:
        approx_outline=intersections(np.array(primary['apron_edge']),camera,approx,np.array([0,0,1]))
        minimum=max(minimum,1.12*np.linalg.norm(approx_outline[:,:2]-approx[:2],axis=1).max())
    road = xyz[(dist>minimum)&(dist<max(1.6,radius*3.2,minimum+.6))&(np.abs(xyz[:,2]-approx[2])<.25)]
    origin, normal, n_road = fit_road(road)
    if normal[2] < .85:
        raise ValueError('Expected a road-like laser plane')
    axis = camera[1][0]-intr['cx']*camera[1][2]
    axis = axis-normal*(axis@normal)
    axis /= np.linalg.norm(axis)
    basis = np.column_stack([axis,np.cross(normal,axis),normal])
    plane_xy = (xyz-origin) @ basis[:,:2]

    seam_observed='lid_seam' in primary
    def estimate(level):
        return estimate_circles(fit_rows,cameras,origin,basis,level,spec.get('lid_radius_clearance_design_m',0.))
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
    apron=None
    radius_px=157.
    if any('apron_edge' in r for r in fit_rows):
        if not all('apron_edge' in r for r in fit_rows):
            raise ValueError('Annotate the apron in each geometry fitting frame')
        def radial_fraction(x,y):
            theta=np.arctan2(y,x)
            boundary=np.interp(theta,np.arange(len(apron))*(2*np.pi/len(apron)),apron,period=2*np.pi)
            return (np.hypot(x,y)-outer)/(boundary-outer)
        rel_xy=plane_xy-[cx,cy]
        knots=np.linspace(0.,1.,5)
        edge_height=0.
        # A raised patch's boundary pixels intersect a different plane from the
        # surrounding asphalt. Alternate the polygon lift and laser height estimates.
        for _ in range(5):
            apron=apron_profile(fit_rows,cameras,center,basis,lambda cam: adjustment(spec['camera'],cam),edge_height)
            if np.min(apron)<=outer:
                raise ValueError('The measured pavement repair must surround the entire casting')
            fraction=radial_fraction(*rel_xy.T)
            support=[]; profile=[]
            for knot in knots:
                keep=(abs(fraction-knot)<=.18)&(fraction>=0)&(fraction<=1)&(abs(heights)<.12)
                if keep.sum()<5:
                    raise ValueError('Insufficient laser support on the repair border')
                profile.append(float(np.median(heights[keep])))
                support.append(int(keep.sum()))
            if abs(profile[-1]-edge_height)<.0001:
                break
            edge_height=profile[-1]
        radius_px=min(radius_px,180*outer/np.max(apron))
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
                boundary=np.asarray(row.get('apron_edge',annotation))
                box=tuple(np.round([*np.maximum(0,boundary.min(0)-27),*np.minimum([im.shape[1],im.shape[0]],boundary.max(0)+27)]).astype(int))
                crop=Image.fromarray(im[:,:,::-1]).crop(box)
                crop.save(out/(key_frame+'.png'))
                if row in test_rows:
                    for points,color in ((annotation,(249,194,72)),(pred,(234,107,142)),(corrected,(48,227,208))):
                        cv2.polylines(im,[np.int32(np.round(points*16))],True,color[::-1],1,cv2.LINE_AA,shift=4)
                    Image.fromarray(im[:,:,::-1]).crop(box).save(out/('holdout-'+key_frame+'.png'))
    # Warp only the two fitting photographs, with higher pixel-area views weighted more.
    pixels=480
    y,x=np.indices((pixels,pixels),dtype=float)
    map_x=(x-239.5)*outer/radius_px
    map_y=(239.5-y)*outer/radius_px
    ortho=grid_world(map_x,map_y)
    if apron is not None:
        fraction=radial_fraction(map_x,map_y)
        relief=np.interp(fraction,knots,profile)
        relief=np.where(fraction<0,level,relief*np.clip(1-(fraction-1)*4,0,1))
        ortho+=(relief-level)[:,:,None]*normal
    maps=[]; weights=[]
    for f in training:
        cam=cameras[f]
        uv=(project(ortho,cam)+adjustment(spec['camera'],cam)).astype('f4')
        rgb=cv2.imread(str(camera_root/f'{f:02}.jpg'))[:,:,::-1]
        valid=(uv>=0).all(axis=2)&(uv<[rgb.shape[1]-1,rgb.shape[0]-1]).all(axis=2)
        maps.append(cv2.remap(rgb,uv,None,cv2.INTER_LINEAR))
        weights.append(valid*abs(cv2.contourArea(render_circle(outer,cam).astype('f4'))))
    weights=np.array(weights)
    if not np.all(weights.sum(axis=0)>0):
        raise ValueError('No fitting photograph covers part of the requested orthophoto')
    fused=np.sum(np.asarray(maps)*weights[:,:,:,None],axis=0)/weights.sum(axis=0)[:,:,None]
    Image.fromarray(np.uint8(np.round(fused))).save(out/'surface.png')
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
    if not seam_observed:
        report['lid_diameter_source']='designed_radial_clearance'
    if apron is not None:
        angle=np.arange(len(apron))*(2*np.pi/len(apron))
        xy=apron[:,None]*np.c_[np.cos(angle),np.sin(angle)]
        report.update(apron_radius_m=apron.tolist(),apron_xy_extent_m=np.ptp(xy,axis=0).tolist(),
                      apron_heights_m=profile,apron_height_samples=support)
    save(out/'capture-fit.json',report)
    def landmarks(radius):
        return [[239.5-radius,239.5],[239.5,239.5-radius],[239.5+radius,239.5],[239.5,239.5+radius]]
    save(out/'survey.json',dict(family=spec['family'],credit='PandaSet, Scale AI, Inc. and Hesai Photonics Technology Co., Ltd. CC BY 4.0 and PandaSet Dataset Terms.',
        source_url='https://pandaset.org/',license_url='https://creativecommons.org/licenses/by/4.0/',
        modifications='Annotated, road-plane registered and blended fitting photographs. Geometry below the observed surface and alternative lid states are designed.',
        capture_record='capture-fit.json',capture_crop=label(primary)+'.png',frame=dict(image='surface.png',landmarks_xy=landmarks(radius_px),canonical_radius=radius_px,pavement_patch_xyxy=[8,8,76,76]),
        lid=dict(image='surface.png',landmarks_xy=landmarks(radius_px*inner/outer)),
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
