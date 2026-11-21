"""Measure the diameter of a circular road fixture and the depth of a visible shaft floor."""
import hashlib

import numpy as np
from PIL import Image


def registered_depth(path, scale):
    if path.suffix.lower()=='.npy':
        raw = np.load(path,allow_pickle=False)
    else:
        with Image.open(path) as im:
            raw = np.asarray(im)
    if raw.ndim!=2 or raw.dtype.kind not in 'iuf' or not np.isfinite(scale) or scale<=0:
        raise ValueError("Depth must be a single optical-Z channel with a positive metres-per-unit scale")
    return np.asarray(raw,dtype='f8')*scale


def rays(xy, intrinsics):
    fx,fy,cx,cy = intrinsics
    x,y = np.asarray(xy,dtype=float).T
    return np.column_stack(((x-cx)/fx,(y-cy)/fy,np.ones(len(x))))


def fit_plane(points, tolerance):
    if len(points)<100:
        raise ValueError("Fewer than 100 valid planar depth samples")
    rng = np.random.default_rng(1949)
    sample = points[rng.choice(len(points),min(4096,len(points)),replace=False)]
    best = 0
    for _ in range(128):
        a,b,c = sample[rng.choice(len(sample),3,replace=False)]
        n = np.cross(b-a,c-a)
        length = np.linalg.norm(n)
        if length < 1e-8:
            continue
        n /= length
        count = np.count_nonzero(abs((sample-a)@n)<tolerance)
        if count>best:
            best,normal,offset=count,n,n@a
    if best < .6*len(sample):
        raise ValueError("Insufficient consensus around one surface plane")
    for _ in range(3):
        keep=abs(points@normal-offset)<tolerance
        inliers=points[keep]
        center=inliers.mean(axis=0)
        normal=np.linalg.svd(inliers-center,full_matrices=False)[2][-1]
        if normal@center<0:normal=-normal
        offset=normal@center
    residuals=points@normal-offset
    keep=abs(residuals)<tolerance
    return normal,offset,dict(inlier_fraction=float(keep.mean()),inliers=int(keep.sum()),samples=len(points),
                              rms_m=float(np.sqrt(np.mean(residuals[keep]**2))))


def cardinal_points(landmarks, intrinsics, normal, offset, size):
    xy=np.asarray(landmarks,dtype=float)
    if xy.shape!=(4,2) or not np.isfinite(xy).all() or (xy<0).any() or (xy>=size).any():
        raise ValueError("Four in-frame cardinal rim landmarks are required")
    lines=rays(xy,intrinsics)
    denominator=lines@normal
    if denominator.min() <= .025:
        raise ValueError("The photographed rim lies near or behind the road-plane horizon")
    return lines*(offset/denominator[:,None])


def diameter(landmarks, intrinsics, normal, offset, size):
    points=cardinal_points(landmarks,intrinsics,normal,offset,size)
    axes=points[2:]-points[:2]
    spans=np.linalg.norm(axes,axis=1)
    value=float(spans.mean())
    center_error=float(np.linalg.norm((points[0]+points[2]-points[1]-points[3])/2))
    if value<.03 or abs(np.diff(spans)[0])>.18*value or center_error>.12*value:
        raise ValueError("The landmarks do not support two diameters of a round planar rim")
    return value,dict(opposite_spans_m=spans.tolist(),diameter_midpoint_separation_m=center_error)


def calibrate(spec, rgb_size, matrix, ref_dir, center, radius):
    survey = spec['rgbd']
    depth_path=ref_dir/survey['depth']
    depth=registered_depth(depth_path,float(survey['metres_per_unit']))
    if depth.shape!=(rgb_size[1],rgb_size[0]):
        raise ValueError("The depth map must be registered to the original frame photograph")
    intrinsics=np.asarray(survey['intrinsics_px'],float)
    if intrinsics.shape!=(4,) or not np.isfinite(intrinsics).all() or min(intrinsics[:2])<=0:
        raise ValueError("intrinsics_px must contain positive fx, fy and finite cx, cy")
    tol=float(survey.get('plane_tolerance_m',.015))
    if not np.isfinite(tol) or tol<=0:
        raise ValueError("plane_tolerance_m must be positive")
    y,x=np.nonzero(np.isfinite(depth)&(depth>0))
    image_xy=np.column_stack((x,y))
    unit=np.column_stack((image_xy,np.ones(len(x))))@np.linalg.inv(matrix).T
    with np.errstate(divide='ignore',invalid='ignore'):
        ring_distance=np.linalg.norm(unit[:,:2]/unit[:,2:]-center,axis=1)/radius
    select=(ring_distance>=1.12)&(ring_distance<=1.65)
    ix=np.flatnonzero(select)
    if len(ix)>60000:
        ix=np.random.default_rng(230).choice(ix,60000,replace=False)
    points=rays(image_xy[ix],intrinsics)*depth[y[ix],x[ix],None]
    n,d,quality=fit_plane(points,tol)
    measured,observations={},{}
    for field,marks in [('frame_diameter_m','landmarks_xy'),('aperture_diameter_m','aperture_landmarks_xy')]:
        if marks in spec['frame']:
            measured[field],observations[field]=diameter(spec['frame'][marks],intrinsics,n,d,rgb_size)
    hashes={'depth':hashlib.sha256(depth_path.read_bytes()).hexdigest()}
    if survey.get('bottom_mask'):
        mask_path=ref_dir/survey['bottom_mask']
        with Image.open(mask_path) as im:mask=np.asarray(im.convert('L'))>0
        if mask.shape!=depth.shape:
            raise ValueError("Bottom mask shape differs from registered depth")
        # Reject masks spilling onto adjacent asphalt. Masked visible returns may
        # project near the rim in oblique views, so apply the outer (not inner) ring.
        sel=mask[y,x]&(ring_distance<1.)
        if sel.sum()<24 or sel.sum()<.4*mask.sum():
            raise ValueError("Too few valid registered bottom returns inside the identified rim")
        z=(rays(image_xy[sel],intrinsics)*depth[y[sel],x[sel],None])@n-d
        value=float(np.median(z))
        if value<=max(.060,3*quality['rms_m']):
            raise ValueError("The designated shaft bottom is not resolved below the road plane")
        measured['shaft_depth_m']=value
        observations['shaft_depth_m']=dict(samples=len(z),coverage=float(sel.sum()/mask.sum()),
                                            p10_p90_m=np.percentile(z,[10,90]).tolist())
        hashes['bottom_mask']=hashlib.sha256(mask_path.read_bytes()).hexdigest()
    return dict(source_kind=survey['source_kind'],measured=measured,observations=observations,
                road_plane=dict(normal_camera=n.tolist(),offset_m=float(d),**quality),
                registered_inputs_sha256=hashes,intrinsics_px=intrinsics.tolist())

def component_basis(points):
    # X points from left to right; Y from bottom to top. These are the component
    # axes marked by the operator, including azimuth about the local road normal.
    x=points[2]-points[0]
    y=points[1]-points[3]
    x=x/np.linalg.norm(x)
    y=y/np.linalg.norm(y)
    if abs(x@y)>.20:
        raise ValueError("Opposing cardinal landmarks do not form perpendicular diameters")
    y-=x*(x@y)
    y/=np.linalg.norm(y)
    return np.column_stack([x,y,np.cross(x,y)])


def observed_cover(spec, original_size, matrix, refs, center, radius, thickness):
    # A separate capture can observe the deployed cover from a different camera.
    # Its own rim datum re-registers that camera to the same component axes.
    capture=spec['pose_capture']
    rim=capture['frame_landmarks_xy']
    road=calibrate(dict(rgbd=capture,frame=dict(landmarks_xy=rim)),original_size,matrix,refs,center,radius)
    n,d=road['road_plane']['normal_camera'],road['road_plane']['offset_m']
    intr=road['intrinsics_px']
    frame=cardinal_points(rim,intr,n,d,original_size)
    frame_axes=component_basis(frame)
    if frame_axes[:,2]@np.asarray(n)>-.98:
        raise ValueError("The road-frame landmark order must yield an upward Z axis")
    z=registered_depth(refs/capture['depth'],capture['metres_per_unit'])
    mask_file=refs/capture['lid_mask']
    with Image.open(mask_file) as im:mask=np.asarray(im.convert('L'))>0
    if mask.shape!=z.shape:raise ValueError("Cover mask and registered optical Z differ in shape")
    y,x=np.where(mask&np.isfinite(z)&(z>0))
    valid_coverage=float(len(x)/max(1,mask.sum()))
    if not mask.any() or valid_coverage<.4:raise ValueError("Insufficient visible lid depth coverage")
    if len(x)>60000:
        ix=np.random.default_rng(230).choice(len(x),60000,replace=False)
        x,y=x[ix],y[ix]
    cloud=rays(np.column_stack((x,y)),intr)*z[y,x,None]
    tolerance=float(capture.get('plane_tolerance_m',.015))
    normal,offset,quality=fit_plane(cloud,tolerance)
    marks=capture['lid_landmarks_xy']
    points=cardinal_points(marks,intr,normal,offset,original_size)
    lid_diameter,diameters=diameter(marks,intr,normal,offset,original_size)
    lid_axes=component_basis(points)
    if lid_axes[:,2]@normal>-.98:
        raise ValueError("The annotated surface must show the tread face with its four ordered cardinal points")
    planar=cloud[abs(cloud@normal-offset)<tolerance]
    local_xy=(planar-points.mean(axis=0))@lid_axes[:,:2]
    inside=float(np.mean(np.linalg.norm(local_xy,axis=1)<=lid_diameter*.56))
    if inside<.80:raise ValueError("Depth mask and annotated cover perimeter disagree")
    quality.update(valid_mask_fraction=valid_coverage,inliers_inside_perimeter_fraction=inside)
    rotation=frame_axes.T@lid_axes
    lid_center=frame_axes.T@(points.mean(axis=0)-frame.mean(axis=0))
    t=np.eye(4)
    t[:3,:3]=rotation
    t[:3,3]=lid_center-rotation@np.array([0.,0.,thickness+.010])
    angle=float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1))))
    if angle>95:raise ValueError("Observed cover face is beyond this recipe's near-vertical range")
    hashes=road['registered_inputs_sha256']|{'lid_mask':hashlib.sha256(mask_file.read_bytes()).hexdigest(),
            'image':hashlib.sha256((refs/capture['image']).read_bytes()).hexdigest()}
    frame_to_camera=np.eye(4)
    frame_to_camera[:3,:3]=frame_axes
    frame_to_camera[:3,3]=frame.mean(axis=0)
    return dict(frame_to_camera=frame_to_camera.tolist(),intrinsics_px=intr,camera_size=list(original_size),
                transform=t.tolist(),angle_degrees=angle,face_center_local_m=lid_center.tolist(),
                lid_diameter_m=lid_diameter,diameter_fit=diameters,face_plane=quality,road_plane=road['road_plane'],
                frame_diameter_m=road['measured']['frame_diameter_m'],source_kind=capture['source_kind'],
                registered_inputs_sha256=hashes)
