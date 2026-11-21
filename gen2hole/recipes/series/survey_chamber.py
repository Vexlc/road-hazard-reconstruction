"""Reconstruct a round chamber and pose a rigid photographed lid in local Z-up metres."""
import argparse
import hashlib
import json

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

import chambers
from mesh_io import Mesh, ROOT, intact_pavement, smooth, write_json


CENTER = 239.5
RADIUS = 157.0


def cardinal_homography(size, landmarks, radius=RADIUS):
    """Inverse homography from the canonical disk's four cardinal points into source RGB."""
    source_xy = np.asarray(landmarks, dtype=float)
    if source_xy.shape != (4, 2) or not np.isfinite(source_xy).all():
        raise ValueError("Expected left, top, right and bottom pixel coordinates")
    if (source_xy < 0).any() or (source_xy >= size).any():
        raise ValueError("A landmark lies outside its source image")
    edges = np.roll(source_xy, -1, axis=0) - source_xy
    if (np.cross(edges, np.roll(edges, -1, axis=0)) <= 0).any():
        raise ValueError("The four cyclic landmarks must form a convex quadrilateral")
    base = np.array([[-1,0],[0,-1],[1,0],[0,1]]) * radius + CENTER
    equations, values = [], []
    for (x,y),(u,v) in zip(base,source_xy):
        equations.extend([[x,y,1,0,0,0,-u*x,-u*y], [0,0,0,x,y,1,-v*x,-v*y]])
        values.extend([u,v])
    return np.append(np.linalg.solve(equations,values),1).reshape(3,3)


def rectify(image, landmarks, pixels=800, radius=RADIUS):
    matrix = cardinal_homography(image.size,landmarks,radius)
    y,x = np.mgrid[0:pixels,0:pixels] * (479/(pixels-1))
    projected = matrix @ np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
    if projected[2].min() <= 0 <= projected[2].max():
        raise ValueError("Rectification crosses the projective horizon")
    u,v = (projected[:2] / projected[2]).reshape(2,pixels,pixels)
    rgb = np.asarray(image.convert('RGB'))
    rectified = np.stack([map_coordinates(rgb[:,:,c], [v,u], order=1, mode='reflect') for c in range(3)], axis=-1)
    return rectified, matrix.tolist()


def lid_pose(radius, angle, offset, z):
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a),0,-np.sin(a)],[0,1,0],[np.sin(a),0,np.cos(a)]])
    pivot = np.array([-radius,0,0])
    transform = np.eye(4)
    transform[:3,:3] = rotation
    transform[:3,3] = pivot - rotation @ pivot + [*offset,z]
    return transform


def slide_support(shaft, cover, offset):
    """Find the first vertical contact of the displaced rigid ribs with the frame or road."""
    import bpy  # Registers Blender's bundled mathutils module in standalone Python.
    from mathutils.bvhtree import BVHTree
    points=[]; faces=[]
    for name,vertices,indices,*_ in shaft.parts:
        if name not in ('Road around the aperture','Chipped concrete collar','Cast ring and cover seat'):
            continue
        faces.extend((indices+len(points)).tolist())
        points.extend(vertices.tolist())
    tree=BVHTree.FromPolygons(points,faces,all_triangles=True)
    underside=next(part[1] for part in cover.parts if part[0]=='Lid underside with casting ribs')
    highest=max(p[2] for p in points)+1.
    offsets=[]
    for x,y,z in underside:
        hit,*_=tree.ray_cast((x+offset[0],y+offset[1],highest),(0,0,-1))
        if hit is not None:
            offsets.append(hit.z-z)
    if not offsets:
        raise ValueError('The displaced lid has no contact with the frame or road')
    return float(max(offsets)),len(offsets)


def reconstruct(config):
    refs = ROOT/'references'
    cache = ROOT/'mesh_cache'
    cache.mkdir(exist_ok=True)
    maps, hashes, images, sizes = {}, {}, {}, {}
    patch = None
    frame_radius=float(config['frame'].get('canonical_radius',RADIUS))
    if not np.isfinite(frame_radius) or not 2 <= frame_radius <= CENTER:
        raise ValueError('Invalid canonical frame radius')
    needs_cover = any(not state.get('absent') for state in config['states'])
    for role in (('frame','lid') if needs_cover else ('frame',)):
        path = refs/(role+'.png')
        with Image.open(path) as image:
            sizes[role] = image.size
            images[role], maps[role] = rectify(image,config[role]['landmarks_xy'],radius=frame_radius if role=='frame' else RADIUS)
            if role == 'frame' and 'pavement_patch_xyxy' in config[role]:
                x0,y0,x1,y1 = config[role]['pavement_patch_xyxy']
                if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height and min(x1-x0,y1-y0) >= 32):
                    raise ValueError("The intact road patch must be inside the photograph and at least 32 pixels across")
                patch = np.asarray(image.convert('RGB').crop((x0,y0,x1,y1)))
        hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
        Image.fromarray(images[role]).save(refs/(role+'_ortho.png'))
    write_json(ROOT/'calibration/registration.json',dict(canonical_479px_to_source=maps, center=CENTER, radius=frame_radius, lid_radius=RADIUS))
    gauge = None
    if 'rgbd' in config:
        from chamber_depth import calibrate
        gauge = calibrate(config, sizes['frame'], maps['frame'], refs, CENTER, frame_radius)
        write_json(ROOT/'calibration/depth_measurements.json',gauge)
        config = dict(config,**gauge['measured'])
    measured_pose = None
    observed_states = [s for s in config['states'] if s.get('pose_from_capture')]
    if bool(observed_states) != ('pose_capture' in config):
        raise ValueError("Provide pose_capture and reference it from a state with pose_from_capture")
    if observed_states:
        from chamber_depth import observed_cover
        for state in observed_states:
            if state.get('absent') or 'offset_m' in state or 'angle_deg' in state:
                raise ValueError("An observed lid state takes its pose from its capture")
        capture=config['pose_capture']
        with Image.open(refs/capture['image']) as im:
            matrix=cardinal_homography(im.size,capture['frame_landmarks_xy'])
            measured_pose=observed_cover(config,im.size,matrix,refs,CENTER,RADIUS,float(config['lid_thickness_m']))
        diameter=measured_pose['frame_diameter_m']
        if gauge and abs(diameter-config['frame_diameter_m'])>max(.01,.05*diameter):
            raise ValueError("Frame diameters across the two registered depth captures disagree")
        config=dict(config,lid_diameter_m=measured_pose['lid_diameter_m'])
        if gauge is None:config['frame_diameter_m']=diameter
        write_json(ROOT/'calibration/observed_pose.json',measured_pose)
    frame, aperture, depth = (float(config[k]) for k in ('frame_diameter_m','aperture_diameter_m','shaft_depth_m'))
    if not np.isfinite([frame,aperture,depth]).all() or not (0 < aperture < frame and depth > .060):
        raise ValueError("Expected a positive aperture smaller than the frame and a shaft deeper than its seat")
    lid = float(config['lid_diameter_m']) if needs_cover else 0.
    thick = float(config['lid_thickness_m']) if needs_cover else 0.
    if needs_cover and not (aperture < lid < frame and 0 < thick < depth):
        raise ValueError("Cover must fit its seat and have positive thickness")
    offsets = [np.asarray(measured_pose['face_center_local_m'][:2] if s.get('pose_from_capture') else s.get('offset_m',[0,0]),float)
               for s in config['states']]
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in offsets):
        raise ValueError("offset_m must be a finite XY translation")
    inner_radius = frame_radius*aperture/frame
    outer = [CENTER,CENTER,frame_radius,frame_radius]
    cfg = dict(shape='round', model='missing', width=frame, length=frame, depth=depth, outer=outer,
               inner=[CENTER,CENTER,inner_radius,inner_radius], collar=1.06,
               patch_half_extent_m=max(np.abs(v).max()+lid/2+.15 for v in offsets),
               wall_end=[CENTER,CENTER+.22*inner_radius,.55*inner_radius,.50*inner_radius],
               bottom_photo=[CENTER,CENTER+.23*inner_radius,.33*inner_radius,.32*inner_radius])
    capture=None
    layout={'frame_ortho':cfg}
    if 'capture_record' in config:
        capture=json.loads((ROOT/'calibration'/config['capture_record']).read_text())
        if not np.allclose([frame,lid],[capture['frame_diameter_m'],capture['lid_diameter_m']],rtol=0,atol=1e-8):
            raise ValueError('Survey diameters disagree with the fitted road capture')
        if 'apron_radius_m' in capture:
            cfg.update(apron_radius_m=capture['apron_radius_m'],apron_heights_m=capture['apron_heights_m'],
                       collar=max(capture['apron_radius_m'])/(frame/2))
        face_z=capture['lid_face_above_road_m']
        if not np.isfinite(face_z) or not (-.06 < face_z-thick-.010 < face_z < .05):
            raise ValueError('The photographed seated face must fit the local iron seat profile')
        cfg.update(closed_seat=dict(radius=lid/2,face_z=face_z,thickness=thick),wall_source='designed_lining')
        # The photographed lid hides the entire shaft. Supply a neutral concrete
        # material here so that surface lettering never appears on the shaft wall.
        rng=np.random.default_rng(193501)
        grey=np.clip(rng.normal(76,8,(480,480)),38,112)
        lining=np.uint8(grey[:,:,None]*np.array([1.03,1.,.96]))
        Image.fromarray(lining).save(refs/'designed_lining.png')
        layout['designed_lining']=cfg
    shaft = Mesh()
    grain_source = images['frame']
    if patch is not None:
        height, width = grain_source.shape[:2]
        grain_source = np.pad(patch,((0,height),(0,width),(0,0)),mode='reflect')[:height,:width]
    grain = intact_pavement(grain_source,'frame_ortho',cfg)
    if patch is not None:
        # A projective warp has no observations outside the photo. Keep the cast
        # ring and collar, then feather into measured asphalt without mirroring a second rim.
        y,x = np.indices(images['frame'].shape[:2])*(479/(images['frame'].shape[0]-1))
        radius = np.hypot(x-CENTER,y-CENTER)/frame_radius
        keep_radius=1.13
        if 'apron_radius_m' in cfg:
            azimuth=np.arctan2(CENTER-y,x-CENTER)
            profile=cfg['apron_radius_m']
            keep_radius=np.interp(azimuth,np.arange(len(profile))*2*np.pi/len(profile),profile,period=2*np.pi)/(frame/2)+.16
        opacity = smooth((keep_radius-radius)/.07)[:,:,None]
        h,w = images['frame'].shape[:2]
        frame_image = np.uint8(images['frame']*opacity + grain[:h,:w]*(1-opacity))
        Image.fromarray(frame_image).save(refs/'frame_ortho.png')
    field, fit = chambers.build(shaft,'frame_ortho',cfg,layout,grain)
    cover = Mesh()
    if needs_cover:
        # Texture from the rectified cover, separate from the photo which registers the cast frame.
        lid_outer=[CENTER,CENTER,RADIUS,RADIUS]
        lid_config = dict(cfg,model='survey',outer=lid_outer,lid=dict(type='displaced',ellipse=lid_outer,diameter=lid,thickness=thick))
        chambers.lid_model(cover,'A00',lid_config,images['lid'],frame/2,frame/2)
    extent = field['width']/2
    y,x = np.meshgrid(np.linspace(extent,-extent,field['alpha'].shape[0]),
                      np.linspace(-extent,extent,field['alpha'].shape[1]),indexing='ij')
    for state,offset in zip(config['states'],offsets):
        key = state['id']
        absent = state.get('absent',False)
        angle = float(state.get('angle_deg',0))
        if not np.isfinite(angle) or not 0 <= angle <= 90 or (absent and (angle or offset.any())):
            raise ValueError(f"Invalid cover pose for {key}")
        mesh = Mesh()
        mesh.parts.extend(shaft.parts)
        mesh.materials.update(shaft.materials)
        result = dict(field)
        matrix = None
        contacts = None
        if not absent:
            # The canonical disc's lowest ribs are at Z=0. Its near edge bears on the
            # iron seat when lifted; a sliding disc clears the raised frame before resting on asphalt.
            ring_profile = np.linalg.norm(offset)
            clearance = .006 if 0 < ring_profile < (frame+lid)/2 else 0.
            if state.get('pose_from_capture'):
                angle=measured_pose['angle_degrees']
                matrix=np.asarray(measured_pose['transform'])
            else:
                matrix = lid_pose(lid/2,angle,offset,clearance)
                if capture and offset.any() and angle==0:
                    matrix[2,3],contacts=slide_support(shaft,cover,offset)
                if capture and not offset.any():
                    # Rotate around the seated face's edge in its measured plane.
                    home=np.array([0.,0.,thick+.010])
                    matrix[:3,3]+=np.array([0.,0.,face_z])-matrix[:3,:3]@home
            mesh.materials.update(cover.materials)
            for name,vertices,indices,uv,material,road in cover.parts:
                mesh.parts.append((name,(vertices@matrix[:3,:3].T+matrix[:3,3]).astype('f4'),indices,uv,material,road))
            if offset.any():
                dist = np.hypot(x-offset[0],y-offset[1])
                result['alpha'] = np.maximum(field['alpha'],smooth((lid/2+.11-dist)/.09))
        meta = dict(fit,id=key,label=state['label'],category='missing' if absent else 'raised',family=config['family'],
                    recipe_version='photo-component-v1',scale_kind='designed_metric',source_image='references/frame.png',
                    source_sha256=hashes['frame'],component_source_sha256=hashes,
                    geometry_source='Four planar landmarks per component; round frame and lid; designed shaft and hidden faces',
                    state=state['label'],lid_angle_degrees=None if absent else angle,lid_offset_m=None if absent else offset.tolist(),
                    lid_diameter_m=lid if needs_cover else None,lid_thickness_m=thick if needs_cover else None, lid_transform=None if absent else matrix.tolist(),
                    seed=193501, provenance='calibration/survey.json')
        if contacts is not None:
            meta['lid_support']=dict(method='vertical contact between underside vertices and fixed mesh triangles',hits=contacts,lift_m=float(matrix[2,3]))
        if capture:
            meta.update(scale_kind='calibrated_rgb_lidar_with_designed_completion',
                        measurement_record='calibration/capture-fit.json',
                        metric_observations={k:capture[k] for k in ('frame_diameter_m','lid_diameter_m','lid_face_above_road_m')},
                        pose_source='observed_seated' if not absent and not angle and not offset.any() else 'designed_variant',
                        geometry_source='Photographed outer edge and lid seam on a lidar-fitted road plane. Closed face height from laser returns. Aperture, hidden shaft, relief and underside are design completions.')
            if 'lid_diameter_source' in capture:
                meta['metric_observations'].pop('lid_diameter_m')
                meta['lid_diameter_source']=capture['lid_diameter_source']
            if 'apron_xy_extent_m' in capture:
                meta['metric_observations']['apron_xy_extent_m']=capture['apron_xy_extent_m']
                meta['metric_observations']['apron_radial_heights_m']=capture['apron_heights_m']
                meta['geometry_source']='Visible metal perimeter, polygonal road repair and laser face and repair heights. Inner seam, aperture, shaft, underside and alternative states use designed completion.'
        if measured_pose:
            meta.update(scale_kind='registered_depth_with_designed_completion',measurement_record='calibration/observed_pose.json',
                        metric_observations=dict(frame_diameter_m=frame,lid_diameter_m=lid))
        if state.get('pose_from_capture'):
            meta.update(pose_source='registered_depth',pose_source_kind=measured_pose['source_kind'],
                        pose_observations='calibration/observed_pose.json',lid_face_center_m=measured_pose['face_center_local_m'],
                        lid_offset_reference='observed face centre relative to frame centre, in road XY metres')
        if gauge:
            meta.update(scale_kind='registered_depth_with_designed_completion',metric_observations=dict(meta.get('metric_observations',{}),**gauge['measured']),
                        depth_source_kind=gauge['source_kind'],measurement_record='calibration/depth_measurements.json')
            if 'shaft_depth_m' in gauge['measured']:
                meta['depth_measured_m'] = gauge['measured']['shaft_depth_m']
                del meta['depth_design_m']
        Image.fromarray(grain).save(cache/(key+'_ground.jpg'),quality=96,subsampling=0)
        mesh.save(ROOT/'assets'/meta['category']/key,cache,meta,result)
        print(key,meta['label'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    reconstruct(json.loads((ROOT/'calibration/survey.json').read_text()))
