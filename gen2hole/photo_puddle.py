"""Build a portable pool from a photographed shoreline, without baking its reflection into the bed."""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from fuse_pit import road_texture
from oblique_pits import digest, grid_world, polygon_mask, project, put


def observed_pool(rgb, prior, rim, radius):
    if not np.isfinite(radius) or radius < 1:
        raise ValueError('Expected a positive edge search radius in source pixels')
    inner, outer = distance_transform_edt(prior), distance_transform_edt(~prior)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype('f4')
    core = inner > radius
    exterior = (outer > radius) & (outer <= 2*radius) & rim
    if not core.any() or not exterior.any() or np.any(core & ~rim):
        raise ValueError('Expected wet and dry seeds inside the photographed bank')
    wet, dry = np.median(grey[core]), np.median(grey[exterior])
    if wet-dry < 12:
        raise ValueError('The bright-pool edge is not separable from its bank')
    membership = np.clip((grey-dry)/(wet-dry),0,1)
    membership[core] = 1
    membership[(outer > radius) | ~rim] = 0
    count, labels = cv2.connectedComponents((membership >= .5).astype('u1'))
    if count < 2:
        raise ValueError('No connected pool')
    pool = labels == np.bincount(labels[core]).argmax()
    membership[(labels != 0) & ~pool] = 0
    return membership, pool


def projected_area(mask, camera, normal, offset):
    contours, _ = cv2.findContours(mask.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pixels = max(contours,key=cv2.contourArea)[:,0]
    rays = np.column_stack(((pixels-camera[:2,2])/camera.diagonal()[:2],np.ones(len(pixels))))
    points = rays*(-offset/(rays@normal))[:,None]
    return float(.5*abs(np.cross(points,np.roll(points,-1,axis=0)).sum(0)@normal))


def source_limits(pool, membership, rim, planes):
    rows, cols = np.where(pool)
    limits, areas = {}, []
    for name, model in planes.items():
        camera, normal, offset = np.array(model['K']), np.array(model['normal']), model['offset_m']
        rays = np.column_stack(((cols-camera[0,2])/camera[0,0],(rows-camera[1,2])/camera[1,1],np.ones(len(cols))))
        def contained(drop):
            local = rays*((-offset-drop)/(rays@normal))[:,None]+drop*normal
            col,row = np.rint(project(local,camera)).astype(int).T
            valid = (row>=0)&(col>=0)&(row<rim.shape[0])&(col<rim.shape[1])
            return rim[row[valid],col[valid]].sum() >= .999*len(row)
        low, high = 0., .01
        while contained(high) and high < .5:
            high *= 2
        for _ in range(20):
            mid = (low+high)/2
            if contained(mid):
                low = mid
            else:
                high = mid
        limits[name] = low
        for threshold in (.25,.75):
            for drop in (0,low):
                areas.append(projected_area(membership>=threshold,camera,normal,offset+drop))
    return limits, [min(areas),max(areas)]


def basin_shape(water, coverage):
    if water[[0,-1]].any() or water[:,[0,-1]].any():
        raise ValueError('Shoreline requires an intact border around the reconstruction')
    rows, cols = np.where(water)
    index = np.full(water.shape,-1,'i4')
    ids = index[rows,cols] = np.arange(len(rows))
    diagonal = np.zeros(len(rows))
    rr, cc, vv = [], [], []
    for dr,dc in ((0,-1),(0,1),(-1,0),(1,0)):
        neighbor = index[rows+dr,cols+dc]
        inside = neighbor>=0
        rr.extend(ids[inside]); cc.extend(neighbor[inside]); vv.extend(-np.ones(inside.sum()))
        diagonal += 1
        here = coverage[rows[~inside],cols[~inside]]
        there = coverage[rows[~inside]+dr,cols[~inside]+dc]
        fraction = np.clip((here-.5)/np.maximum(here-there,1e-5),.10,1)
        diagonal[~inside] += 1/fraction-1
    system = coo_matrix((np.r_[vv,diagonal],(np.r_[rr,ids],np.r_[cc,ids]))).tocsr()
    depression = np.zeros(water.shape,'f4')
    solution = spsolve(system,np.ones(len(rows)))
    depression[water] = solution/solution.max()
    return depression


def prepare(bundle, spec_path, profile_path, site, road_config):
    profile = json.loads(profile_path.read_text())
    spec = json.loads(spec_path.read_text())
    key = spec['id']
    if key != profile['id'] or spec.get('reference_kind') != 'photograph':
        raise ValueError('Expected a matching photograph and shoreline profile')
    depths = np.asarray(profile['depth_hypotheses_m'])
    if depths.shape != (3,) or not np.isfinite(depths).all() or min(depths)<=0 or min(np.diff(depths))<=0:
        raise ValueError('Expected three increasing positive underwater depth hypotheses')
    refs, cal = bundle/'references', bundle/'calibration'
    meta = json.loads((cal/(key+'.json')).read_text())
    capture = json.loads((cal/(key+'_capture.json')).read_text())
    camera = np.float32(capture['road_planes']['direct']['K'])
    origin = np.float32(capture['local_origin_in_camera'])
    axes = np.float32(capture['local_axes_in_camera'])
    with np.load(cal/(key+'.npz')) as archive:
        field = dict(archive)
    span = float(field['width'])
    footprint = field['defect_mask']
    size = len(footprint)
    pitch = span/(size-1)
    rgb = np.asarray(Image.open(refs/(key+'.jpg')).convert('RGB'))
    prior = polygon_mask(rgb.shape[:2], [profile['waterline']])
    source_bank = polygon_mask(rgb.shape[:2],[spec['rim']])
    membership, source_water = observed_pool(rgb,prior,source_bank,profile['shore_search_px'])
    limits, area_range = source_limits(source_water,membership,source_bank,capture['road_planes'])
    drop = .5*min(limits.values())
    def sample(array, height, size, interp):
        surface = grid_world(size, span, origin, axes) + height*axes[:,2]
        return cv2.remap(array, project(surface,camera), None, interp)
    coverage = sample(membership,-drop,size,cv2.INTER_LINEAR)
    projected = coverage > .5
    water = projected & footprint
    if not projected.any() or water.sum()/projected.sum() < .95:
        raise ValueError('The projected waterline is not contained by the bank')
    shore = distance_transform_edt(~water)*pitch
    rim = np.maximum(0,distance_transform_edt(footprint)-1)*pitch
    bank = rim / np.maximum(rim+shore, pitch)
    bowl = basin_shape(water,coverage)
    height = np.where(water,-drop-depths[1]*bowl, -drop*bank).astype('f4')
    seed = np.unravel_index(height.argmin(),height.shape)
    nominal_liters = np.maximum(-drop-height.astype('f8'),0).sum()*pitch**2*1000
    up, provenance = [0,0,1], 'Source road assumed level; no IMU or water-depth measurement'
    if site:
        from downstream.place import SLOTS
        plan = Path(json.loads(Path(road_config).read_text())['plan'])
        placement = json.loads(plan.read_text())['slots_by_site'][site][SLOTS[site][0]]
        up, provenance = placement['basis_columns'][2], f"Road plan {digest(plan)}, slot {placement['slot']}"
    info = dict(site=site or 'source', up_local=up, up_source=provenance, input_liters=float(nominal_liters), bed_optics='wet-asphalt',
                seed_uv=(np.asarray(seed[::-1])/(size-1)).tolist())
    field.update(height=height, photo_shoreline=water, normalized_submerged_bed=bowl)
    field.pop('prediction_support',None)
    np.savez_compressed(cal/(key+'.npz'), **field)
    resolution = 1025
    shape = (resolution,resolution)
    flat = sample(rgb,0,resolution,cv2.INTER_LINEAR)
    road_pixels = polygon_mask(rgb.shape[:2],spec['road_polygons'])
    road = sample(road_pixels.astype('u1'),0,resolution,cv2.INTER_NEAREST)>0
    distance = abs(flat.astype('f4')-np.median(flat[road],axis=0)).max(axis=2)
    clean = road & (distance<np.percentile(distance[road],87))
    bed = cv2.resize(footprint.astype('f4'),shape)
    donor = road_texture(flat, clean, bed>.5)
    forbidden = source_water | polygon_mask(rgb.shape[:2],profile.get('ignore_color',[]))
    valid = source_bank & ~cv2.dilate(forbidden.astype('u1'),np.ones((5,5),'u1')).astype(bool)
    valid &= rgb.max(axis=2)<np.percentile(rgb[road_pixels].max(axis=1),90)
    surface_h = cv2.resize(height,shape)[:,:,None]
    bank_rgb = sample(rgb,surface_h,resolution,cv2.INTER_LINEAR)
    blend = sample(np.minimum(distance_transform_edt(valid)/3,1).astype('f4'),surface_h,resolution,cv2.INTER_LINEAR)
    moisture = cv2.resize(np.float32(bank),shape)
    blend *= .65*moisture*(1-cv2.resize(water.astype('f4'),shape))
    grain = donor.astype('f4')-cv2.GaussianBlur(donor.astype('f4'),(0,0),3)
    intrinsic = (donor+np.clip(1.5*grain,-10,10)*bed[:,:,None])*(1-.14*moisture[:,:,None])
    texture = np.uint8(np.clip(intrinsic*(1-blend[:,:,None])+bank_rgb*blend[:,:,None],0,255))
    Image.fromarray(texture).save(refs/(key+'_ortho.png'))
    Image.fromarray(donor).save(refs/(key+'_pavement.jpg'),quality=96)
    capture['pre_water_baseline'] = {name:capture.pop(name) for name in (
        'direct_visible_color_fraction','footprint_prediction_support_fraction','source_valid_pit_pixels')}
    capture.update(shoreline_profile_sha256=digest(profile_path),submerged_bed_measured=False,
                   source_rim_clearance_max_m=limits,assumed_freeboard_m=drop,
                   projected_area_sensitivity_m2=area_range,underwater_depth_hypotheses_m=depths.tolist(),
                   raw_depth_role='Road plane and footprint scale only; does not measure through the water',
                   reference_water_liters_model=float(nominal_liters),waterline_grid_coverage=float(water.sum()/projected.sum()))
    put(cal/(key+'_capture.json'),capture)
    meta.update(recipe_version='oblique-photo-shoreline-v2',state='standing_water',water=info,
                geometry_source='Photographed bank and waterline on an estimated road plane; submerged bed is designed',
                depth_p95_m=float(np.percentile(-height[footprint],95)),water_optics='refracted-bed',
                relief_profile='calibration/'+key+'-water.json')
    put(cal/(key+'.json'),meta)
    shutil.copyfile(profile_path,cal/(key+'-water.json'))
