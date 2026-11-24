"""Reconstruct a road-plane-relative surface from a photograph and cached MoGe predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.spatial import cKDTree

from fuse_pit import road_texture
from gen2hole import ROOT, run_step


def put(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygon_mask(shape, polygons):
    out = np.zeros(shape, 'u1')
    for poly in polygons:
        cv2.fillPoly(out, [np.int32(poly)], 1)
    return out.astype(bool)


def camera_rays(shape, K):
    y, x = np.indices(shape, dtype='f4')
    return np.stack([(x-K[0, 2])/K[0, 0], (y-K[1, 2])/K[1, 1], np.ones(shape, 'f4')], -1)


def plane_fit(depth, rays, road, withheld):
    xyz = (depth[road & ~withheld, None] * rays[road & ~withheld])[::4]
    good = np.ones(len(xyz), bool)
    for _ in range(7):
        centre = xyz[good].mean(0)
        _, _, vt = np.linalg.svd(xyz[good]-centre, full_matrices=False)
        normal = vt[-1]
        if centre @ normal > 0:
            normal = -normal
        offset = float(-centre @ normal)
        residual = xyz @ normal + offset
        good = np.abs(residual) <= np.quantile(np.abs(residual), .82)
    heights = depth * (rays @ normal) + offset
    return normal, offset, heights


def grid_world(size, span, origin, axes):
    t = np.linspace(-span/2, span/2, size, dtype='f4')
    x, y = np.meshgrid(t, t[::-1])
    return origin + x[..., None]*axes[:, 0] + y[..., None]*axes[:, 1]


def project(world, K):
    return np.float32(world[..., :2] / world[..., 2:] * K.diagonal()[:2] + K[:2, 2])


def spall_relief(height, footprint, pitch, rgb, origin, axes, span, camera, profile):
    surface = grid_world(len(height), span, origin, axes)
    photo = cv2.remap(rgb, project(surface, camera), None, cv2.INTER_LINEAR)
    luminance = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype('f4')
    support = footprint.astype('f4')
    distance = np.maximum(distance_transform_edt(footprint) - 1, 0) * pitch
    edge = np.clip(distance / profile['lip_width_m'], 0, 1)
    bowl = np.clip(distance / profile['basin_width_m'], 0, 1)
    sigma = profile['aggregate_radius_m'] / pitch
    surround = cv2.GaussianBlur(luminance * support, (0, 0), sigma) / np.maximum(
        cv2.GaussianBlur(support, (0, 0), sigma), 1e-5)
    dark = np.clip((surround - luminance) / profile['dark_contrast_srgb'], 0, 1)
    depth = profile['bed_m'] * bowl + profile['chip_m'] * dark + np.maximum(-height, 0)
    return np.float32(-edge * np.minimum(depth, profile['cap_m']))


def build_case(spec_path, inputs, bundle, spall_profile=None):
    spec = json.loads(spec_path.read_text())
    key = spec['id']
    refs, cal = bundle/'references', bundle/'calibration'
    src = inputs/(key+'.jpg')
    cache = inputs/(key+'-depth.npz')
    rgb = np.array(ImageOps.exif_transpose(Image.open(src)).convert('RGB'))
    shape = rgb.shape[:2]
    if spec['image_size'] != list(shape[::-1]):
        raise ValueError(f'{key}: annotation coordinates do not match the photo dimensions')
    if 'source_sha256' in spec and spec['source_sha256'] != digest(src):
        raise ValueError(f'{key}: annotated source hash does not match the photo')
    mask = polygon_mask(shape, [spec['rim']])
    excluded = polygon_mask(shape, spec.get('exclude_polygons', []))
    overlay = np.zeros(shape, 'u1')
    for x0, y0, x1, y1, width in spec.get('overlay_segments', []):
        cv2.line(overlay, (x0, y0), (x1, y1), 1, width)
    forbidden = excluded | overlay.astype(bool)
    y, x = np.indices(shape)
    held = (x//16 + 3*(y//16)) % 5 == 0
    road = polygon_mask(shape, spec['road_polygons']) & ~forbidden & ~mask
    with np.load(cache) as data:
        if digest(src) != str(data['image_sha256']) or digest(spec_path) != str(data['spec_sha256']):
            raise ValueError(f'{key}: depth cache belongs to a different photo or outline')
        predictions, plane_records = {}, {}
        for side in ('direct', 'flip'):
            K = data['K_'+side]
            depth = data['depth_'+side]
            rays = camera_rays(shape, K)
            valid = road & np.isfinite(depth) & (depth > 0)
            n, offset, h = plane_fit(depth, rays, valid, held)
            plane_records[side] = dict(K=K.tolist(), normal=n.tolist(), offset_m=offset,
                road_reserved_model_residual_abs_p95_m=float(np.percentile(abs(h[held & valid]), 95)))
            predictions[side] = h
        weights_hash = str(data['weights_sha256'])
    model = plane_records['direct']
    K, n, offset = np.float32(model['K']), np.float32(model['normal']), model['offset_m']
    rays = camera_rays(shape, K)
    both = np.isfinite(predictions['direct']) & np.isfinite(predictions['flip'])
    observed = mask & both & ~forbidden
    h = np.minimum(.5*(predictions['direct']+predictions['flip']), 0)
    # A ray through an inferred below-road point intersects the plane closer to the camera.
    # Anchor the observed lip on the plane and interpolate only inside its footprint.
    contour = np.int32(spec['rim'])
    rim_rays = rays[contour[:, 1], contour[:, 0]]
    rim_xyz = -(offset/(rim_rays@n))[:, None] * rim_rays
    ex = np.float32([1, 0, 0]) - n*n[0]
    ex /= np.linalg.norm(ex)
    axes = np.column_stack([ex, np.cross(n, ex), n])
    plane_origin = -n*offset
    rim_xy = ((rim_xyz - plane_origin)@axes)[:, :2]
    design = None
    if 'design_long_axis_m' in spec:
        target = float(spec['design_long_axis_m'])
        inferred = max(cv2.minAreaRect(np.ascontiguousarray(rim_xy,dtype='f4'))[1])
        if spec.get('reference_kind') != 'generated' or not np.isfinite([target,inferred]).all() or min(target,inferred)<=0:
            raise ValueError('A positive design extent requires a generated reference with a valid footprint')
        factor = target/inferred
        design = dict(long_axis_m=target, original_model_long_axis_m=inferred, depth_multiplier=factor)
        rim_xy *= factor
        plane_origin *= factor
        h *= factor
        for side, plane in plane_records.items():
            plane['offset_m'] *= factor
            plane['road_reserved_model_residual_abs_p95_m'] *= factor
            predictions[side] *= factor
        offset = model['offset_m']
    centre = (rim_xy.max(0)+rim_xy.min(0))/2
    origin = plane_origin + axes[:, :2]@centre
    rim_xy -= centre
    span_xy = np.ptp(rim_xy, axis=0)
    border = np.clip(span_xy.max()*.09, .12, .55)
    span = float(span_xy.max()+2*border)
    size = 321
    pitch = span/(size-1)
    rim_grid = np.rint((rim_xy*[1, -1]+span/2)/pitch).astype('i4')
    footprint = polygon_mask((size, size), [rim_grid])
    true_rays = rays[observed]
    samples = true_rays * ((h[observed]-offset)/(true_rays@n))[:, None]
    local = (samples-origin)@axes
    interior = cv2.distanceTransform(mask.astype('u1'), cv2.DIST_L2, 5)[observed]
    # Exclude unreliable pixels at the very edge. Add road-height anchors between rim vertices.
    local = local[interior > 2]
    anchors = []
    for a, b in zip(rim_xy, np.roll(rim_xy,-1,axis=0)):
        anchors.extend(np.linspace(a, b, max(2, int(np.linalg.norm(a-b)/(.45*pitch))), endpoint=False))
    anchors = np.asarray(anchors)
    pixels = np.column_stack(np.nonzero(footprint))[:, ::-1]
    targets = (pixels*pitch-span/2)*[1,-1]
    nearest = cKDTree(local[:, :2]).query(targets)[0]
    xyz = np.vstack([local, np.column_stack([anchors, np.zeros(len(anchors))])])
    d, ix = cKDTree(xyz[:, :2]).query(targets,k=8)
    weights = 1/np.maximum(d,.10*pitch)**2
    height = np.zeros((size, size), 'f4')
    height[footprint] = (xyz[ix, 2]*weights).sum(1)/weights.sum(1)
    height = np.minimum(median_filter(height, 3), 0)*np.clip(distance_transform_edt(footprint)/2, 0, 1)
    if spall_profile:
        profile = json.loads(spall_profile.read_text())
        params = [profile[name] for name in ('lip_width_m','basin_width_m','aggregate_radius_m',
                                             'dark_contrast_srgb','bed_m','chip_m','cap_m')]
        if profile['id'] != key or not np.isfinite(params).all() or min(params) <= 0:
            raise ValueError('Expected a matching shallow-spall profile with positive finite parameters')
        np.savez_compressed(cal/(key+'_monocular.npz'), height=height, defect_mask=footprint, width=span)
        raw_p95 = float(np.percentile(-height[footprint], 95))
        height = spall_relief(height, footprint, pitch, rgb, origin, axes, span, K, profile)
        shutil.copyfile(spall_profile, cal/(key+'-spall.json'))
    outside = distance_transform_edt(~footprint)*pitch
    alpha = np.float32(np.clip(1-outside/border, 0, 1))
    tex_size = 1025
    surface = grid_world(tex_size, span, origin, axes)
    tex_height = cv2.resize(height, (tex_size, tex_size), interpolation=cv2.INTER_LINEAR)
    uv = project(surface + tex_height[...,None]*n, K)
    def remap(arr, interpolation=cv2.INTER_LINEAR):
        return cv2.remap(arr, uv, None, interpolation, borderMode=cv2.BORDER_REPLICATE)
    # Quilt photographed road in ground coordinates. Reject paint and highlights before
    # drawing pavement patches, so lane markings cannot repeat in the inspection apron.
    texture = remap(rgb)
    observed_road = remap(road.astype('u1'), cv2.INTER_NEAREST) > 0
    chroma = texture.astype('f4')
    road_median = np.median(chroma[observed_road], axis=0)
    variation = np.max(np.abs(chroma-road_median), axis=2)
    clean_road = observed_road & (variation <= np.percentile(variation[observed_road], 85))
    tex_footprint = cv2.resize(footprint.astype('u1'),(tex_size,tex_size),interpolation=cv2.INTER_NEAREST)>0
    apron = road_texture(texture, clean_road, tex_footprint)
    donor = apron[tex_size//4:tex_size//4*3,tex_size//4:tex_size//4*3].copy()
    pit_safe = observed & (cv2.distanceTransform((~forbidden).astype('u1'),cv2.DIST_L2,5)>2)
    _, idx = distance_transform_edt(~pit_safe, return_indices=True)
    pit_color = remap(rgb[tuple(idx)])
    # Blend reliable local road gradually; single pixels at the lip keep their photographed colour.
    road_weight = cv2.GaussianBlur(clean_road.astype('f4'),(0,0),4)
    road_weight *= (variation < 35)
    apron = apron * (1-road_weight[...,None]) + texture*road_weight[...,None]
    tex_mask = cv2.resize(footprint.astype('f4'), (tex_size,tex_size), interpolation=cv2.INTER_LINEAR)[...,None]
    texture = np.uint8(np.clip(pit_color*tex_mask+apron*(1-tex_mask),0,255))
    grid_supported = np.zeros_like(height, dtype='u1')
    grid_supported[footprint] = nearest < 2*pitch
    direct_color = remap(pit_safe.astype('u1'), cv2.INTER_NEAREST)
    texture_support = (direct_color > 0) & (tex_mask[...,0]>.5)
    Image.fromarray(texture).save(refs/(key+'_ortho.png'))
    Image.fromarray(donor).save(refs/(key+'_pavement.jpg'), quality=97)
    shutil.copyfile(src, refs/(key+'.jpg'))
    shutil.copyfile(cache, cal/(key+'-depth.npz'))
    shutil.copyfile(spec_path, cal/(key+'-outline.json'))
    np.savez_compressed(cal/(key+'.npz'), height=np.float32(height), width=span, length=span, alpha=alpha,
                        defect_mask=footprint, water_mask=np.zeros_like(footprint), prediction_support=grid_supported)
    rect = cv2.minAreaRect(rim_xy.astype('f4'))[1]
    stats = dict(road_planes=plane_records, local_origin_in_camera=origin.tolist(), local_axes_in_camera=axes.tolist(),
                 image_sha256=digest(src), weights_sha256=weights_hash, network='MoGe-2 ViT-L normal',
                 direct_flip_abs_p95_m=float(np.percentile(abs(predictions['direct'][observed]-predictions['flip'][observed]),95)),
                 footprint_prediction_support_fraction=float(grid_supported[footprint].mean()),
                 direct_visible_color_fraction=float(texture_support.sum()/(tex_mask[...,0]>.5).sum()),
                 source_valid_pit_pixels=int(observed.sum()), pavement_texture='Quilt of radiometrically filtered intact-road samples in plane coordinates')
    if design:
        stats['design_scale'] = design
    if spall_profile:
        stats['relief_prior'] = dict(profile=profile, monocular_p95_m=raw_p95, measured=False)
    put(cal/(key+'_capture.json'), stats)
    meta = dict(id=key,label=spec['label'],category='pothole',family=key,state='estimated',seed=307101+int(key[1:]),
        recipe_version='oblique-photo-depth-v1',source_image='references/'+src.name,source_sha256=digest(src),
        source=spec['source'],geometry_source='Plane-aligned original/flip MoGe-2 single-photo estimates, with an annotated rim',
        scale_kind='Uncalibrated monocular metric prior', measurement_record='calibration/'+key+'_capture.json',
        width_m=float(span_xy[0]),length_m=float(span_xy[1]),oriented_extent_m=sorted(map(float,rect),reverse=True),
        outline_area_m2=float(cv2.contourArea(np.ascontiguousarray(rim_xy, dtype="f4"))),depth_p95_m=float(np.percentile(-height[footprint],95)),
        source_grid_pitch_m=float(pitch))
    if design:
        meta.update(scale_kind='Designed long axis; relative depth inferred from a generated image',
                    design_long_axis_m=target)
    if spall_profile:
        meta.update(recipe_version='oblique-photo-shallow-spall-v1',
                    geometry_source='Photographed rim and monocular road plane; bounded shallow relief guided by local aggregate contrast, not measured depth',
                    relief_profile='calibration/'+key+'-spall.json', relief_cap_m=profile['cap_m'])
    for field in ('source_url', 'license', 'license_url', 'author', 'framing', 'reference_kind'):
        if field in spec:
            meta[field] = spec[field]
    put(cal/(key+'.json'),meta)
    print(key,meta['outline_area_m2'],meta['oriented_extent_m'],meta['depth_p95_m'],stats['direct_flip_abs_p95_m'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specs',nargs='+',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'config/local.json')
    p.add_argument('--detail-profile', type=Path, help='Fit marked scarp and scan detail for this profile ID')
    p.add_argument('--detail-inputs', type=Path, help='Cached inset depths and checked scan maps')
    p.add_argument('--spall-profile', type=Path, help='Use an explicit shallow-relief prior for one photographed spall')
    p.add_argument('--puddle-profile', type=Path, help='Fit a photographed shoreline over an explicitly designed submerged bed')
    p.add_argument('--water-site', choices=['109','034','110'], help='Refill for the destination pothole anchor in the configured road plan')
    p.add_argument('--visible-surface', action='store_true', help='Reproject the photo on final, visible triangles')
    p.add_argument('--pavement', type=Path, default=ROOT.parent/'work/reconstruction-inputs/pbr/Asphalt031')
    args = p.parse_args()
    if bool(args.detail_profile) != bool(args.detail_inputs):
        p.error('Supply --detail-profile and --detail-inputs together')
    if args.visible_surface and not args.detail_profile:
        p.error('--visible-surface requires a dry-pit detail profile')
    if args.spall_profile and (len(args.specs) != 1 or args.detail_profile):
        p.error('--spall-profile requires one outline and cannot be combined with a scarp fit')
    if args.water_site and not args.puddle_profile:
        p.error('--water-site requires --puddle-profile')
    if args.puddle_profile and (len(args.specs)!=1 or args.detail_profile or args.spall_profile):
        p.error('--puddle-profile requires one outline and no other relief profiles')
    if args.out.exists():
        p.error('Choose a new output directory')
    cfg = json.loads(args.config.read_text())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    with TemporaryDirectory(dir=args.out.parent,prefix='.oblique-') as tmp:
        stage = Path(tmp)/'run'
        bundle = stage/'series'
        for part in ('references','calibration'):
            (bundle/part).mkdir(parents=True)
        rebuild = bundle/'rebuild'
        for part in ('common','series'):
            shutil.copytree(ROOT/'recipes'/part,rebuild,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for spec in args.specs:
            build_case(spec,args.inputs,bundle,args.spall_profile)
        if args.puddle_profile:
            from photo_puddle import prepare
            prepare(bundle,args.specs[0],args.puddle_profile,args.water_site,cfg.get('road_config'))
        keys = [json.loads(s.read_text())['id'] for s in args.specs]
        if args.detail_profile:
            from oblique_detail import refine
            if json.loads(args.detail_profile.read_text())['id'] not in keys:
                raise ValueError('Detail profile ID must occur in the survey')
            refine(bundle,args.detail_profile,args.detail_inputs,args.pavement)
            if args.visible_surface:
                from large_pits.texture import finish
                finish(bundle,args.detail_profile,args.detail_inputs)
        put(bundle/'run.json',dict(edition='series',ids=keys,mesh_only=False,preview=True,published={},
            calibration='oblique_single_photo',reference_sha256={p.name:digest(p) for p in (bundle/'references').iterdir()}))
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',PYTHONDONTWRITEBYTECODE='1')
        run_step(cfg['mesh_python'], rebuild/'fused_pit_mesh.py', keys, env)
        run_step(cfg['compute_python'],rebuild/'export_splats.py',['--cache',bundle/'mesh_cache',*keys],env)
        run_step(cfg['mesh_python'],rebuild/'render_mesh.py',['--cache',bundle/'mesh_cache','--out',bundle/'views',*keys],env)
        run_step(cfg['mesh_python'],rebuild/'verify.py',[],env)
        put(stage/'run.json',dict(editions=['series'],assets=keys))
        stage.rename(args.out)


if __name__ == '__main__':
    main()
