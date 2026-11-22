"""Build a textured crack network from a registered binary mask, preserving its branches."""
import hashlib

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

from mesh_io import ROOT, grid_faces, smooth, source


def build(mesh,key,cfg):
    img = source(key)
    mask_path = ROOT/'calibration'/(key+'_outline.png')
    mask = np.asarray(Image.open(mask_path).convert('L'))>127
    if mask.shape!=img.shape[:2] or mask.shape[0]!=mask.shape[1]:
        raise ValueError("Crack recipe needs a square RGB crop and same-size binary mask")
    if not 0 < mask.sum() < .40*mask.size:
        raise ValueError("Crack foreground must occupy less than 40% of the pavement crop")
    scanned = cfg.get('depth_profile') == 'scanned_relief'
    span = float(cfg['span'])
    if not np.isfinite(span) or span <= 0:
        raise ValueError("span must be positive")
    if scanned:
        step = int(cfg['step'])
        if step < 1 or (len(mask)-1) % step:
            raise ValueError("Scan grid does not span the RGB crop")
        mask = mask[::step,::step]
    n=mask.shape[0]
    pitch=span/(n-1)
    axis=np.linspace(-span/2,span/2,n,dtype='f4')
    x,y=np.meshgrid(axis,-axis)
    margin=np.minimum(span/2-abs(x),span/2-abs(y))
    taper=smooth(margin/max(.012,4*pitch))
    if scanned:
        with np.load(ROOT/'calibration'/(key+'_samples.npz'), allow_pickle=False) as samples:
            measured = samples['height']
        if measured.shape != mask.shape or not np.isfinite(measured).all():
            raise ValueError("Missing or misregistered scan samples")
        height = measured * taper
        details = {k:cfg[k] for k in ('material','source_pitch_m','source_page','source_author','tile_shape',
                   'road_plane_pixel','plane_fit_stride','material_scale','source_files','step','origin_col_row','intervals')}
        for optional in ('parent', 'normal_scale_fit', 'paired'):
            if optional in cfg:
                details[optional] = cfg[optional]
        details.update(geometry_source='Registered scan displacement and publisher Blender scale; outer edges taper to the road',
                       depth_profile='scanned_relief',source_grid_pitch_m=pitch)
        if 'normal_scale_fit' in cfg:
            details['geometry_source']='Scan displacement scaled to fitted tangent normals in alternate source blocks'
    else:
        depth,slope = (float(cfg[k]) for k in ('depth','wall_slope'))
        if not all(np.isfinite(v) and v>0 for v in (depth,slope)):
            raise ValueError("depth and wall_slope must be positive")
        half_width=np.maximum(0,distance_transform_edt(mask)-.5)*pitch
        height=-np.minimum(depth,half_width*slope)*taper
        details=dict(depth_profile='masked_groove',geometry_source='Registered crack mask with designed groove side slope and depth limit',
                     depth_design_m=depth,wall_slope=slope,source_pixel_pitch_design_m=pitch)
    outside=distance_transform_edt(~mask)*pitch
    alpha=smooth((.095-outside)/.06)*taper
    xyz=np.stack((x,y,height),axis=-1)
    uv=np.stack((.5+x/span,.5+y/span),axis=-1)
    triangles=grid_faces(n,n)
    # Include the bounding triangles of even a one-pixel trace.
    groove=mask.ravel()[triangles].any(axis=1)
    material=mesh.material('Scanned pavement and fissures' if scanned else 'Photographed asphalt and open fissure',
                           img,.94,0.,0. if scanned else 1.8,lossless_color=scanned)
    captured = height
    if 'slab' in cfg:
        if not scanned:
            raise ValueError('A detached slab requires registered surface heights')
        from faulted_slab import build as build_slab
        height, footprint, details['slab'] = build_slab(mesh,key,cfg,xyz,uv,triangles,groove,material)
        mask = mask | footprint
        alpha = smooth((.095-distance_transform_edt(~mask)*pitch)/.06)*taper
    else:
        mesh.add('Road adjoining the fracture',xyz,triangles[~groove],uv,material,True)
        mesh.add('Fracture lip and exposed bed',xyz,triangles[groove],uv,material)
    rows,cols=np.where(mask)
    fields=dict(height=height,alpha=alpha,defect_mask=mask,water_mask=np.zeros_like(mask),width=span,length=span)
    if 'slab' in cfg:
        fields.update(captured_height=captured,slab_socket=footprint)
    meta=dict(state='crack_network',surface_class='crack',**details,
              width_m=float(np.ptp(cols)*pitch),length_m=float(np.ptp(rows)*pitch),
              depth_p95_m=float(np.percentile(-height[mask],95)),
              metric_source_span_m=span,components=int(label(mask,np.ones((3,3)))[1]),
              mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),mask_pixels=int(mask.sum()),
              road_projection=[239.5,239.5,479/span,479/span,len(img)],native_scene_category='pothole')
    if 'slab' in cfg:
        meta.update(state='faulted_slab',surface_class='faulted_slab',
                    geometry_source='Photogrammetric top and optical fracture boundary; designed grade, clearance, thickness and foundation')
    return fields,meta
