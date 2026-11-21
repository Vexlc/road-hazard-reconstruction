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
    n=mask.shape[0]
    span,depth,slope = (float(cfg[k]) for k in ('span','depth','wall_slope'))
    if not all(np.isfinite(v) and v>0 for v in (span,depth,slope)):
        raise ValueError("span, depth and wall_slope must be positive")
    pitch=span/(n-1)
    axis=np.linspace(-span/2,span/2,n,dtype='f4')
    x,y=np.meshgrid(axis,-axis)
    # Each marked pixel has a cell of finite width. An inward V cross-section
    # preserves even a one-pixel branch and caps the widest crevices at design depth.
    half_width=np.maximum(0,distance_transform_edt(mask)-.5)*pitch
    margin=np.minimum(span/2-abs(x),span/2-abs(y))
    taper=smooth(margin/max(.012,4*pitch))
    height=-np.minimum(depth,half_width*slope)*taper
    outside=distance_transform_edt(~mask)*pitch
    alpha=smooth((.095-outside)/.06)*taper
    xyz=np.stack((x,y,height),axis=-1)
    uv=np.stack((.5+x/span,.5+y/span),axis=-1)
    triangles=grid_faces(n,n)
    # Include both slopes bordering a thin marked centreline. Outside vertices
    # keep their photographed color and exactly zero height.
    groove=mask.ravel()[triangles].any(axis=1)
    material=mesh.material('Photographed asphalt and open fissure',img,.94,0.,1.8)
    mesh.add('Road adjoining the fracture',xyz,triangles[~groove],uv,material,True)
    mesh.add('Fracture lip and exposed bed',xyz,triangles[groove],uv,material)
    rows,cols=np.where(mask)
    fields=dict(height=height,alpha=alpha,defect_mask=mask,water_mask=np.zeros_like(mask),width=span,length=span)
    meta=dict(state='crack_network',surface_class='crack',depth_profile='masked_groove',
              geometry_source='Registered crack mask with designed groove side slope and depth limit',
              width_m=float(np.ptp(cols)*pitch),length_m=float(np.ptp(rows)*pitch),
              depth_design_m=depth,depth_p95_m=float(np.percentile(-height[mask],95)),wall_slope=slope,
              metric_source_span_m=span,source_pixel_pitch_design_m=pitch,components=int(label(mask,np.ones((3,3)))[1]),
              mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),mask_pixels=int(mask.sum()),
              road_projection=[239.5,239.5,479/span,479/span,n],native_scene_category='pothole')
    return fields,meta
