"""Metric height fields from a source silhouette, detrended monocular depth, and image detail."""
import hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

from fracture_wall import add_side, side_weights
from mesh_io import ROOT, grid_faces, lookup, luminance, road_atlas, smooth, source


def remove_plane(depth, mask):
    n = depth.shape[0]
    y, x = np.mgrid[-1:1:complex(n), -1:1:complex(n)].astype("f4")
    outside = (distance_transform_edt(~mask) > n*.023)[::10, ::10]
    def basis(a, b):
        return np.stack([np.ones_like(a), a, b, a*b, a*a, b*b], -1)
    a = basis(x[::10,::10][outside], y[::10,::10][outside])
    b = depth[::10,::10][outside]
    use = np.ones(len(b), bool)
    for _ in range(5):
        coef = np.linalg.lstsq(a[use], b[use], rcond=None)[0]
        e = b-a@coef
        mad = max(np.median(abs(e[use]-np.median(e[use]))) * 1.4826, .01)
        use = abs(e-np.median(e[use])) < mad * 2.8
    return (coef[0]+coef[1]*x+coef[2]*y+coef[3]*x*y+coef[4]*x*x+coef[5]*y*y-depth), coef


def build(mesh, key, cfg, grain_atlas):
    img = source(key)
    outline = np.asarray(Image.open(ROOT/'calibration'/(key+'_outline.png'))) > 127
    depth_path = ROOT/'calibration'/(key+'_depth.npz')
    with np.load(depth_path) as data:
        if str(data['source_sha256']) != hashlib.sha256((ROOT/'references'/(key+'.png')).read_bytes()).hexdigest():
            raise ValueError('Relative depth does not match RGB')
        depth = data['inverse_depth'].astype('f4')
    d, plane = remove_plane(depth, outline)
    metric_unit = cfg['span'] / (img.shape[0]-1)
    signed = (distance_transform_edt(outline)-distance_transform_edt(~outline)).astype('f4') * metric_unit
    signed = gaussian_filter(signed, 1.0)
    ref = np.percentile(d[signed > .012], 94)
    if ref <= .025:
        raise ValueError('No supported inward depression in '+key)
    learned = np.clip(d / ref, 0, 1.08)
    z = cfg['depth']
    near_edge = smooth((signed+.001) / max(.014, z*.31))
    profile = .42*smooth((signed+.001)/max(.020, z*.32)) + .58*smooth(signed/max(.04, z*1.48))
    h = -z * (.68*learned*near_edge + .32*profile)
    lum = luminance(img)
    grain = gaussian_filter(lum, 1.6)-gaussian_filter(lum, 8.)
    h += np.clip(grain* (.013 + .052*near_edge), -.009, .012)
    water = np.zeros_like(outline)
    if 'water' in cfg:
        water = np.asarray(Image.open(ROOT/'calibration'/(key+'_water.png'))) > 127
        sw = (distance_transform_edt(water)-distance_transform_edt(~water))*metric_unit
        weight = smooth((sw+.013)/.014)
        h = (1-weight)*h + weight*(-z*.87)
    # Reference XY extent is independent of the padded, host-blended patch.
    rows, cols = np.where(outline)
    cx = (cols.min()+cols.max())/2 / (img.shape[1]-1) * 479
    cy = (rows.min()+rows.max())/2 / (img.shape[0]-1) * 479
    extent = cfg['span']*.785
    def axes(n):
        a = np.linspace(-extent, extent, n, dtype='f4')
        x,y = np.meshgrid(a,-a)
        return x, y, cx+x*479/cfg['span'], cy-y*479/cfg['span']
    x,y,u,v = axes(1880)
    atlas = road_atlas(img, u, v, grain_atlas)
    mat = mesh.material('Pavement and exposed aggregate', atlas, .76 if key=='wet' else .90, 0., 3.4)
    n = 401
    x,y,u,v = axes(n)
    dist_to_orig = np.minimum.reduce([u,479-u,v,479-v])
    height = lookup(h,u,v)*smooth(dist_to_orig/8)
    alpha = smooth((extent*.94-np.maximum(abs(x),abs(y)))/.14)
    seg = lookup(outline.astype("f4"),u,v)
    wat = lookup(water.astype("f4"),u,v)
    xyz = np.stack([x,y,height],-1)
    uv = np.stack([.5+x/(2*extent), .5+y/(2*extent)],-1)
    faces = grid_faces(n,n)
    meanseg = seg.ravel()[faces].mean(axis=1)
    meanwater = wat.ravel()[faces].mean(axis=1)
    water_face = meanwater>.49
    inside = (meanseg>.45)&~water_face
    mesh.add('Road adjoining the fracture', xyz, faces[~inside & ~water_face], uv, mat, True)
    steep=side_weights(height,float(x[0,1]-x[0,0])).ravel()
    side=inside & (steep[faces].max(1)>.045)
    mesh.add('Fracture lip and exposed bed', xyz, faces[inside & ~side], uv, mat)
    if side.any():
        add_side(mesh,xyz.reshape(-1,3),faces[side],uv.reshape(-1,2),atlas,steep,key)
    if water_face.any():
        # A planar, separately shaded patch follows the segmented wet line.
        lo = uv.reshape(-1,2)[faces[water_face]].min(axis=(0,1))-.003
        hi = uv.reshape(-1,2)[faces[water_face]].max(axis=(0,1))+.003
        ww = np.linspace(lo[0],hi[0],512)
        vv = np.linspace(hi[1],lo[1],512)
        tx,ty=np.meshgrid(ww,vv)
        liquid = lookup(atlas,tx,1-ty,scale=1.)
        water_mat = mesh.material('Muddy standing water',liquid,.135,0.,.035)
        mesh.add('Water surface', xyz, faces[water_face], (uv-lo)/(hi-lo), water_mat)
    width=(cols.max()-cols.min())*metric_unit
    length=(rows.max()-rows.min())*metric_unit
    calibration={'width_m':float(width),'length_m':float(length),'depth_design_m':z,'depth_p95_m':float(-np.percentile(height[seg>.6],5)),
                 'relative_depth_plane_coefficients':plane.tolist(),'metric_source_span_m':cfg['span'],
                 'relative_depth_p94_after_plane_removal':float(ref),
                 'relative_depth_source':'Depth Anything V2 Small, 518/770 plus horizontal flip',
                 'geometry_source':'fitted monocular depth and registered silhouette with designed metric scale',
                  'road_projection':[float(cx),float(cy),479/cfg['span'],479/cfg['span'],img.shape[0]],
                 'state':cfg['family']}
    field={'height':height,'alpha':alpha,'width':extent*2,'length':extent*2,'defect_mask':seg,'water_mask':wat}
    return field,calibration
