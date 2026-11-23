"""Reproject the photograph onto the final triangles and resolve unobserved material in 3-D."""
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt


def surface_samples(height, width, rows, cols, size):
    """Interpolate on the same diagonal as the native mesh, including half-texel UVs."""
    n = len(height)
    s = (cols + .5) * (n - 1) / size
    t = (rows + .5) * (n - 1) / size
    i, j = s.astype(int), t.astype(int)
    u, v = s - i, t - j
    a, b = height[j, i], height[j, i + 1]
    c, d = height[j + 1, i], height[j + 1, i + 1]
    z = np.where(u + v <= 1, a + u*(b-a) + v*(c-a),
                 d + (1-u)*(c-d) + (1-v)*(b-d))
    return np.stack(np.broadcast_arrays((s/(n-1)-.5)*width, (.5-t/(n-1))*width, z), -1).astype('f4')


def visibility(height, width, points, camera):
    """Intersect source rays with the undilated mesh; block texels behind the first surface."""
    n = len(height)
    line = np.linspace(-width/2, width/2, n, dtype='f4')
    vertices = np.stack((*np.meshgrid(line, line[::-1]), height), -1).reshape(-1, 3)
    ids = np.arange(n*n, dtype='u4').reshape(n,n)
    a, b, c, d = ids[:-1,:-1], ids[:-1,1:], ids[1:,:-1], ids[1:,1:]
    faces = np.stack((a,c,b,c,d,b), -1).reshape(-1,3)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    seen = np.empty(points.shape[:2], bool)
    for row in range(0, len(points), 96):
        block = points[row:row+96]
        direction = block-camera
        rays = np.concatenate((np.broadcast_to(camera, block.shape), direction), -1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype('f4')), nthreads=4)
        # Distances are in metres along the ray. The tolerance only absorbs raster roundoff.
        seen[row:row+96] = abs(hit['t_hit'].numpy()-1)*np.linalg.norm(direction,axis=2) < .001
    return seen


def scan_bands(profile, inputs, points, pitch):
    """Sample albedo and micro-relief from the same scan coordinates on all three planes."""
    y_deriv, x_deriv = np.gradient(cv2.GaussianBlur(points[:,:,2], (0,0), 2), pitch)
    weights = np.stack((x_deriv**2, y_deriv**2, np.ones_like(x_deriv)), -1)
    weights /= weights.sum(-1, keepdims=True)
    x, y, z = points.transpose(2,0,1)
    grain = np.zeros_like(x)
    relief = np.zeros_like(x)
    for name, mix in [('stony_dirt_path', .68), ('gravelly_sand', .32)]:
        entry = profile['materials'][name]
        color = cv2.imread(str(inputs/Path(entry['albedo']['url']).name), cv2.IMREAD_GRAYSCALE)
        disp = cv2.imread(str(inputs/Path(entry['displacement']['url']).name), cv2.IMREAD_UNCHANGED)
        if disp.ndim == 3:
            disp = disp[:,:,0]
        angle = -.54 if name == 'stony_dirt_path' else .22
        cs, sn = np.cos(angle), np.sin(angle)
        for field, target, amplitude in [(color, grain, 1.), (disp, relief, .0018)]:
            tex = field.astype('f4')
            rate = tex.shape[1]/entry['width_m']
            high = tex - cv2.GaussianBlur(tex, (0,0), .033*rate)
            high /= max(1e-6, float(np.percentile(abs(high), 85)))
            for axis, (a,b) in enumerate(((y,z), (x,z), (x,y))):
                uv = np.stack(((cs*a-sn*b)*rate+.35*tex.shape[1],
                               -(sn*a+cs*b)*rate+.42*tex.shape[0]), -1).astype('f4')
                detail = cv2.remap(high, uv, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                target += mix*amplitude*weights[:,:,axis]*detail
    return grain, relief, 1-weights[:,:,2]


def finish(bundle, profile_path, inputs):
    from oblique_pits import polygon_mask, put

    profile = json.loads(profile_path.read_text())
    key = profile['id']
    cal = bundle/'calibration'
    refs = bundle/'references'
    capture = json.loads((cal/(key+'_capture.json')).read_text())
    spec = json.loads((cal/(key+'-outline.json')).read_text())
    meta = json.loads((cal/(key+'.json')).read_text())
    rgb = np.array(ImageOps.exif_transpose(Image.open(bundle/meta['source_image'])).convert('RGB'))
    with np.load(cal/(key+'.npz')) as field:
        height, inside, width = field['height'], field['defect_mask'], float(field['width'])
    if len(height) != height.shape[1]:
        raise ValueError('Expected a square road-aligned height grid')
    color_path, normal_path = refs/(key+'_ortho.png'), bundle/meta['normal_texture']
    color = np.array(Image.open(color_path))
    normals = np.array(Image.open(normal_path))
    size = len(color)
    pitch = width/size
    mask = cv2.resize(inside.astype('f4'), (size,size), interpolation=cv2.INTER_LINEAR)
    rows, cols = np.nonzero(mask > 0)
    region = np.s_[max(rows.min()-8,0):min(rows.max()+9,size), max(cols.min()-8,0):min(cols.max()+9,size)]
    rows, cols = np.ogrid[region]
    points = surface_samples(height, width, rows, cols, size)
    K = np.float32(capture['road_planes']['direct']['K'])
    axes = np.float32(capture['local_axes_in_camera'])
    origin = np.float32(capture['local_origin_in_camera'])
    visible = visibility(height,width,points,-origin@axes)
    cam = points@axes.T + origin
    uv = (cam[:,:,:2]/cam[:,:,2:] * K.diagonal()[:2] + K[:2,2]).astype('f4')
    valid = polygon_mask(rgb.shape[:2],[spec['rim']]) & ~polygon_mask(rgb.shape[:2],spec.get('exclude_polygons',[]))
    valid = valid.astype('u1')
    for x0,y0,x1,y1,width_px in spec.get('overlay_segments',[]):
        cv2.line(valid,(x0,y0),(x1,y1),0,width_px+2)
    def sample(data):
        return cv2.remap(data, uv, None, cv2.INTER_LINEAR)
    bed = mask[region] > .5
    clean = (sample(valid.astype('f4')) > .999) & bed & visible
    if not clean.any():
        raise ValueError('The source camera has no unobstructed texture samples')
    # Local minimum singular value gives the available source pixels per world texel.
    du_row,du_col = np.gradient(uv[:,:,0])
    dv_row,dv_col = np.gradient(uv[:,:,1])
    det = abs(du_row*dv_col-du_col*dv_row)
    scale = du_row**2+du_col**2+dv_row**2+dv_col**2
    smax = np.sqrt((scale+np.sqrt(np.maximum(scale**2-4*det**2,0)))*.5)
    coverage = np.clip(det/np.maximum(smax,.0001)/.09, 0,1)*clean
    nearest = distance_transform_edt(~clean,return_distances=False,return_indices=True)
    observed = sample(rgb).astype('f4')[tuple(nearest)]
    broad = cv2.GaussianBlur(observed,(0,0),.045/pitch)
    # Smooth donor colour across self-occlusions; never copy a sharp edge along the sight line.
    appearance = broad + coverage[...,None]*(observed-broad)
    grain, relief, wall = scan_bands(profile,inputs,points,pitch)
    amplitude = 12.0 + 8.0*(1-coverage)
    appearance += (amplitude*grain)[...,None]
    weight = mask[region][...,None]
    color[region] = np.rint(np.clip(color[region]*(1-weight)+appearance*weight,0,255))
    # Correct the planar micro-normal where it stretches across an exposed bank.
    dr, dc = np.gradient(cv2.GaussianBlur(relief,(0,0),.65),pitch)
    hrow, hcol = np.gradient(points[:,:,2],pitch)
    normal = np.stack((-dc/np.hypot(1,hcol),-dr/np.hypot(1,hrow),np.ones_like(dc)), -1)
    normal /= np.linalg.norm(normal,axis=2,keepdims=True)
    old = normals[region]/127.5-1
    mix = (mask[region]*wall)[...,None]
    new = (1-mix)*old+mix*normal
    new /= np.maximum(np.linalg.norm(new,axis=2,keepdims=True),1e-5)
    normals[region] = np.rint(np.clip((new+1)*127.5,0,255))
    Image.fromarray(color).save(color_path)
    Image.fromarray(normals).save(normal_path)
    capture['texture_visibility'] = dict(bed_texels=int(bed.sum()), first_hit_texels=int((bed&visible).sum()),
        clean_texels=int(clean.sum()), hidden_texels=int((bed&~visible).sum()), three_plane_scan=True,
        texel_pitch_m=pitch, ray_tolerance_m=.001)
    put(cal/(key+'_capture.json'),capture)
    meta['recipe_version'] = 'oblique-photo-visible-v3'
    put(cal/(key+'.json'),meta)
