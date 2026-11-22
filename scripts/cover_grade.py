"""Measure installed cover grades, pavement seams and bearing contact on the exported GLB."""
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from road_geometry import Surface


def check_source_relief(bundle, path, profile, trace, dest):
    """Compare the mesh with original displacement pixels beyond the vertices' resampling footprint."""
    from road_geometry import glb_triangles

    face = next(v for name, v, _ in glb_triangles(path, names=True) if name == 'Lid face')
    size = np.asarray(profile['size_m'])
    uv = face[:, :2] / (size * [1, -1]) + 0.5
    mapping = np.array(trace['pixel_to_canonical'])
    n = trace['resolution']
    inverse = np.linalg.inv(mapping)
    locations = uv*(n-1)
    vertex_pixels = cv2.perspectiveTransform(locations[None], inverse)[0]
    canonical = np.load(bundle/'calibration/relief.npy')
    values = map_coordinates(canonical, [uv[:, 1]*(n-1), uv[:, 0]*(n-1)], order=1, mode='nearest')
    low, high = np.percentile(values, [3, 97])
    source = np.asarray(Image.open(bundle/'references'/profile['displacement']))
    y, x = np.mgrid[1:source.shape[0]:2, 1:source.shape[1]:2]
    pixels = np.c_[x.ravel(), y.ravel()].astype('f8')
    xy = (cv2.perspectiveTransform(pixels[None], mapping)[0]/(n-1)-0.5)*size*[1,-1]
    # Two successive bilinear samples use a small neighbourhood around each mesh vertex.
    # Project the canonical sample neighbourhood back to the source, then allow for its own bilinear lookup.
    support = max(np.linalg.norm(cv2.perspectiveTransform((locations+delta)[None], inverse)[0]-vertex_pixels,
                                 axis=1).max() for delta in [(-1,-1),(-1,1),(1,-1),(1,1)])
    exclusion = float(support + np.sqrt(2) + 0.05)
    keep = cKDTree(vertex_pixels).query(pixels)[0] > exclusion
    keep &= (abs(xy/size) < 0.5).all(axis=1)
    xy, pixels = xy[keep], pixels[keep].astype(int)
    surface = Surface(path, include_names={'Lid face'})
    origins = np.c_[xy, np.ones(len(xy))]
    dist, _, _ = surface.cast(origins, np.broadcast_to([0.,0.,-1.], origins.shape))
    hit = np.isfinite(dist)
    px, py = pixels[hit].T
    observed = (1-dist[hit])*1000
    reference = (np.clip((source[py, px].astype(float)-low)/(high-low), 0, 1)-0.5)*profile['relief_span_m']*1000
    if len(reference) < 1000 or not np.isfinite(reference).all():
        raise ValueError('Insufficient independent source pixels on the lid face')
    error = abs(observed-reference)
    corr = float(np.corrcoef(observed, reference)[0,1])
    # The median reference height minimizes the mean absolute error of a constant surface.
    flat_mae = float(np.mean(abs(reference-np.median(reference))))
    if not np.isfinite(corr) or error.mean() >= flat_mae:
        raise ValueError('The surface failed to recover more relief than a constant-height face')
    photo = np.asarray(Image.open(bundle/'references/frame.png').convert('RGB'))
    overlay = np.uint8(photo*0.53)
    palette = np.array([[85,185,154],[190,200,114],[240,169,74],[202,74,52]])
    # Paint disjoint 2 × 2 display cells for the stride-2 measurements.
    colors = palette[np.digitize(error, [0.5,1,2])]
    for dx, dy in [(0,0),(-1,0),(0,-1),(-1,-1)]:
        overlay[py+dy,px+dx] = colors
    box = np.array(profile['corners_xy'])
    crop = (*box.min(0), *(box.max(0)+1))
    Image.fromarray(overlay).crop(crop).save(dest/'source-relief.png')
    return dict(heldout_source_pixels=len(reference), vertex_exclusion_radius_px=float(exclusion),
                design_relief_span_mm=1000*profile['relief_span_m'], pearson_r=corr,
                median_absolute_error_mm=float(np.median(error)), p95_absolute_error_mm=float(np.percentile(error,95)),
                mean_absolute_error_mm=float(error.mean()), flat_mean_absolute_error_mm=flat_mae)


def check_grade(path, parts, field_file, reference_ring, profile, state):
    grade = state['installation_grade']
    bearing = np.deg2rad(grade['uphill_deg'])
    direction = np.array([np.cos(bearing), np.sin(bearing)])
    tilt = np.deg2rad(grade['tilt_deg'])
    normal = np.r_[-np.sin(tilt) * direction, np.cos(tilt)]
    center = np.array([0, 0, grade['center_z_m']])
    stock = {part: (v, f) for part, v, f in parts}
    flat, flat_faces = reference_ring
    ring, faces = stock[5]
    radius = np.array(profile['size_m'])/2 + profile['frame_band_m']
    if not np.array_equal(faces, flat_faces):
        raise ValueError('Ring topology changed')
    # Measure the rim plane on its original outermost vertex loop, including the inclined state.
    radial = flat[:, :2]/radius
    distance = np.linalg.norm(radial, axis=1) if profile['shape']=='round' else np.max(abs(radial), axis=1)
    edge = ring[np.isclose(distance, 1, atol=1e-5)]
    plane = np.linalg.lstsq(np.c_[edge[:, :2], np.ones(len(edge))], edge[:, 2], rcond=None)[0]
    if not np.allclose(plane, np.r_[np.tan(tilt)*direction, center[2]], rtol=0, atol=1e-6):
        raise ValueError('Measured outer frame plane differs from the specified installation')
    expected = flat.astype(float).copy()
    projected = flat[:, :2] @ direction
    expected[:, :2] += ((np.cos(tilt)-1)*projected - np.sin(tilt)*flat[:, 2])[:, None] * direction
    expected[:, 2] = projected*np.sin(tilt) + flat[:, 2]*np.cos(tilt) + center[2]
    ring_error = float(np.abs(expected - ring).max())
    if ring_error > 2e-6:
        raise ValueError('Installed seat changed shape')
    road = stock[1][0]
    if road[:, 2].any() or np.max(cKDTree(road[:, :2]).query(edge[:, :2])[0]) > 1e-6:
        raise ValueError('Pavement is off datum or disconnected from the collar')
    if (8 in stock) != bool(np.max(np.abs(edge[:, 2])) > 1e-6):
        raise ValueError('Unexpected exposed collar')
    if 8 in stock:
        collar, triangles = stock[8]
        ends = np.vstack([edge, edge * [1, 1, 0]])
        if cKDTree(ends).query(collar)[0].max() > 1e-6 or cKDTree(collar).query(ends)[0].max() > 1e-6:
            raise ValueError('Exposed collar does not meet both the ring and the road')
        tri = collar[triangles].astype(float)
        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        midpoint = tri.mean(1)
        if np.any(np.sum(midpoint[:, :2]*normals[:, :2], axis=1)*midpoint[:, 2] <= 0):
            raise ValueError('The visible collar faces away from the exposed side')
    metrics = dict(center_z_m=float(plane[2]), tilt_deg=float(np.rad2deg(np.arctan(np.linalg.norm(plane[:2])))),
                   rim_range_m=[float(edge[:, 2].min()), float(edge[:, 2].max())],
                   ring_rigidity_error_m=ring_error, floor_z_m=float(stock[7][0][:, 2].min()))
    if not state.get('absent'):
        angle = np.arange(128)*2*np.pi/128
        outline = np.c_[np.cos(angle), np.sin(angle)]
        if profile['shape'] == 'rect':
            outline /= np.max(abs(outline), axis=1, keepdims=True)
        local = outline * (np.array(profile['size_m'])/2-profile['seat_overlap_m']/2)
        xy = local + ((np.cos(tilt)-1)*(local@direction))[:, None]*direction
        origins = np.c_[xy, (local@direction)*np.sin(tilt)] + center + normal
        directions = np.broadcast_to(-normal, origins.shape)
        seat = Surface(path, include_names={'Cast ring and cover seat'}).cast(origins, directions)[0]
        bottom = Surface(path, include_names={'Lid underside'}).cast(origins, directions)[0]
        gaps = abs(seat-bottom)
        if not np.isfinite(gaps).all() or gaps.max() > 0.0002:
            raise ValueError('Seated cover lost bearing contact')
        metrics.update(bearing_probes=len(gaps), max_bearing_gap_m=float(gaps.max()))
    # Cross-check the Blender height-field export against Open3D rays on the delivered GLB.
    with np.load(field_file) as field:
        step = 4
        axis = np.linspace(-float(field['width'])/2, float(field['width'])/2, len(field['height']))[1:-1:step]
        x, y = np.meshgrid(axis, -axis)
        origins = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
        for name, omit in [('height', ()), ('substrate_height', (9, 10, 11))]:
            surface = Surface(path, exclude=omit)
            hits, _, _ = surface.cast(origins, np.broadcast_to([0., 0., -1.], origins.shape))
            error = abs(1 - hits.reshape(x.shape) - field[name][1:-1:step, 1:-1:step])
            if not np.isfinite(error).all() or error.max() > 0.0002:
                raise ValueError('Grid and exported GLB disagree: ' + name)
            metrics[name + '_max_error_m'] = float(error.max())
        metrics['grid_probes_per_layer'] = x.size
    return metrics
