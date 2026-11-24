"""Hydrostatic storage on the piecewise-linear triangles of a measured road patch."""

import numpy as np
from scipy.ndimage import binary_dilation, label, map_coordinates

# Each cell of grid_faces has the diagonal from bottom left to top right.
NEIGHBORS = [[0, 1, 1], [1, 1, 1], [1, 1, 0]]


def component(elevation, head, seed):
    components, _ = label(elevation < head, structure=NEIGHBORS)
    number = components[seed]
    return (components == number) if number else np.zeros(elevation.shape, bool)


def resting_seed(elev, inlet):
    """Follow descending mesh edges from the inlet, crossing level terraces when needed."""
    r, c = inlet
    offsets = np.argwhere(NEIGHBORS) - 1
    offsets = offsets[np.any(offsets != 0, axis=1)]
    while 0 < r < elev.shape[0] - 1 and 0 < c < elev.shape[1] - 1:
        neighbors = offsets + [r, c]
        drop = elev[r, c] - elev[tuple(neighbors.T)]
        if drop.max() > 0:
            # Regular square cells have diagonal edges sqrt(2) longer.
            r, c = neighbors[np.argmax(drop / np.linalg.norm(offsets, axis=1))]
            continue
        if (drop == 0).any():
            tiles, _ = label(elev == elev[r, c], structure=NEIGHBORS)
            terrace = tiles == tiles[r, c]
            if drains(terrace):
                break
            outlets = binary_dilation(terrace, structure=NEIGHBORS) & (elev < elev[r, c])
            if outlets.any():
                r, c = np.unravel_index(np.where(outlets, elev, np.inf).argmin(), elev.shape)
                continue
        return int(r), int(c)
    raise ValueError('The inlet drains out of the measured patch before reaching a depression')


def drains(mask):
    return mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()


def submerged_triangles(faces, wet):
    flags = wet.ravel()[faces]
    keep = flags.any(axis=1)
    return faces[keep], flags[keep]


def integrate(faces, flags, drop, triangle_area):
    """Integral of positive vertical depth, exactly linear on each clipped triangle."""
    depths = drop[faces]
    count = flags.sum(axis=1)
    full = depths[count == 3].sum() / 3
    for n in (1, 2):
        selected = depths[count == n]
        if not len(selected):
            continue
        # For one wet corner integrate the small similar triangle. For two wet
        # corners integrate the whole signed triangle less its negative tip.
        d = np.sort(selected, axis=1)
        tip, a, b = (d[:, 2], d[:, 0], d[:, 1]) if n == 1 else (d[:, 0], d[:, 1], d[:, 2])
        small = tip**3 / ((tip - a) * (tip - b)) / 3
        full += small.sum() if n == 1 else (d.sum(axis=1) / 3 - small).sum()
    return float(full * triangle_area)


def make_interface(xyz, faces, flags, drop):
    """Clip the original oriented triangles and weld shoreline intersections by edge ID."""
    tris = [faces[flags.all(axis=1)]]
    extra_edges, fragments = [], []
    edge_offset = 0
    for n in (1, 2):
        subset = flags.sum(axis=1) == n
        cropped, inside = faces[subset], flags[subset]
        if not len(cropped):
            continue
        solo = (inside if n == 1 else ~inside).argmax(axis=1)
        row = np.arange(len(cropped))
        a, b, c = (cropped[row, (solo + i) % 3] for i in range(3))
        edges = np.concatenate([np.column_stack([a, b]), np.column_stack([c, a])])
        extra_edges.append(np.sort(edges, axis=1))
        ab = len(xyz) + edge_offset + row
        ca = ab + len(row)
        edge_offset += len(edges)
        if n == 1:
            fragments.append(np.column_stack([a, ab, ca]))
        else:
            fragments.extend([np.column_stack([ab, b, c]), np.column_stack([ab, c, ca])])
    vertices = xyz.copy()
    vertices[:, 2] += drop
    if extra_edges:
        edges, inverse = np.unique(np.concatenate(extra_edges), axis=0, return_inverse=True)
        a, b = edges.T
        t = drop[a] / (drop[a] - drop[b])
        shoreline = xyz[a] + (xyz[b] - xyz[a]) * t[:, None]
        # Endpoints exactly on the plane can serve several crossing edges.
        addition = np.where(t == 0, a, np.where(t == 1, b, len(xyz) + np.arange(len(t))))
        mapping = np.r_[np.arange(len(xyz)), addition[inverse]]
        tris.extend(mapping[f] for f in fragments)
        vertices = np.vstack([vertices, shoreline])
    faces = np.concatenate(tris)
    p = vertices[faces]
    faces = faces[np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) > 1e-13]
    used, local = np.unique(faces, return_inverse=True)
    return vertices[used], local.reshape(-1, 3)


def solve(xyz, faces, field, spec):
    normal = np.array(spec['up_local'], dtype=float)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or abs(np.linalg.norm(normal) - 1) > 1e-6
        or normal[2] <= 0
    ):
        raise ValueError('up_local must be a unit upward world vertical expressed in asset coordinates')
    uv = np.asarray(spec['seed_uv'], dtype=float)
    liters = float(spec['input_liters'])
    if uv.shape != (2,) or not np.all((0 < uv) & (uv < 1)) or not np.isfinite(liters) or liters <= 0:
        raise ValueError('Supply a positive finite volume and an interior seed_uv (origin top left)')
    shape = field['height'].shape
    seed = tuple(np.rint(uv[::-1] * (np.array(shape) - 1)).astype(int))
    if field['defect_mask'][seed] < 0.5:
        raise ValueError('Water inlet is outside the measured pit')
    elev = (xyz @ normal).reshape(shape)
    seed = resting_seed(elev, seed)
    bottom = float(elev[seed])
    low, high = bottom, float(np.nextafter(elev.max(), np.inf))
    for _ in range(40):
        mid = (low + high) / 2
        if drains(component(elev, mid, seed)):
            high = mid
        else:
            low = mid
    spill = low
    pool = component(elev, spill, seed)
    if not pool.any():
        raise ValueError('The selected seed drains immediately at this road orientation')
    # After finding the spill, only this connected catchment can enter volume queries.
    possible = faces[pool.ravel()[faces].any(axis=1)]
    tri = xyz[faces[0], :2]
    area = abs(float(np.linalg.det(tri[1:] - tri[0]))) * 0.5

    def storage(head):
        wet = component(elev, head, seed)
        clipped, flags = submerged_triangles(possible, wet)
        depth = (head - elev.ravel()) / normal[2]
        return integrate(clipped, flags, depth, area) * 1000, wet, clipped, flags, depth

    capacity = storage(spill)[0]
    target = min(liters, capacity)
    head = spill
    if liters < capacity:
        low, high = bottom, spill
        for _ in range(38):
            mid = (low + high) / 2
            if storage(mid)[0] < target:
                low = mid
            else:
                high = mid
        head = (low + high) / 2
    stored, wet, clipped, flags, depth = storage(head)
    if abs(stored - target) > max(1e-8, target * 1e-7):
        raise ValueError(
            'Volume crosses a disconnected depression at a saddle. Choose a separate basin or inlet.'
        )
    water, water_faces = make_interface(xyz, clipped, flags, depth)
    p = water[water_faces]
    surface_area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum() / 2
    stats = dict(
        spec,
        basin_seed_rc=list(seed),
        retained_liters=stored,
        spilled_liters=max(0.0, liters - capacity),
        capacity_liters=capacity,
        free_surface_m=head,
        spill_elevation_m=spill,
        max_water_depth_m=float((head - elev[wet].min())),
        surface_area_m2=float(surface_area),
    )
    return water, water_faces, wet, depth.reshape(shape), stats


def add_water(mesh, xyz, faces, field, color, spec):
    points, triangles, wet, depth, stats = solve(xyz, faces, field, spec)
    width = float(field['width'])
    uv = points[:, :2] / width + 0.5
    # Bake the photographed bed seen through the water into a portable opaque
    # base colour. Specular lighting still comes from the smooth dielectric.
    r, c = np.indices(color.shape[:2], dtype='f4')
    coords = [r * (len(depth) - 1) / (len(color) - 1), c * (depth.shape[1] - 1) / (color.shape[1] - 1)]
    thickness = np.maximum(0, map_coordinates(depth, coords, order=1)) * spec['up_local'][2]
    trans = np.exp(-thickness[:, :, None] * np.array([13.0, 9.0, 7.0]))
    water_color = 0.73 * color * trans + np.array([57.0, 67.0, 62.0]) * (1 - trans) + [7.0, 10.0, 12.0]
    if spec.get('bed_optics') == 'wet-asphalt':
        water_color = color * (.85 * np.exp(-thickness[:, :, None]*np.array([3.2,2.8,2.5])))
    material = mesh.material(
        'Standing water over photographed bed', np.uint8(water_color.clip(0, 255)), 0.07, 0, 0
    )
    mesh.add('Water surface', points, triangles, uv, material)
    field['water_mask'] = wet.astype('f4')
    field['water_depth_local_z'] = np.where(wet, depth, 0).astype('f4')
    return stats
