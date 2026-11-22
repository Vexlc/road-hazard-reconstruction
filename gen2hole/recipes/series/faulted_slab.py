"""Extract an enclosed photographed pavement block and build its socket and rigid solid."""

import numpy as np
from scipy import ndimage
from PIL import Image

from mesh_io import ROOT, source


def block_cells(mask, cfg, step, pitch):
    col, row = cfg['seed_col_row_in_crop']
    if not (0 <= row < len(mask) and 0 <= col < mask.shape[1]):
        raise ValueError('Slab seed outside the scan')
    labels, _ = ndimage.label(~mask)
    key = labels[row, col]
    if not key or key in np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]]:
        raise ValueError('Seed must be on pavement entirely enclosed by photographed cracks')
    region = ndimage.binary_fill_holes(labels == key)
    cells = region[step // 2 : -1 : step, step // 2 : -1 : step]
    gap, inset = cfg['socket_clearance_m'], cfg['support_inset_m']
    if not np.isfinite([gap, inset]).all() or min(gap, inset) < pitch:
        raise ValueError('The clearance and support inset must be resolved by the scan grid')
    socket = ndimage.distance_transform_edt(~cells) * pitch <= gap
    support = ndimage.distance_transform_edt(cells) * pitch > inset
    for subset in (cells, socket, support):
        diagonal = (
            (subset[1:, 1:] == subset[:-1, :-1])
            & (subset[1:, :-1] == subset[:-1, 1:])
            & (subset[1:, 1:] != subset[1:, :-1])
        )
        if (
            ndimage.label(subset)[1] != 1
            or diagonal.any()
            or not np.array_equal(subset, ndimage.binary_fill_holes(subset))
            or subset[[0, -1], :].any()
            or subset[:, [0, -1]].any()
        ):
            raise ValueError(
                'Grid cannot resolve a simple interior slab, socket or support; choose a finer pitch or another seed'
            )
    return cells, socket, support, int(region.sum())


def boundary(faces):
    """Single directed CCW perimeter from the cap's existing triangles."""
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, idx, counts = np.unique(np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True)
    outline = edges[idx[counts == 1]]
    next_vertex = dict(outline.tolist())
    ids = [int(outline[0, 0])]
    for _ in range(len(outline) - 1):
        ids.append(next_vertex[ids[-1]])
    if next_vertex[ids[-1]] != ids[0] or len(set(ids)) != len(outline):
        raise ValueError('Expected one simple closed boundary')
    return np.array(ids)


def sides(mesh, name, ring, top, bottom, mat, inside=False, material_offset=0):
    """Unwrap each vertical face in metres along its perimeter, with a seam at the first edge."""
    xy = top[ring, :2]
    travel = np.r_[0, np.cumsum(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1))]
    a = top[ring]
    b = bottom[ring]
    points = np.stack([a, b, np.roll(b, -1, axis=0), np.roll(a, -1, axis=0)], axis=1)
    u = np.column_stack([travel[:-1], travel[:-1], travel[1:], travel[1:]]) / travel[-1]
    v = 0.5 + (points[:, :, 2] - material_offset) / 0.22
    faces = (4 * np.arange(len(ring))[:, None, None] + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    mesh.add(name, points, faces[:, ::-1] if inside else faces, np.stack([u, v], axis=-1), mat)


def build(mesh, key, cfg, xyz, uv, faces, groove, mat):
    options = cfg['slab']
    full = np.asarray(Image.open(ROOT / 'calibration' / (key + '_outline.png')).convert('L')) > 0
    pitch = cfg['span'] / (len(xyz) - 1)
    cells, socket, support, source_pixels = block_cells(full, options, cfg['step'], pitch)
    cellfaces = faces.reshape(*cells.shape, 2, 3)
    slab_faces = cellfaces[cells].reshape(-1, 3)
    socket_faces = cellfaces[socket].reshape(-1, 3)
    in_socket = np.repeat(socket.ravel(), 2)
    host = xyz.reshape(-1, 3)
    coords = uv.reshape(-1, 2)
    mesh.add('Road adjoining the fracture', host, faces[~in_socket & ~groove], coords, mat, True)
    if (groove & ~in_socket).any():
        mesh.add('Fracture lip and exposed bed', host, faces[groove & ~in_socket], coords, mat)
    aggregate = source('fracture')
    # This exposed material is an appearance prior. Only the upper surface has scan observations.
    aggregate = np.asarray(Image.fromarray(aggregate).resize((512, 512), Image.Resampling.LANCZOS))
    exposed = mesh.material('Designed exposed aggregate', aggregate, 0.97, 0.0, 0.7)
    off = options['offset_m']
    floor_z = options['base_z_m']
    bottom_z = options['slab_bottom_z_m']
    slab_ids = np.unique(slab_faces)
    if (
        not np.isfinite([floor_z, bottom_z, 0 if off is None else off]).all()
        or not floor_z < bottom_z < host[slab_ids, 2].min()
        or (off is not None and bottom_z + off <= floor_z)
    ):
        raise ValueError('The plate must have positive thickness and fit above its foundation')
    slab_ring = boundary(slab_faces)
    socket_ring = boundary(socket_faces)
    floor = host.copy()
    floor[:, 2] = floor_z
    sides(mesh, 'Slab socket wall', socket_ring, host, floor, exposed, inside=True)
    floor_faces = cellfaces[socket if off is None else socket & ~support].reshape(-1, 3)
    mesh.add('Exposed slab foundation', floor, floor_faces, coords, exposed)
    values = np.full(len(host), -np.inf)
    outside_ids = np.unique(faces[~in_socket])
    values[np.unique(socket_faces)] = floor_z
    values[outside_ids] = host[outside_ids, 2]
    if off is not None:
        top = host.copy()
        top[:, 2] += off
        bottom = host.copy()
        bottom[:, 2] = bottom_z + off
        mesh.add('Detached pavement top', top, slab_faces, coords, mat)
        mesh.add('Detached pavement underside', bottom, slab_faces[:, ::-1], coords, exposed)
        sides(mesh, 'Detached pavement edges', slab_ring, top, bottom, exposed, material_offset=off)
        bearing = boundary(cellfaces[support].reshape(-1, 3))
        sides(mesh, 'Subgrade bearing', bearing, bottom, floor, exposed)
        mesh.add('Subgrade bearing surface', bottom, cellfaces[support].reshape(-1, 3), coords, exposed)
        values[slab_ids] = np.maximum(values[slab_ids], top[slab_ids, 2])
    if not np.isfinite(values).all():
        raise ValueError('Incomplete top surface')
    physical = np.zeros(len(host), bool)
    physical[np.unique(socket_faces)] = True
    face_z = host[slab_faces, 2].mean(1)
    metrics = dict(
        options,
        source_area_m2=source_pixels * cfg['source_pitch_m'] ** 2,
        area_m2=cells.sum().item() * pitch**2,
        socket_area_m2=socket.sum().item() * pitch**2,
        thickness_range_m=[
            float(host[slab_ids, 2].min() - bottom_z),
            float(host[slab_ids, 2].max() - bottom_z),
        ],
        volume_m3=float((face_z - bottom_z).sum() * pitch**2 / 2),
        grid_xy_extent_m=np.ptp(host[slab_ids, :2], axis=0).tolist(),
        top_faces=len(slab_faces),
        edge_segments=len(slab_ring),
        boundary_xy=host[slab_ring, :2].tolist(),
    )
    return values.reshape(xyz.shape[:2]), physical.reshape(xyz.shape[:2]), metrics
