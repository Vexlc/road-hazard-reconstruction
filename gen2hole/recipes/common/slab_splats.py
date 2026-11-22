"""Resolve pavement openings with closely spaced surfels on the two lips and thin vertical walls."""

import numpy as np
from scipy.spatial import cKDTree


def cap_samples(divisions):
    """Centroids of an equal-area triangular subdivision in barycentric coordinates."""
    rows, cols = np.indices((divisions, divisions))
    upright = rows + cols < divisions
    inverted = rows + cols < divisions - 1
    u = np.r_[rows[upright] + 1 / 3, rows[inverted] + 2 / 3] / divisions
    v = np.r_[cols[upright] + 1 / 3, cols[inverted] + 2 / 3] / divisions
    return np.stack([1 - u - v, u, v], axis=1)


def wall_samples(triangle, pitch):
    # A wall triangle has a vertical edge and a third vertex at the next perimeter corner.
    common = np.flatnonzero(np.all(triangle[:, :2] == np.roll(triangle[:, :2], -1, axis=0), axis=1))
    if len(common) != 1:
        raise ValueError('Expected a vertical edge in the slab wall')
    a = common[0]
    b, c = (a + 1) % 3, (a + 2) % 3
    width = np.linalg.norm(triangle[a, :2] - triangle[c, :2])
    height = abs(triangle[a, 2] - triangle[b, 2])
    columns = max(1, int(np.ceil(width / (pitch / 8))))
    u = (np.arange(columns) + 0.5) / columns
    heights = height * (1 - u)
    counts = np.maximum(1, np.ceil(heights / (pitch / 2))).astype(int)
    offset = np.r_[0, counts.cumsum()[:-1]]
    v = (np.arange(counts.sum()) - np.repeat(offset, counts) + 0.5) / np.repeat(counts, counts)
    u = np.repeat(u, counts)
    bary = np.empty((len(u), 3), dtype='f4')
    bary[:, a], bary[:, b], bary[:, c] = (1 - u) * (1 - v), (1 - u) * v, u
    scales = np.column_stack(
        [
            np.full(len(u), 0.70 * width / columns),
            0.78 * np.repeat(heights / counts, counts),
            np.full(len(u), pitch / 128),
        ]
    )
    return bary, scales


def refine(triangles, parts, active, pitch):
    wall = np.isin(parts, ['Slab socket wall', 'Detached pavement edges', 'Subgrade bearing'])
    lip = np.isin(parts, ['Slab socket wall', 'Detached pavement edges'])
    if not lip.any() or not wall.any():
        raise ValueError('No pavement socket in the supplied mesh')
    corners = np.unique(triangles[lip, :, :2].reshape(-1, 2), axis=0)
    distance = cKDTree(corners).query(triangles[:, :, :2].mean(axis=1))[0]
    cap = (~wall) & active & (distance <= 2 * pitch)
    cap_ids = np.flatnonzero(cap)
    divisions = 8
    pattern = cap_samples(divisions)
    faces = [np.repeat(cap_ids, len(pattern))]
    bary = [np.tile(pattern, (len(cap_ids), 1))]
    tangent_sigma = 0.74 * pitch / (divisions * np.sqrt(2))
    scales = [np.broadcast_to([tangent_sigma, tangent_sigma, pitch / 128], (len(faces[0]), 3))]
    for face in np.flatnonzero(wall & active):
        points, sigma = wall_samples(triangles[face], pitch)
        faces.append(np.full(len(points), face))
        bary.append(points)
        scales.append(sigma)
    return (
        cap | (wall & active),
        np.concatenate(faces),
        np.concatenate(bary).astype('f4'),
        np.concatenate(scales),
    )
