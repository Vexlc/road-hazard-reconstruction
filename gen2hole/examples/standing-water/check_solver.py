"""Analytic storage, separating ridges and a sloped inlet on a two-pit triangular patch."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipes/series'))
from mesh_io import grid_faces
from pit_water import solve


def check():
    # A single interior vertex lowered one metre creates six unit-grid
    # half-triangles. Capacity is one cubic metre. Below the brim, V scales as h³.
    h = np.zeros((9, 11))
    h[4, 3] = h[4, 7] = -1.0
    y, x = np.indices(h.shape)
    xyz = np.stack([x, -y, h], -1).reshape(-1, 3)
    faces = grid_faces(*h.shape)
    field = dict(height=h, defect_mask=h < 0)
    for liters, depth, retained in [(125.0, 0.5, 125.0), (1200.0, 1.0, 1000.0)]:
        v, f, wet, _, stats = solve(
            xyz, faces, field, dict(up_local=[0, 0, 1], seed_uv=[0.3, 0.5], input_liters=liters)
        )
        assert np.isclose(stats['retained_liters'], retained)
        assert np.isclose(stats['spilled_liters'], liters - retained)
        assert np.allclose(v[:, 2], depth - 1)
        assert wet[4, 3] and not wet[4, 7]
        assert np.all(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 2] > 0)
    h[4, 4] = -0.1
    field['defect_mask'] = h < 0
    xyz[:, 2] = h.ravel()
    _, _, wet, _, stats = solve(
        xyz, faces, field, dict(up_local=[0, 0, 1], seed_uv=[0.4, 0.5], input_liters=20)
    )
    assert stats['basin_seed_rc'] == [4, 3] and wet[4, 3] and not wet[4, 4]
    y, x = np.indices((9, 9))
    slope = .03 * x
    vertices, faces, wet, _, state = solve(
        np.stack([x, -y, slope], -1).reshape(-1, 3), grid_faces(*slope.shape),
        dict(height=slope, defect_mask=np.ones_like(slope)),
        dict(up_local=[0, 0, 1], seed_uv=[.5, .5], input_liters=5.), allow_drain=True,
    )
    assert not (len(vertices) or len(faces) or wet.any())
    assert state['retained_liters'] == 0 and state['spilled_liters'] == 5 and state['free_surface_m'] is None
    print('Analytic volume, overflow, inlet descent and isolated basins passed')


if __name__ == '__main__':
    check()
