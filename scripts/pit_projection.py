"""Ray-project road-aligned metric height fields into registered capture cameras."""

import numpy as np
import open3d as o3d


class SingleView:
    """A height field with the same mask-to-triangle rule as the published mesh."""

    def __init__(self, field):
        z = field['height']
        width = field['width']
        x = np.linspace(-width / 2, width / 2, len(z))
        xx, yy = np.meshgrid(x, -x)
        points = np.dstack((xx, yy, z)).reshape(-1, 3).astype('f4')
        ids = np.arange(z.size, dtype='u4').reshape(z.shape)
        a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
        tri = np.stack([np.stack([a, c, b], -1), np.stack([c, d, b], -1)], -2).reshape(-1, 3)
        is_pit = field['defect_mask'].ravel()[tri].mean(1) > 0.5
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.scene.add_triangles(o3d.core.Tensor(points), o3d.core.Tensor(tri))
        self.is_pit = is_pit

    def cast(self, eye, rays):
        result = self.scene.cast_rays(
            o3d.core.Tensor(np.ascontiguousarray(np.column_stack((eye, rays)), dtype='f4')), nthreads=2
        )
        depth = result['t_hit'].numpy()
        hits = np.isfinite(depth)
        parts = np.zeros(len(hits), 'u1')
        parts[hits] = 1 + self.is_pit[result['primitive_ids'].numpy()[hits]]
        return depth, parts, None


def camera_image(view, pose, surface, frame):
    basis = np.array(frame['local_axes_in_anchor_camera'])
    center = np.array(frame['local_origin_in_anchor_m'])
    eye = (pose[:3, 3] - center) @ basis
    rows, cols = view.mask.shape
    depth = np.empty((rows, cols), 'f4')
    label = np.zeros((rows, cols), bool)
    projection = np.linalg.inv(view.K).T @ pose[:3, :3].T @ basis
    for top in range(0, rows, 64):
        bottom = min(top + 64, rows)
        y, x = np.mgrid[top:bottom, 0:cols]
        directions = np.c_[x.ravel(), y.ravel(), np.ones(x.size)] @ projection
        dist, part, _ = surface.cast(np.broadcast_to(eye, directions.shape), directions)
        depth[top:bottom] = dist.reshape(x.shape)
        label[top:bottom] = part.reshape(x.shape) == 2
    return depth, label
