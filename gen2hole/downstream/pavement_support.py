"""Restrict the difference between two native renders to a surveyed pavement repair."""

import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, map_coordinates


class PavementSupport:
    """A metric transition around the measured opening, including solid geometry above grade."""

    def __init__(self, mesh, transition_m):
        if not np.isfinite(transition_m) or transition_m <= 0:
            raise ValueError("Expected a positive road repair width in metres")
        self.mesh = mesh
        self.width = transition_m
        self.step = mesh.extent[::-1] / (np.array(mesh.cut.shape) - 1)
        self.padding = np.ceil(transition_m / self.step).astype(int) + 1
        padded = np.pad(mesh.cut, [(n, n) for n in self.padding])
        self.distance = distance_transform_edt(~padded, sampling=self.step)
        self.part = {gid: entry[0] for gid, entry in mesh.surfaces.items()}

    def weights(self, origins_world, rays_world, anchor):
        basis = np.asarray(anchor["basis_columns"])
        origin = (origins_world - anchor["center_world"]) @ basis
        origin = np.broadcast_to(origin, rays_world.shape)
        direction = rays_world @ basis
        t = np.divide(
            -origin[:, 2], direction[:, 2], out=np.full(len(direction), -1.0), where=direction[:, 2] < -1e-6
        )
        xy = origin[:, :2] + t[:, None] * direction[:, :2]
        rc = ((xy / self.mesh.extent * [1, -1] + 0.5) * (np.array(self.mesh.cut.shape[::-1]) - 1))[:, ::-1]
        distance = map_coordinates(
            self.distance, (rc + self.padding).T, order=1, mode="constant", cval=self.width
        )
        # Keep the lip fully supported; fade only over the surrounding road.
        w = 1 - np.clip((distance / self.width - 0.12) / 0.88, 0, 1)
        weight = np.where(t > 0, w * w * (3 - 2 * w), 0.0)
        # An exposed cover can overhang its seat. Trace the actual triangles for those rays.
        probe = np.flatnonzero(weight < 1)
        if len(probe):
            rays = np.c_[origin[probe], direction[probe]].astype("f4")
            hit = self.mesh.scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
            depth = hit["t_hit"].numpy()
            gid = hit["geometry_ids"].numpy()
            solid = np.isin(gid, [g for g, part in self.part.items() if part > 1])
            above = origin[probe, 2] + np.where(np.isfinite(depth), depth, 0.0) * direction[probe, 2] >= 0
            weight[probe[solid & above & np.isfinite(depth)]] = 1.0
        return weight
