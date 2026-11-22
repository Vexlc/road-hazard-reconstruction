"""Read the physical pit bed and its planimetric footprint from an exported GLB."""

from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'tools/asset_atlas'
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles, sha256


class PitBed:
    def __init__(self, path):
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.bed_ids = []
        self.annotation = None
        self.registration_mm = None
        diagnostic = path.with_name('depth_diagnostics.npz')
        if diagnostic.is_file():
            with np.load(diagnostic) as depth:
                self.annotation = depth['projected_mask'].astype('f4')
                self.axes = [depth[axis] for axis in ('x', 'y')]
                self.heights = depth['height']
            self.annotation_sha = sha256(diagnostic)
            if (
                self.annotation.shape != tuple(len(a) for a in self.axes[::-1])
                or self.heights.shape != self.annotation.shape
            ):
                raise ValueError('Annotation and measured height grid differ in ' + str(path))
        bounds = []
        for name, xyz, faces in glb_triangles(path, names=True):
            if name == 'Water surface':
                continue
            geom = self.scene.add_triangles(o3d.core.Tensor(xyz), o3d.core.Tensor(faces))
            if name.startswith('Fracture '):
                self.bed_ids.append(geom)
            if self.annotation is not None:
                sample = xyz[:: max(1, len(xyz) // 4096)]
                coords = self.grid_indices(sample[:, :2])
                indices = np.rint(coords).astype(int)
                if (
                    np.max(np.abs(coords - indices)) > 0.001
                    or indices.min() < 0
                    or np.any(indices >= np.asarray(self.heights.shape)[:, None])
                ):
                    raise ValueError('Capture grid and GLB vertices are not aligned: ' + str(path))
                error = 1000 * float(np.max(abs(self.heights[tuple(indices)] - sample[:, 2])))
                if error > 0.02:
                    raise ValueError('Capture heights and GLB differ: ' + str(path))
                self.registration_mm = max(self.registration_mm or 0, error)
            bounds.append((xyz.min(0), xyz.max(0)))
        self.low, self.high = np.min([b[0] for b in bounds], 0), np.max([b[1] for b in bounds], 0)
        self.start = float(self.high[2]) + 0.125
        self.kind = (
            'capture_annotation'
            if self.annotation is not None
            else ('bed_triangles' if self.bed_ids else 'height_intersection')
        )

    def grid_indices(self, xy):
        return np.stack(
            [(xy[..., i] - a[0]) / ((a[-1] - a[0]) / (len(a) - 1)) for i, a in enumerate(self.axes)][::-1]
        )

    def top(self, xy):
        rays = np.zeros((*xy.shape[:-1], 6), 'f4')
        rays[..., :2] = xy
        rays[..., 2] = self.start
        rays[..., 5] = -1
        hit = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
        z = self.start - hit['t_hit'].numpy()
        labeled = np.isin(hit['geometry_ids'].numpy(), self.bed_ids) if self.bed_ids else None
        if self.annotation is not None:
            weight = map_coordinates(self.annotation, self.grid_indices(xy), order=1, mode='constant', cval=0)
            labeled = (weight >= 0.5) & np.isfinite(z)
        return z, labeled

    @staticmethod
    def footprint(z, labeled, plane):
        return labeled if labeled is not None else np.isfinite(z) & (z < plane)
