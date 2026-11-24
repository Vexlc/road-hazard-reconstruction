"""Calm water over a measured bed, viewed through a calibrated host's optical rays."""

from io import BytesIO

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from pavement_appearance import apply_apron
from prop_lighting import linear, srgb
from road_geometry import glb_triangles


class DirectionalProbe:
    """Observed sRGB along world-space bearings. Unknown directions have zero reflection support."""

    def __init__(self, rays, rgb, angle_deg):
        rays, rgb = np.asarray(rays), np.asarray(rgb)
        if (
            rays.ndim != 2
            or rays.shape[1:] != (3,)
            or not len(rays)
            or rgb.shape != rays.shape
            or rgb.dtype != np.uint8
            or not np.allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-4)
            or not 0 < angle_deg <= 45
        ):
            raise ValueError(
                "Expected unit world bearings, one sRGB triplet per bearing and a positive probe radius ≤45 degrees"
            )
        self.rays, self.rgb, self.angle_deg = rays, rgb, float(angle_deg)
        self.tree = cKDTree(rays)
        self.light = linear(rgb)
        self.chord = 2 * np.sin(np.deg2rad(angle_deg) * 0.5)

    @classmethod
    def from_packet(cls, view):
        keys = ("reflection_rays_world", "reflection_srgb", "reflection_radius_deg")
        if not any(k in view for k in keys):
            return None
        if not all(k in view for k in keys):
            raise ValueError("Incomplete directional reflection probe")
        return cls(view[keys[0]], view[keys[1]], float(view[keys[2]]))

    def packet(self):
        return dict(
            reflection_rays_world=self.rays,
            reflection_srgb=self.rgb,
            reflection_radius_deg=np.float64(self.angle_deg),
        )

    def sample(self, direction):
        distances, idx = self.tree.query(direction, k=4, distance_upper_bound=self.chord)
        weight = np.maximum(1 - distances / self.chord, 0) ** 2
        light = self.light[np.minimum(idx, len(self.light) - 1)]
        result = (light * weight[..., None]).sum(1) / np.maximum(weight.sum(1)[:, None], 1e-20)
        # Fade the last angular interval at an unobserved boundary.
        support = np.clip(2 * (1 - distances[:, 0] / self.chord), 0, 1)
        return result, support


def interface(incident, normal):
    cosine = np.clip(-incident @ normal, 0.0, 1.0)
    ior = 1.333
    transmitted_cos = np.sqrt(1 - (1 - cosine**2) / ior**2)
    reflected = incident + (2 * cosine)[:, None] * normal
    refracted = incident / ior + (cosine / ior - transmitted_cos)[:, None] * normal
    rs = (cosine - ior * transmitted_cos) / (cosine + ior * transmitted_cos)
    rp = (ior * cosine - transmitted_cos) / (ior * cosine + transmitted_cos)
    return reflected, refracted, 0.5 * (rs**2 + rp**2)


class PondBed:
    def __init__(self, folder, water):
        self.up = np.asarray(water["up_local"])
        self.bed = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.material = {}
        for part, xyz, triangles, uv, texture in glb_triangles(folder / "asset.glb", appearance=True):
            if part == 4:
                continue
            gid = self.bed.add_triangles(o3d.core.Tensor(xyz), o3d.core.Tensor(triangles))
            self.material[gid] = triangles, uv, np.asarray(Image.open(BytesIO(texture)).convert("RGB"))

    def trace(self, points, directions, appearance):
        eps = 2e-5
        starts = points + eps * directions
        ray = o3d.core.Tensor(np.column_stack((starts, directions)).astype("f4"))
        hit = self.bed.cast_rays(ray, nthreads=2)
        distance = hit["t_hit"].numpy() + eps
        gid, face, uv = (hit[k].numpy() for k in ("geometry_ids", "primitive_ids", "primitive_uvs"))
        result = np.zeros((len(points), 3), dtype=np.float32)
        for key, (tri, texcoords, texture) in self.material.items():
            valid = gid == key
            if not valid.any():
                continue
            bary = np.column_stack((1 - uv[valid].sum(1), uv[valid]))
            tex = (texcoords[tri[face[valid]]] * bary[:, :, None]).sum(1)
            size = np.array(texture.shape[1::-1])
            pos = tex * size - 0.5
            cell = np.floor(pos).astype(np.intp)
            dx, dy = (pos - cell).T[:, :, None]
            col, row = (cell % size).T
            col1, row1 = ((cell + 1) % size).T
            color = ((1 - dx) * texture[row, col] + dx * texture[row, col1]) * (1 - dy) + (
                (1 - dx) * texture[row1, col] + dx * texture[row1, col1]
            ) * dy
            xy = points[valid, :2] + directions[valid, :2] * distance[valid, None]
            result[valid] = linear(apply_apron(color, xy, appearance))
        return distance, result

    def shade(self, baked, pose, directions, origins, ranges, seat, probe):
        basis = np.asarray(seat["basis_columns"])
        local_ray = directions @ pose[:3, :3].T @ basis
        points = (pose[:3, 3] - seat["center_world"]) @ basis + local_ray * ranges[:, None]
        if origins is not None:
            points += origins @ pose[:3, :3].T @ basis
        reflected, transmitted, fresnel = interface(local_ray, self.up)
        env, support = probe.sample(reflected @ basis.T)
        base = linear(baked)
        appearance = seat.get("pavement_appearance")
        if appearance is not None:
            path, color = self.trace(points, transmitted, appearance)
            hit = np.isfinite(path)
            base[hit] = color[hit] * (0.74 * np.exp(-path[hit, None] * [7.0, 5.5, 4.5]))
            bank, material = self.trace(points, reflected, appearance)
            bank_hit = np.isfinite(bank)
            env[bank_hit], support[bank_hit] = material[bank_hit], 1.0
        fraction = (fresnel * support * ((local_ray @ self.up) < 0))[:, None]
        return srgb(base * (1 - fraction) + env * fraction)
