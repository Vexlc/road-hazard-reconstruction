"""Render unlit field textures and unit-ray first returns from the exported GLB."""

from io import BytesIO
import numpy as np
from PIL import Image
import open3d as o3d
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[3] / "releases/field-chambers/gen2hole/downstream/sensors"
    ),
)
from road_geometry import glb_triangles, part_id


class Renderer:
    def __init__(self, path, exclude=()):
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.parts = []
        for name, v, f, uv, data in glb_triangles(path, names=True, appearance=True):
            pid = part_id(name)
            if pid in exclude:
                continue
            self.scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
            self.parts.append((pid, f, uv, np.asarray(Image.open(BytesIO(data)).convert("RGB"))))

    def render(self, eye, rays):
        ray = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        packet = np.concatenate([np.broadcast_to(eye, ray.shape), ray], axis=-1).astype("f4")
        hits = self.scene.cast_rays(o3d.core.Tensor(packet), nthreads=2)
        t = hits["t_hit"].numpy()
        rgb = np.zeros((*t.shape, 3), "u1") + 34
        parts = np.zeros(t.shape, "u1")
        ids = hits["geometry_ids"].numpy()
        tri = hits["primitive_ids"].numpy()
        bc = hits["primitive_uvs"].numpy()
        for gid, (pid, f, uv, texture) in enumerate(self.parts):
            mask = ids == gid
            if not mask.any():
                continue
            p = bc[mask]
            weights = np.c_[1 - p.sum(1), p]
            q = (uv[f[tri[mask]]] * weights[:, :, None]).sum(1)
            ix = np.int32(np.clip(q * texture.shape[1::-1], 0, np.array(texture.shape[1::-1]) - 1))
            rgb[mask] = texture[ix[:, 1], ix[:, 0]]
            parts[mask] = pid
        return rgb, parts, t
