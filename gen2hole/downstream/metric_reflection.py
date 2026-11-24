"""Visible, finite-distance reflectors reconstructed from registered metric host samples."""

import numpy as np
import open3d as o3d

from host_depth import geometry, sampled
from prop_lighting import linear


class MetricReflectors:
    fields = ("reflector_vertices_world", "reflector_triangles", "reflector_srgb")

    def __init__(self, vertices, triangles, colors):
        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles)
        colors = np.asarray(colors)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not np.isfinite(vertices).all()
            or triangles.ndim != 2
            or triangles.shape[1:] != (3,)
            or not len(triangles)
            or triangles.dtype.kind not in "ui"
            or triangles.min() < 0
            or triangles.max() >= len(vertices)
            or colors.shape != vertices.shape
            or colors.dtype != np.uint8
        ):
            raise ValueError("Expected finite metric XYZ, indexed reflector triangles and uint8 vertex sRGB")
        self.origin = vertices.mean(0)
        self.vertices, self.triangles, self.colors = vertices, triangles.astype("u4"), colors
        self.linear_rgb = linear(colors)
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.scene.add_triangles(
            o3d.core.Tensor((vertices - self.origin).astype("f4")), o3d.core.Tensor(self.triangles)
        )

    @classmethod
    def from_packet(cls, packet):
        if not any(key in packet for key in cls.fields):
            return None
        if not all(key in packet for key in cls.fields):
            raise ValueError("Incomplete metric reflection geometry")
        return cls(*(packet[k] for k in cls.fields))

    @classmethod
    def merge(cls, members):
        if not members:
            raise ValueError("No observed reflecting surface")
        if len(members) == 1:
            return members[0]
        offsets = np.r_[0, np.cumsum([len(m.vertices) for m in members[:-1]])]
        return cls(
            np.concatenate([m.vertices for m in members]),
            np.concatenate([m.triangles + int(offset) for m, offset in zip(members, offsets)]),
            np.concatenate([m.colors for m in members]),
        )

    def packet(self):
        return dict(zip(self.fields, (self.vertices, self.triangles, self.colors)))

    def trace(self, points, rays):
        # Rebase before the float32 intersection to retain local detail in large map coordinates.
        eps = 2e-5
        origins = (points - self.origin) + eps * rays
        query = np.column_stack((origins, rays)).astype("f4")
        hits = self.scene.cast_rays(o3d.core.Tensor(query), nthreads=2)
        dist = hits["t_hit"].numpy() + eps
        facing = np.einsum("ij,ij->i", hits["primitive_normals"].numpy(), rays) < 0
        dist[~facing] = np.inf
        valid = np.isfinite(dist)
        tri = self.triangles[hits["primitive_ids"].numpy()[valid]]
        uv = hits["primitive_uvs"].numpy()[valid]
        bary = np.column_stack((1 - uv.sum(1), uv))
        color = np.zeros((len(rays), 3), dtype="f4")
        color[valid] = (self.linear_rgb[tri] * bary[..., None]).sum(1)
        return dist, color


def triangulate(points, colors, directions, labels, max_edge):
    """Connect measured neighbours of the same surface only; orient front faces toward their sensor."""
    if not np.isfinite(max_edge) or max_edge <= 0:
        raise ValueError("max_edge_m must be a positive metric distance")
    ids = np.arange(labels.size, dtype="i4").reshape(labels.shape)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    tri = np.concatenate((np.stack((a, b, c), -1).reshape(-1, 3), np.stack((b, d, c), -1).reshape(-1, 3)))
    group = labels.ravel()[tri]
    tri = tri[(group[:, 0] > 0) & (group == group[:, :1]).all(1)]
    return observed_faces(
        points.reshape(-1, 3), colors.reshape(-1, 3), directions.reshape(-1, 3), tri, max_edge
    )


def observed_faces(verts, colors, directions, tri, max_edge):
    v = verts[tri]
    edges = v[:, [1, 2, 0]] - v
    normal = np.cross(edges[:, 0], -edges[:, 2])
    twice_area = np.linalg.norm(normal, axis=1)
    direction = directions[tri].mean(1)
    incidence = np.einsum("ij,ij->i", normal, direction)
    # Reject grazing facets spanning sparse depth edges or a discontinuity.
    keep = (
        (np.linalg.norm(edges, axis=2).max(1) <= max_edge)
        & (twice_area > 1e-10)
        & (abs(incidence) > 0.15 * twice_area)
    )
    tri = tri[keep]
    if not len(tri):
        return None
    flip = incidence[keep] > 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    used, remapped = np.unique(tri, return_inverse=True)
    return MetricReflectors(verts[used], remapped.reshape(-1, 3), colors[used])


def from_depth(packet, stride, max_edge, calibration=None):
    if stride < 1 or "reflector_labels" not in packet:
        raise ValueError("Supply a positive sampling stride and registered reflector_labels")
    rays, dist, protected, pose, origins = geometry(packet, calibration, with_origins=True)
    labels = sampled(packet["reflector_labels"], dist.shape, "reflector_labels")
    rgb = np.asarray(packet["rgb"])
    if (
        labels.dtype.kind not in "biu"
        or (labels < 0).any()
        or rgb.dtype != np.uint8
        or rgb.shape != (*dist.shape[1:], 3)
    ):
        raise ValueError("Use integer reflector labels and aligned uint8 RGB")
    valid = np.isfinite(dist) & ~protected & (labels > 0)
    if "occluder_range" in packet:
        foreground = sampled(packet["occluder_range"], dist.shape, "occluder_range")
        valid &= ~(foreground < dist - 1e-5)
    members = []
    for i in range(len(rays)):
        sub = np.s_[::stride, ::stride]
        good = valid[i][sub]
        direction = rays[i][sub] @ pose[:3, :3].T
        points = direction * np.where(good, dist[i][sub], 0.0)[..., None] + pose[:3, 3]
        if origins is not None:
            points += origins[i][sub] @ pose[:3, :3].T
        mesh = triangulate(points, rgb[sub], direction, np.where(good, labels[i][sub], 0), max_edge)
        if mesh is not None:
            members.append(mesh)
    if not members:
        raise ValueError("No supported reflecting surface in the supplied depth and labels")
    return MetricReflectors.merge(members)
