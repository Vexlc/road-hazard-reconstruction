"""Triangulate measured steep surfaces in a road-aligned fan; keep acquisition limits open."""

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, cKDTree
from sections import Section, main_component


def local_fit(uv, z, k):
    d, ix = cKDTree(uv).query(uv, k=k)
    a = np.concatenate([uv[ix] - uv[:, None, :], np.ones((*ix.shape, 1))], axis=-1)
    w = np.exp(-0.5 * (d / np.maximum(0.004, d[:, -1:] * 0.56)) ** 2)
    base = w.copy()
    for i in range(3):
        at = a.transpose(0, 2, 1) * w[:, None, :]
        gram = at @ a
        gram[:, 0, 0] += 0.000004
        gram[:, 1, 1] += 0.000004
        sol = np.linalg.solve(gram, (at @ z[ix][..., None]))[:, :, 0]
        diff = abs((a * sol[:, None]).sum(2) - z[ix])
        clip = np.maximum(0.0015, 1.65 * np.median(diff, axis=1, keepdims=True))
        w = base * np.minimum(1.0, clip / np.maximum(diff, 1e-10))
    return np.clip(sol[:, 2], z[ix].min(1), z[ix].max(1))


def triangles(points, labels, shape, kind, h, k, blend):
    valid_labels = labels.copy()
    valid_labels[labels] = main_component(points[labels, :2])
    if kind == "bowl":
        sec = None
        keep = (abs(points[:, :2]) <= shape / 2 + 0.002).all(1)
        uv = points[keep, :2]
        v = points[keep]
        lab = valid_labels[keep]
    else:
        sec = Section(points, valid_labels, kind)
        _, t = sec.coordinates(points[:, :2])
        keep = valid_labels | (points[:, 2] < 0.025)
        if kind == "curved":
            keep &= valid_labels | (t >= -0.006)
        keep &= (abs(points[:, :2]) <= shape / 2 + 0.002).all(1)
        v = points[keep]
        lab = valid_labels[keep]
        uv = sec.chart(v, h)
    xy, ids = np.unique(uv.round(8), return_index=True, axis=0)
    v = v[ids]
    lab = lab[ids]
    z = v[:, 2] * (1 - blend) + local_fit(xy, v[:, 2], k) * blend if blend else v[:, 2]
    verts = np.c_[xy, z] if sec is None else sec.unproject(xy, z, h)
    f = Delaunay(xy).simplices.copy()
    tri = verts[f]
    # Bound gaps using physical 3D edge length and the local sampling density.
    spacing = cKDTree(v).query(v, k=3)[0][:, 2]
    edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    limit = np.minimum(0.12, np.maximum(0.077, np.max(spacing[f], axis=1) * 3.6))
    f = f[(edges <= limit[:, None]).all(1)]
    if sec is not None and kind == "curved":
        # Clockwise angle from the lower radial axis plus positive radial distance.
        f = f[:, ::-1]
    # Connect opposite labels at interpolated class boundaries, retaining direct source
    # vertices on both sides instead of assigning a whole mixed triangle by plurality.
    mixed = (lab[f].sum(1) > 0) & (lab[f].sum(1) < 3)
    same = f[~mixed]
    fm = f[mixed]
    singleton = np.where(lab[fm].sum(1)[:, None] == 1, lab[fm], ~lab[fm])
    j = singleton.argmax(1)
    a, b, c = np.take_along_axis(fm, (j[:, None] + np.arange(3)) % 3, axis=1).T
    edge = np.sort(np.concatenate((np.c_[a, b], np.c_[a, c])), axis=1)
    pair, rev = np.unique(edge, axis=0, return_inverse=True)
    ab, ac = np.split(len(verts) + rev, 2)
    verts = np.vstack((verts, verts[pair].mean(1)))
    tri = np.vstack([same, np.c_[a, ab, ac], np.c_[ab, b, c], np.c_[ab, c, ac]])
    parts = np.r_[lab[same[:, 0]], lab[a], lab[b], lab[b]]
    return verts, tri, parts, sec


def scene(v, f):
    mesh = o3d.t.geometry.RaycastingScene(nthreads=1)
    mesh.add_triangles(o3d.core.Tensor(v.astype("f4")), o3d.core.Tensor(f.astype("u4")))
    return mesh


def hits(mesh, xy):
    ray = np.zeros((*xy.shape[:-1], 6), "f4")
    ray[..., :2] = xy
    ray[..., 2] = 1
    ray[..., 5] = -1
    res = mesh.cast_rays(o3d.core.Tensor(ray), nthreads=1)
    return 1 - res["t_hit"].numpy(), res["primitive_ids"].numpy()
