"""Tangent frames for glTF normal textures, shared by mesh hits and Gaussian appearance samples."""

import numpy as np


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def tangent_mesh(attrs, faces, *, authored_v_up=False):
    """Separate mirrored UV charts and average U derivatives using the incident corner angles."""
    if "TANGENT" in attrs:
        return attrs, faces
    pos = attrs["POSITION"][faces].astype("f8")
    uv = attrs["TEXCOORD_0"][faces].astype("f8")
    if authored_v_up:
        # Poly Haven authors its OpenGL normal maps in Blender UV space. glTF image rows grow downward.
        uv[..., 1] = 1 - uv[..., 1]
    e1, e2 = pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0]
    u1, u2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
    sign = np.where(det < 0, -1, 1)
    tangent = unit((e1 * u2[:, 1:2] - e2 * u1[:, 1:2]) * sign[:, None])
    tangent[abs(det) < 1e-12] = 0
    # Vertex indices already separate discontinuous positions, normals and UVs. A mirrored
    # chart sharing a seam endpoint must get its own handedness at that position as well.
    keys, remap = np.unique(faces.astype("i8") * 2 + (sign[:, None] < 0), return_inverse=True)
    triangles = remap.reshape(faces.shape)
    split = {k: v[keys // 2] for k, v in attrs.items()}
    accumulated = np.zeros((len(keys), 3))
    for i in range(3):
        a = pos[:, (i + 1) % 3] - pos[:, i]
        b = pos[:, (i + 2) % 3] - pos[:, i]
        weight = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(1))
        np.add.at(accumulated, triangles[:, i], tangent * weight[:, None])
    n = unit(split["NORMAL"].astype("f8"))
    t = accumulated - n * (accumulated * n).sum(1)[:, None]
    degenerate = np.linalg.norm(t, axis=1) < 1e-9
    if degenerate.any():
        axis = np.eye(3)[np.argmin(abs(n[degenerate]), axis=1)]
        t[degenerate] = np.cross(axis, n[degenerate])
    split["TANGENT"] = np.float32(np.c_[unit(t), 1 - 2 * (keys % 2)])
    return split, triangles.astype("u4")


def mapped_normal(geometric, tangent, encoded, scale):
    """Return unit shading normals; input tangent.w preserves mirrored-chart handedness."""
    n = unit(geometric)
    t = tangent[:, :3] - n * (n * tangent[:, :3]).sum(1)[:, None]
    t = unit(t)
    b = np.cross(n, t) * np.sign(tangent[:, 3:4])
    tex = np.float32(encoded) * (2.0 / 255.0) - 1
    return np.float32(unit((t * tex[:, :1] + b * tex[:, 1:2]) * scale + n * tex[:, 2:3]))
