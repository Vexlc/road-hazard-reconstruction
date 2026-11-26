"""Refine the context-camera pose using feature pairs and the shared circular scale reference."""

from scipy.optimize import least_squares
import cv2
import numpy as np
from cameras import Camera, FOCAL, HERE, PP, S

K = np.array([[FOCAL, 0, PP[0]], [0, FOCAL, PP[1]], [0, 0, 1.0]])


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def triangulate(c0, c1, a, b):
    left = unit(c0.ray(a))
    right = unit(c1.ray(b))
    dot = (left * right).sum(-1)
    base = c1.eye - c0.eye
    t0 = (left @ base - dot * (right @ base)) / (1 - dot * dot)
    t1 = (dot * (left @ base) - right @ base) / (1 - dot * dot)
    p0 = c0.eye + left * t0[..., None]
    p1 = c1.eye + right * t1[..., None]
    return 0.5 * (p0 + p1), np.linalg.norm(p0 - p1, axis=-1)


def pose(pairs):
    c0 = Camera("detail")
    c1 = Camera("street")
    a, b, s = pairs.T[:2].T, pairs[:, 2:4], pairs[:, 4]
    inside = np.all(
        (a > 22) & (a < np.array(S["size"]) - 22) & (b > 22) & (b < np.array(S["size"]) - 22),
        axis=1,
    )
    valid = inside & (s > 0.12)
    split = np.arange(len(s)) % 5 != 0
    selected = valid & split
    cv2.setRNGSeed(211)
    F, ok = cv2.findFundamentalMat(a[selected], b[selected], cv2.USAC_MAGSAC, 1.6, 0.9999, 30000)
    train = np.flatnonzero(selected)[ok.ravel() > 0]
    aa, bb = a[train], b[train]
    ray0 = unit(c0.ray(aa))
    uv1 = np.column_stack(((bb - PP) / FOCAL, np.ones(len(bb))))
    disk = c0.plane(aa)
    # The web in the shared casting lies close to the rim plane. Use high-scoring
    # correspondences on it to establish translation scale; depth is free elsewhere.
    ring = (np.linalg.norm(disk[:, :2], axis=1) < 0.305) & (s[train] > 0.24)
    if ring.sum() < 10:
        raise ValueError("Insufficient matched casting features")
    r = cv2.Rodrigues(c1.R)[0]
    t = -c1.R @ c1.eye

    def install(q):
        c1.R = cv2.Rodrigues(q[:3])[0]
        c1.eye = -q[3:] @ c1.R

    def residual(q):
        install(q)
        base = unit(c1.eye - c0.eye)
        normal = np.cross(ray0, base) @ c1.R.T
        err = (
            (normal * uv1).sum(axis=1)
            / np.maximum(np.linalg.norm(normal[:, :2], axis=1), 1e-10)
            * FOCAL
        )
        return np.r_[err, ((c1.project(disk[ring]) - bb[ring]) * 0.47).ravel()]

    q = least_squares(
        residual,
        np.r_[r.ravel(), t.ravel()],
        loss="soft_l1",
        f_scale=1.25,
        x_scale="jac",
        ftol=1e-12,
    )
    install(q.x)
    xyz, gap = triangulate(c0, c1, a, b)
    e = np.maximum(
        np.linalg.norm(c0.project(xyz) - a, axis=1), np.linalg.norm(c1.project(xyz) - b, axis=1)
    )
    return (
        c0,
        c1,
        dict(
            train=train,
            heldout=np.flatnonzero(valid & ~split),
            xyz=xyz,
            gap=gap,
            errors=e,
            rim_matches=int(ring.sum()),
        ),
    )
