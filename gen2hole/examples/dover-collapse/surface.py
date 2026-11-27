"""Estimate the torn panel and expose the separately observed shadowed lip."""

import cv2
import numpy as np
from scipy.spatial import cKDTree
from cameras import S, calibrate, triangulate


def polygon(q, path):
    contour = np.float32(path)
    return np.array([cv2.pointPolygonTest(contour, tuple(v), False) >= 0 for v in q])


def lift_lip(cams):
    # The two photos see the upper break from opposite sides. Intersect each
    # reverse epipolar line with the ordered upper-edge trace in the other photo.
    top, toe = [np.array(S[k], float) for k in ["reverse_lip", "reverse_toe"]]
    trace = np.array(S["primary_lip"], float)
    c, d = cams
    rays = d.rays(top)
    a = c.project(d.eye + rays * 5)
    b = c.project(d.eye + rays * 10)
    lines = np.cross(np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))])
    off = lines @ np.c_[trace, np.ones(len(trace))].T
    first = []
    for row in off:
        roots = np.flatnonzero(row[1:] * row[:-1] <= 0)
        j = roots[0] if len(roots) else np.argmin(abs(row[:-1]))
        first.append(
            trace[j] + (trace[j + 1] - trace[j]) * np.clip(row[j] / (row[j] - row[j + 1]), 0, 1)
        )
    upper = triangulate(cams, first, top)
    # Intersect the lower shadow boundary with the local vertical plane along the lip.
    tangent = np.gradient(upper[:, :2], axis=0)
    normal = tangent[:, ::-1] * [1, -1]
    ray = d.rays(toe)
    dist = ((upper[:, :2] - d.eye[:2]) * normal).sum(1) / (ray[:, :2] * normal).sum(1)
    lower = d.eye + dist[:, None] * ray
    return upper, lower


def local_surface(xy, z, query):
    dis, ix = cKDTree(xy).query(query, k=12)
    delta = xy[ix] - query[:, None, :]
    A = np.concatenate([delta, np.ones((*ix.shape, 1))], axis=2)
    prior = 1 / (dis**2 + 0.035**2)
    w = prior.copy()
    for _ in range(3):
        T = A.transpose(0, 2, 1) * w[:, None, :]
        gram = T @ A
        gram[:, 0, 0] += 0.8
        gram[:, 1, 1] += 0.8
        coeff = np.linalg.solve(gram, T @ z[ix][..., None])[..., 0]
        err = abs((A * coeff[:, None, :]).sum(2) - z[ix])
        w = prior * np.minimum(1, 0.012 / np.maximum(err, 1e-7))
    return np.clip(coeff[:, 2], z[ix].min(1) - 0.008, z[ix].max(1) + 0.008), dis[:, 0]


def reconstruct(lane=None, focal_prior=650):
    cams, m = calibrate(lane, focal_prior)
    c, d = cams
    rest = np.array(S["reverse_rest"], float)
    top = np.vstack([S["reverse_lip"], rest])
    toe = np.vstack([S["reverse_toe"], rest])
    inside = polygon(m["px"][1], toe)
    use = m["train"] & inside & (m["error"] < 1.3)
    for k in ["reverse_pipe_mask", "occluded_end_mask"]:
        use &= ~polygon(m["px"][1], S[k])
    z = m["xyz"][use, 2]
    pts = d.at(m["px"][1][use])[:, :2]
    # Reject isolated gross matches against neighbouring pairs, before introducing edge controls.
    nearby = cKDTree(pts).query(pts, k=9)[1]
    keep = abs(z - np.median(z[nearby], axis=1)) < 0.055
    pts, z = pts[keep], z[keep]
    high, low = lift_lip(cams)
    outline_z, _ = local_surface(pts, z, d.at(toe)[:, :2])
    n = len(high)
    # The opposite lip is most legible across the open far break. Dark regions give an
    # optical wall hypothesis; retain the stereo-only deck as a separate product.
    outline_z[:n] = low[:, 2]
    upper_z = outline_z.copy()
    upper_z[:n] = high[:, 2]
    # Subdivide the field in the reverse source image. This keeps the near-vertical
    # shadow wall visible and avoids spanning it with an overhead height-field triangle.
    x = np.arange(int(top[:, 0].min()) - 1, int(top[:, 0].max()) + 2, 1.5)
    y = np.arange(int(top[:, 1].min()) - 1, int(top[:, 1].max()) + 2, 1.5)
    xx, yy = np.meshgrid(x, y)
    uv = np.stack([xx, yy], -1)
    q = uv.reshape(-1, 2)
    ring = []
    zring = []
    for i in range(len(toe)):
        j = (i + 1) % len(toe)
        count = max(1, int(np.ceil(np.linalg.norm(toe[j] - toe[i]) / 5)))
        t = np.arange(count) / count
        ring.extend(toe[i] + t[:, None] * (toe[j] - toe[i]))
        zring.extend(outline_z[i] + t * (outline_z[j] - outline_z[i]))
    flat = d.at(q)[:, :2]
    zq, _ = local_surface(np.vstack([pts, d.at(np.asarray(ring))[:, :2]]), np.r_[z, zring], flat)
    v = d.at(q, zq)
    mask = polygon(q, toe)
    return dict(
        cams=cams,
        matches=m,
        uv=uv,
        v=v,
        mask=mask,
        top=top,
        toe=toe,
        upper_z=upper_z,
        lower_z=outline_z,
        use=np.flatnonzero(use)[keep],
        inside=inside,
        source_pts=pts,
        source_z=z,
    )
