"""Solve two camera poses and the displaced cover plane using road matches and circular castings."""

import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())
SIZE = np.array(S["working_image_size"])
RADIUS = S["assumed_outer_diameter_m"] / 2
C = (SIZE - 1) / 2
FOCAL = 27 * np.linalg.norm(SIZE) / np.hypot(36, 24)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Camera:
    def __init__(self, eye, axes, f):
        self.eye = np.asarray(eye)
        self.axes = np.asarray(axes)
        self.f = f

    def rays(self, p):
        p = (np.asarray(p) - C) / self.f
        return unit(np.concatenate((p, np.ones_like(p[..., :1])), -1) @ self.axes)

    def plane(self, p, z=0, normal=None):
        ray = self.rays(p)
        n = np.array([0, 0, 1.0]) if normal is None else normal
        return self.eye + ray * ((z - self.eye @ n) / (ray @ n))[..., None]

    def project(self, v):
        d = (v - self.eye) @ self.axes.T
        return C + self.f * d[..., :2] / d[..., 2:]

    def cylinder(self, p, r):
        v = self.rays(p)
        o = self.eye
        a = np.sum(v[..., :2] ** 2, axis=-1)
        b = v[..., :2] @ o[:2]
        disc = b * b - a * (o[:2] @ o[:2] - r * r)
        # Back face of bore observed through its top aperture.
        d = (-b + np.sqrt(np.maximum(disc, 0))) / np.maximum(a, 1.0e-12)
        return o + v * d[..., None]


def matrix_rot(v):
    return cv2.Rodrigues(np.asarray(v))[0]


def fit():
    with np.load(HERE / "matches.npz") as table:
        pairs = table["pairs"]
        train = table["train"]
    a, b = pairs.transpose(1, 0, 2)
    traces = [{k: np.array(t) for k, t in S[n]["outlines"].items()} for n in ("detail", "street")]
    road = np.ones(len(a), bool)
    cover = np.zeros(len(a), bool)
    for name in ("rim", "lid"):
        hull = traces[0][name].astype("f4")
        dist = np.array([cv2.pointPolygonTest(hull, tuple(map(float, p)), True) for p in a])
        road &= dist < -12
        if name == "lid":
            cover = dist > 8
    cv2.setRNGSeed(585)
    H, _ = cv2.findHomography(a[road & train], b[road & train], cv2.USAC_MAGSAC, 1.8)
    expected = cv2.perspectiveTransform(a[:, None], H)[:, 0]
    ground = road & (np.linalg.norm(b - expected, axis=1) < 2.8)
    K = np.array([[FOCAL, 0, C[0]], [0, FOCAL, C[1]], [0, 0, 1.0]])
    _, rot, pos, normal = cv2.decomposeHomographyMat(H, K)
    # Resolve the planar decomposition with the near-nadir image of the circular rim.
    index = np.argmax([q.ravel()[2] for q in normal])
    n = normal[index].ravel()
    right = unit([1, 0, 0] - n * n[0])
    up = -n
    basis = np.array([right, np.cross(up, right), up]).T
    near = Camera(np.array([0.0, 0.0, 1.0]), basis, FOCAL)
    xy = near.plane(traces[0]["rim"])[:, :2]
    mid = xy.mean(0)
    scale = RADIUS / np.linalg.norm(xy - mid, axis=1).mean()
    e0 = np.r_[-mid, 1] * scale
    e1 = e0 + (-rot[index].T @ pos[index]).ravel() @ basis * scale
    e = near.plane(traces[0]["lid"])[:, :2].mean(0)
    # Local origin at ring centre. The near camera has zero yaw about the world vertical.
    init = np.r_[
        e0,
        cv2.Rodrigues(basis)[0].ravel(),
        e1,
        cv2.Rodrigues(rot[index] @ basis)[0].ravel(),
        (e - mid) * scale,
        0.045,
        0,
        0,
        0.308,
        0.280,
        -0.018,
    ]

    def unpack(q):
        cams = [Camera(q[i : i + 3], matrix_rot(q[i + 3 : i + 6]), FOCAL) for i in (0, 6)]
        n = unit(np.r_[-q[15:17], 1.0])
        ctr = q[12:15]
        return cams, n, ctr

    # Discard repetitive crosshatch matches using the lid's distinct plane homography.
    HC, _ = cv2.findHomography(a[cover & train], b[cover & train], cv2.USAC_MAGSAC, 2.2)
    fixed = cover & (
        np.linalg.norm(b - cv2.perspectiveTransform(a[:, None], HC)[:, 0], axis=1) < 2.8
    )
    ring_train = np.arange(120) % 4 != 1

    def resid(q):
        (c0, c1), ln, lc = unpack(q)
        f = [
            (c1.project(c0.plane(a[ground & train])) - b[ground & train]).ravel(),
            (c1.project(c0.plane(a[fixed & train], lc @ ln, ln)) - b[fixed & train]).ravel(),
        ]
        for c, tr in zip((c0, c1), traces):
            for k, r, z, n, centre in [
                ("rim", RADIUS, 0, np.array([0, 0, 1.0]), np.zeros(3)),
                ("bore", q[18], q[19], np.array([0, 0, 1.0]), np.array([0, 0, q[19]])),
                ("lid", q[17], lc @ ln, ln, lc),
            ]:
                pv = c.plane(tr[k][ring_train], z, n)
                # Convert radial residual to approximate local image pixels.
                f.append(
                    (np.linalg.norm(pv - centre, axis=1) - r)
                    * c.f
                    / np.linalg.norm(pv - c.eye, axis=1)
                )
        return np.concatenate(f + [np.array([c0.axes[0, 1] * 200])])

    sol = least_squares(resid, init, loss="soft_l1", f_scale=1.3, x_scale="jac", max_nfev=150)
    q = sol.x
    cams, ln, lc = unpack(q)
    return cams, dict(
        parameters=q,
        lid_center=lc,
        lid_normal=ln,
        lid_radius=q[17],
        bore_radius=q[18],
        seat_z=q[19],
        traces=traces,
        ground=ground,
        cover=fixed,
        train=train,
        pairs=pairs,
    )


if __name__ == "__main__":
    cams, reg = fit()
    print(reg["parameters"])
    a, b = reg["pairs"].transpose(1, 0, 2)
    for name in ("ground", "cover"):
        mask = reg[name] & ~reg["train"]
        v = (
            cams[0].plane(a[mask])
            if name == "ground"
            else cams[0].plane(a[mask], reg["lid_center"] @ reg["lid_normal"], reg["lid_normal"])
        )
        print(
            name,
            mask.sum(),
            np.percentile(np.linalg.norm(cams[1].project(v) - b[mask], axis=1), [50, 90]),
        )
