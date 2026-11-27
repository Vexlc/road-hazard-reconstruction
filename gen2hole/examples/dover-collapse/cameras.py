"""Calibrate two opposing views to photographed road lines at an explicit nominal scale."""

from pathlib import Path
import json, hashlib
import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())
SIZE = np.array(S["annotation_size_px"])
SCALE = np.array(S["source"][0]["size"]) / SIZE


def photo(j):
    path = HERE / "references" / ["carriageway.jpg", "reverse.jpg"][j]
    if hashlib.sha256(path.read_bytes()).hexdigest() != S["source"][j]["sha256"]:
        raise ValueError(path)
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def intr(f):
    c = (SIZE - 1) / 2
    return np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])


class Camera:
    def __init__(self, q):
        self.R = cv2.Rodrigues(q[:3])[0]
        self.eye = q[3:6]
        self.K = intr(np.exp(q[6]))
        self.P = self.K @ np.c_[self.R, -self.R @ self.eye]

    def rays(self, p):
        p = np.asarray(p)
        return (
            np.concatenate([p, np.ones((*p.shape[:-1], 1))], axis=-1)
            @ np.linalg.inv(self.K).T
            @ self.R
        )

    def at(self, p, z=0):
        r = self.rays(p)
        return self.eye + r * ((np.asarray(z) - self.eye[2]) / r[..., 2])[..., None]

    def project(self, x):
        q = (x - self.eye) @ self.R.T @ self.K.T
        return q[..., :2] / q[..., 2:]


def triangulate(cams, a, b):
    v = cv2.triangulatePoints(
        cams[0].P, cams[1].P, np.asarray(a, float).T, np.asarray(b, float).T
    ).T
    return v[:, :3] / v[:, 3:]


def calibrate(lane=None, focal_prior=650):
    if lane is None:
        lane = S["nominal_lane_mark_spacing_m"]
    with np.load(HERE / "matches.npz") as d:
        px = [d[k] / SCALE for k in ("native0", "native1")]
        hold = d["hold"]
        good = (d["error"] < 1.5) & ~hold
        epi_inlier = d["error"] < 1.5
    marks = np.array(S["grade_witnesses"], float)
    world = np.float64([[0, 0, 0], [4.3, 0, 0], [5.0, -3.3, 0], [-0.85, -3.3, 0]]) * (lane / 3.3)

    def initial(im):
        ok, r, t = cv2.solvePnP(world, im, intr(690), None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise ValueError("Grade initialization")
        return np.r_[r.ravel(), (-cv2.Rodrigues(r)[0].T @ t).ravel(), np.log(690)]

    a, b = [np.c_[p[good], np.ones(good.sum())] for p in px]

    def residual(q):
        cam = [Camera(z) for z in q.reshape(2, 7)]
        c, d = cam
        rr = d.R @ c.R.T
        tt = d.R @ (c.eye - d.eye)
        F = np.linalg.inv(d.K).T @ np.cross(np.eye(3), tt).T @ rr @ np.linalg.inv(c.K)
        f = a @ F.T
        g = b @ F
        errs = [(f * b).sum(1) / np.sqrt((f[:, :2] ** 2 + g[:, :2] ** 2).sum(1)) / 0.6]
        for j, c in enumerate(cam):
            for x, y, u, v, side in S["grade_lines"][j]:
                uv = np.array([[x, y], [u, v]])
                pts = c.at(uv)
                pts[:, 1] = side * lane
                errs.append((c.project(pts) - uv).ravel() / 2.0)
            for seg in S["verticals"][j]:
                p0, p1 = np.array(seg).reshape(2, 2)
                line = np.cross(np.r_[p0, 1], c.K @ c.R[:, 2])
                errs.append([line @ np.r_[p1, 1] / np.linalg.norm(line[:2]) / 2.5])
            errs.extend(
                [
                    (c.project(world[:1]) - marks[j, :1]).ravel() / 4,
                    [np.log(c.K[0, 0] / focal_prior) / 0.55],
                ]
            )
        for i in [1, 2, 3]:
            errs.append(
                (cam[1].project(cam[0].at(marks[0, i : i + 1])) - marks[1, i : i + 1]).ravel()
                / (12 if i == 3 else 6)
            )
        return np.concatenate(errs)

    q = least_squares(
        residual,
        np.concatenate([initial(p) for p in marks]),
        loss="soft_l1",
        x_scale="jac",
        max_nfev=500,
    )
    cams = [Camera(z) for z in q.x.reshape(2, 7)]
    v = triangulate(cams, *px)
    error = np.max([np.linalg.norm(c.project(v) - p, axis=1) for c, p in zip(cams, px)], 0)
    return cams, dict(xyz=v, px=px, train=good, hold=hold, epipolar=epi_inlier, error=error)
