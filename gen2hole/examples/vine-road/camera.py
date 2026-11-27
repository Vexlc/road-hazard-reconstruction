"""Fit three engineering photographs against each other and stable survey control."""

from pathlib import Path
import json
import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
I = HERE / "inputs"
S = json.loads((HERE / "survey.json").read_text())
NAMES = ("ab", "ac", "cb")
ORDER = ((0, 1), (0, 2), (2, 1))


def read_npz(name):
    with np.load(I / (name + ".npz")) as p:
        return {k: p[k] for k in p.files}


class Camera:
    def __init__(self, q, size):
        self.q = q
        self.eye = q[3:6]
        self.R = cv2.Rodrigues(q[:3])[0]
        self.f = np.exp(q[6])
        self.K = np.array(
            [[self.f, 0, (size[0] - 1) / 2], [0, self.f, (size[1] - 1) / 2], [0, 0, 1]]
        )
        self.P = self.K @ np.c_[self.R, -self.R @ self.eye]

    def project(self, world):
        q = (world - self.eye) @ self.R.T @ self.K.T
        return q[..., :2] / q[..., 2:]

    def rays(self, uv):
        r = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(self.K).T @ self.R
        return r / np.linalg.norm(r, axis=1)[:, None]

    def at(self, uv, z):
        r = self.rays(uv)
        return self.eye + r * ((z - self.eye[2]) / r[:, 2])[:, None]


def cams(q):
    return [Camera(p, sz) for p, sz in zip(np.reshape(q, (3, 7)), S["image_size"])]


def tri(a, b, x, y):
    h = cv2.triangulatePoints(a.P, b.P, x.T, y.T).T
    return h[:, :3] / h[:, 3:]


def homogeneous(x):
    return np.c_[x, np.ones(len(x))]


def epi(a, b, x, y):
    R = b.R @ a.R.T
    t = b.R @ (a.eye - b.eye)
    F = np.linalg.inv(b.K).T @ np.cross(np.eye(3), t).T @ R @ np.linalg.inv(a.K)
    l = x @ F.T
    r = y @ F
    return (l * y).sum(1) / np.sqrt((l[:, :2] ** 2 + r[:, :2] ** 2).sum(1))


class Survey:
    def __init__(self):
        self.pairs = [read_npz(k) for k in NAMES]
        self.controls = [read_npz(k + "-control") for k in ("near", "wide", "third")]
        fields = read_npz("dated-fields")
        self.grid = {
            k: RegularGridInterpolator(
                (fields["yy"], fields["xx"]), fields[k], bounds_error=False, fill_value=None
            )
            for k in ("before", "after", "before_dist", "after_dist")
        }
        self.roads = np.array(S["intact_control_rows_cb"])
        ab, ac, cb = self.pairs
        e1, i = cKDTree(ac["u"]).query(ab["u"])
        e2, j = cKDTree(cb["v"]).query(ab["v"])
        near = (e1 < 2.2) & (e2 < 2) & (np.linalg.norm(ac["v"][i] - cb["u"][j], axis=1) < 3)
        near &= (ab["error"] < 1) & (ac["error"][i] < 1) & (cb["error"][j] < 1)
        self.triples = np.c_[
            0.5 * (ab["u"] + ac["u"][i]),
            0.5 * (ab["v"] + cb["v"][j]),
            0.5 * (ac["v"][i] + cb["u"][j]),
        ][near]
        self.reserved = (ab["hold"] | ac["hold"][i] | cb["hold"][j])[near]

    def height(self, k, xy):
        return self.grid[k](xy[..., :2][..., ::-1])

    def fit(self, priors=(4800, 2700, 2900), omit_near=False, road_half=None):
        pairs = []
        for (i, j), m in zip(ORDER, self.pairs):
            take = (m["error"] < 1) & ~m["hold"]
            pairs.append((i, j, homogeneous(m["u"][take]), homogeneous(m["v"][take])))
        t = self.triples[~self.reserved]
        cb = self.pairs[2]
        road = self.roads[~cb["hold"][self.roads]]
        if road_half is not None:
            road = road[road_half::2]

        def cost(q):
            cam = cams(q)
            a, b, c = cam
            err = [epi(cam[i], cam[j], x, y) / 0.65 for i, j, x, y in pairs]
            err.append((c.project(tri(a, b, t[:, :2], t[:, 2:4])) - t[:, 4:]).ravel() / 1.6)
            for i, (cc, src, pr) in enumerate(zip(cam, self.controls, priors)):
                use = ~src["hold"]
                if i == 0 and omit_near:
                    use = use & False
                err.extend(
                    [
                        (cc.project(src["xyz"][use]) - src["uv"][use]).ravel() / [5, 3.2, 3.8][i],
                        [np.log(cc.f / pr) / 0.55],
                    ]
                )
            xyz = tri(c, b, cb["u"][road], cb["v"][road])
            z = 0.5 * (self.height("before", xyz) + self.height("after", xyz))
            err.append((xyz[:, 2] - z) / 0.16)
            return np.concatenate(err)

        return cams(
            least_squares(
                cost, np.ravel(S["pose_initial"]), x_scale="jac", loss="soft_l1", max_nfev=250
            ).x
        )
