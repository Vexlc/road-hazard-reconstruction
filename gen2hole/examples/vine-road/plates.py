"""Estimate each plate in source pixel space, keeping held-out image neighbourhoods out of the fit."""

import cv2
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import least_squares
from camera import S, ORDER, tri, read_npz

SCALE = 1021 / 720


def inside(uv, path):
    return np.array([cv2.pointPolygonTest(np.float32(path), tuple(q), True) for q in uv])


def occluded(uv):
    valid = np.zeros(len(uv), bool)
    for x, y, rx, ry in S["exclusions_720px"]:
        valid |= np.sum(((uv / SCALE - [x, y]) / [rx, ry]) ** 2, axis=1) <= 1
    return valid


def points(survey, cameras):
    pairs = []
    tests = []
    for name, (i, j), m in zip(("ab", "ac", "cb"), ORDER, survey.pairs):
        x = tri(cameras[i], cameras[j], m["u"], m["v"])
        err = np.maximum(
            np.linalg.norm(cameras[i].project(x) - m["u"], axis=1),
            np.linalg.norm(cameras[j].project(x) - m["v"], axis=1),
        )
        uv = m["u"] if i == 0 else cameras[0].project(x)
        valid = (m["error"] < 1) & (err < 2.6) & ~occluded(uv)
        held = valid & m["hold"]
        tests.append((x[held], uv[held], m["v"][held], j))
        pairs.append((x[valid & ~m["hold"]], uv[valid & ~m["hold"]], 2.5))
    # Keep withheld image patches out of both sparse and dense surface fitting.
    arr = np.vstack([test[1] for test in tests])
    tree = cKDTree(arr)
    for idx, src in enumerate(("ab", "ac")):
        rec = read_npz(src + "-patches")
        # Preserve the original height searched in the documented plane sweep.
        ref = cameras[idx + 1]
        world = ref.at(rec["v"], rec["xyz"][:, 2])
        uv = cameras[0].project(world)
        pairs.append((world, uv, 1.0 if idx == 1 else 0.7))
    xyz = np.vstack([x for x, _, _ in pairs])
    uv = np.vstack([u for _, u, _ in pairs])
    weights = np.concatenate([np.full(len(x), w) for x, _, w in pairs])
    allowed = (tree.query(uv)[0] > 5.5) & ~occluded(uv)
    for k, (i, j) in enumerate(ORDER[:2]):
        ref = cameras[j]
        pixel = ref.project(xyz)
        reserved = tests[k][2]
        allowed &= cKDTree(reserved).query(pixel, p=np.inf)[0] > 4.5
    return xyz[allowed], uv[allowed], weights[allowed], tests


class Panel:
    def __init__(self, cam, poly, points, weights):
        self.cam = cam
        self.poly = np.array(poly) * SCALE
        self.xy = points[:, :2]
        self.origin = np.median(self.xy, axis=0)
        A = np.c_[self.xy - self.origin, np.ones(len(points))]
        q = least_squares(
            lambda v: (A @ v - points[:, 2]) * np.sqrt(weights),
            [0, 0, np.median(points[:, 2])],
            loss="soft_l1",
            f_scale=0.14,
        ).x
        resid = points[:, 2] - A @ q
        self.slope = q[:2]
        self.intercept = q[2] - self.slope @ self.origin
        keep = abs(resid - np.median(resid)) < max(
            0.24, min(0.75, 3 * 1.4826 * np.median(abs(resid - np.median(resid))))
        )
        self.pts = points[keep]
        self.resid = resid[keep]
        self.tree = cKDTree(self.pts[:, :2])
        self.coverage = ConvexHull(self.pts[:, :2]).volume / max(
            0.001, cv2.contourArea(self.world(self.poly)[:, :2].astype("f4"))
        )

    def correction(self, xy):
        ds, ii = self.tree.query(xy, k=min(12, len(self.pts)))
        w = np.exp(-(ds**2) / (2 * 0.90**2))
        return np.sum(w * self.resid[ii], axis=1) / (w.sum(1) + 1.2)

    def world(self, uv):
        r = self.cam.rays(uv)
        n = np.r_[-self.slope, 1.0]
        base = self.intercept - self.cam.eye @ n
        v = self.cam.eye + r * (base / (r @ n))[:, None]
        for _ in range(3):
            v = self.cam.eye + r * ((base + self.correction(v[:, :2])) / (r @ n))[:, None]
        return v


def estimate(survey, cameras):
    xyz, uv, w, tests = points(survey, cameras)
    panels = {}
    counts = {}
    for name, poly in S["panels"].items():
        take = inside(uv / SCALE, poly) > 1.25
        if take.sum() < 6:
            counts[name] = dict(fit_rows=int(take.sum()))
            continue
        panel = Panel(cameras[0], poly, xyz[take], w[take])
        counts[name] = dict(
            fit_rows=int(take.sum()),
            retained=len(panel.pts),
            support_hull_fraction=float(panel.coverage),
            slope=panel.slope.tolist(),
        )
        if panel.coverage > 0.35 and np.linalg.norm(panel.slope) < 1.0:
            panels[name] = panel
    return panels, counts, tests
