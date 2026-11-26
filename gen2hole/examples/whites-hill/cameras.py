"""Register the opening at arm's length and the enclosure viewed from the lane."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares, minimize_scalar

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())


def photograph(key):
    p = HERE / "references" / (key + ".jpg")
    if hashlib.sha256(p.read_bytes()).hexdigest() != S["references"][key]["sha256"]:
        raise ValueError("Unexpected field photograph " + key)
    return np.asarray(Image.open(p).convert("RGB"))


def unit(a):
    return a / np.linalg.norm(a)


def line(p):
    mean = p.mean(0)
    n = np.linalg.svd(p - mean, full_matrices=False)[2][-1]
    return np.r_[n, -mean @ n]


class Cameras:
    def __init__(self, focal=480):
        self.f = focal
        self.K = np.array([[focal, 0, 240], [0, focal, 320], [0, 0, 1.0]])
        self.Kw = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1.0]])
        groups = {k: np.asarray(v, float) for k, v in S["near_edges"].items()}

        def lines(k):
            coords = {
                n: self.normalized(v[np.arange(len(v)) % 4 != 2], k) for n, v in groups.items()
            }
            ls = {n: line(q) for n, q in coords.items()}
            return ls, np.concatenate(
                [(q @ ls[n][:2] + ls[n][2]) * focal for n, q in coords.items()]
            )

        self.k = minimize_scalar(
            lambda k: np.mean(lines(k)[1] ** 2), bounds=(-0.22, 0.42), method="bounded"
        ).x
        ls, _ = lines(self.k)
        vx = unit(np.cross(ls["far"], ls["near"]))
        vy = unit(np.cross(ls["left"], ls["right"]))
        vx *= np.sign(vx[0])
        vy *= np.sign(-vy[1])
        # Closest pair of orthogonal directions by symmetric polar decomposition.
        u, _, v = np.linalg.svd(np.stack([vx, vy], axis=1), full_matrices=False)
        vx, vy = (u @ v).T
        self.R = np.column_stack([vx, vy, np.cross(vx, vy)])

        def intersect(a, b):
            p = np.cross(ls[a], ls[b])
            q = (p / p[2]) @ self.R
            return -q[:2] / q[2]

        fl, fr = intersect("far", "left"), intersect("far", "right")
        self.h = 1 / np.linalg.norm(fr - fl)
        tri = np.stack([fl, fr, intersect("near", "left"), intersect("near", "right")]) * self.h
        self.center = tri.mean(0)
        self.eye = np.r_[-self.center, self.h]
        self.size = np.ptp(tri, axis=0)
        # The close-up contains three feet, while the fourth is outside the crop.
        self.guard_height = S["guard_height_assumed_m"]
        cube = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float
        )
        targets = np.asarray(S["wide_guard_fit"])
        _, r, t = cv2.solvePnP(cube, targets, self.Kw, None, flags=cv2.SOLVEPNP_SQPNP)

        def wide_fit(p):
            q = cube * [p[6], p[7], self.guard_height]
            return (cv2.projectPoints(q, p[:3], p[3:6], self.Kw, None)[0][:, 0] - targets).ravel()

        fit = least_squares(wide_fit, np.r_[r.ravel(), t.ravel(), 0.9, 0.9], loss="soft_l1")
        params = fit.x
        self.wide_error = np.linalg.norm(wide_fit(params).reshape(-1, 2), axis=1)
        self.wR = cv2.Rodrigues(params[:3])[0]
        self.weye = -self.wR.T @ params[3:6]
        W, L = params[6:]
        self.wguard = np.array([[0, 0], [W, 0], [W, L], [0, L]])
        n = self.plane(S["near_guard_contacts"], 0)[:, :2]
        w = self.wguard[[3, 2, 0]]
        nn = n - n.mean(0)
        ww = w - w.mean(0)
        u, _, v = np.linalg.svd(nn.T @ ww)
        self.rot = u @ v
        if np.linalg.det(self.rot) < 0:
            raise ValueError("Mirrored guard registration")
        self.scale = np.sum((nn @ self.rot) * ww) / np.sum(nn * nn)
        self.offset = w.mean(0) - self.scale * n.mean(0) @ self.rot
        self.registration_error = np.linalg.norm(
            self.scale * n @ self.rot + self.offset - w, axis=1
        )
        self.eye *= self.scale
        self.size *= self.scale
        self.guard = (self.wguard - self.offset) @ self.rot.T
        self.wide_eye = np.r_[(self.weye[:2] - self.offset) @ self.rot.T, self.weye[2]]
        self.holdout = np.concatenate(
            [
                abs(self.normalized(v[2::4], self.k) @ ls[k][:2] + ls[k][2]) * focal
                for k, v in groups.items()
            ]
        )

    def normalized(self, uv, k):
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        return cv2.undistortPoints(uv, self.K, np.array([k, 0.0, 0.0, 0.0, 0.0]))[:, 0]

    def rays(self, uv, wide=False):
        uv = np.asarray(uv, float).reshape(-1, 2)
        if wide:
            xyz = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(self.Kw).T @ self.wR
            return np.c_[xyz[:, :2] @ self.rot.T, xyz[:, 2]]
        q = self.normalized(uv, self.k)
        return np.c_[q, np.ones(len(q))] @ self.R

    def plane(self, uv, value, axis=2, wide=False):
        eye = self.wide_eye if wide else self.eye
        q = self.rays(uv, wide)
        return eye + q * ((value - eye[axis]) / q[:, axis, None])

    def project(self, xyz, wide=False):
        shape = xyz.shape[:-1]
        xyz = np.asarray(xyz).reshape(-1, 3)
        if wide:
            delta = xyz - self.wide_eye
            q = np.c_[delta[:, :2] @ self.rot, delta[:, 2]] @ self.wR.T
        else:
            q = (xyz - self.eye) @ self.R.T
        uv = q[:, :2] / q[:, 2, None]
        if not wide:
            uv *= 1 + self.k * (uv * uv).sum(axis=1, keepdims=True)
        return (uv * self.f + ([320, 240] if wide else [240, 320])).reshape(*shape, 2)

    def report(self):
        depths = self.plane(S["wet_wall_samples"], self.size[0] / 2, axis=0)[:, 2]
        return dict(
            focal_px=self.f,
            near_radial_k1=float(self.k),
            opening_xy_m=self.size.tolist(),
            waterline_on_right_wall_m=depths.tolist(),
            guard_height_assumed_m=self.guard_height,
            guard_foot_ties_error_m=self.registration_error.tolist(),
            near_edge_heldout_rms_px=float(np.sqrt(np.mean(self.holdout**2))),
            wide_corner_rms_px=float(np.sqrt(np.mean(self.wide_error**2))),
            near_eye_m=self.eye.tolist(),
            wide_eye_m=self.wide_eye.tolist(),
        )
