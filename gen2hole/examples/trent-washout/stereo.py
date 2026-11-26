"""Estimate relative cameras from rigid, matched features. Report separate field landmarks."""

import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent


def camera(f, center):
    return np.array([[f, 0, center[0]], [0, f, center[1]], [0, 0, 1.0]])


def homogeneous(x):
    return np.column_stack([x, np.ones(len(x))])


class Pair:
    def __init__(self, s, features):
        self.s = s
        center = np.array(s["size"]) / 2
        self.center = center
        f0, f1 = s["nominal_focal_px"]
        a = np.array([p["west"] for p in features])
        b = np.array([p["northwest"] for p in features])
        # Exclude a neighbourhood in either photograph around every manual check.
        for p in s["landmarks"]:
            keep = (np.linalg.norm(a - p["a"], axis=1) > 9) & (
                np.linalg.norm(b - p["b"], axis=1) > 9
            )
            a, b = a[keep], b[keep]
        self.train_points = len(a)
        F, _ = cv2.findFundamentalMat(a, b, cv2.FM_8POINT)
        A = camera(f0, center)
        B = camera(f1, center)
        E = B.T @ F @ A
        _, R, t, _ = cv2.recoverPose(E, (a - center) / f0, (b - center) / f1, np.eye(3))
        t = t.ravel()
        rv = cv2.Rodrigues(R)[0].ravel()

        def decode(x):
            f = np.exp(x[5:])
            r = cv2.Rodrigues(x[:3])[0]
            yaw, el = x[3:5]
            v = np.array([np.cos(yaw) * np.cos(el), np.sin(yaw) * np.cos(el), np.sin(el)])
            return f, r, v

        def distances(x):
            f, r, t = decode(x)
            t0, t1, t2 = t
            cross = np.array([[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]])
            F = (
                np.linalg.inv(camera(f[1], center)).T
                @ cross
                @ r
                @ np.linalg.inv(camera(f[0], center))
            )
            l1 = homogeneous(a) @ F.T
            l0 = homogeneous(b) @ F
            return np.einsum("ij,ij->i", homogeneous(b), l1) / np.sqrt(
                np.sum(l0[:, :2] ** 2 + l1[:, :2] ** 2, axis=1)
            )

        def residual(x):
            return np.r_[distances(x), (x[5:] - np.log([f0, f1])) / 0.18]

        x0 = np.r_[rv, np.arctan2(t[1], t[0]), np.arcsin(t[2]), np.log([f0, f1])]
        opt = least_squares(residual, x0, loss="soft_l1", f_scale=1.1, max_nfev=500)
        self.f, self.R, self.t = decode(opt.x)
        self.training_median = float(np.median(abs(distances(opt.x))))
        self.P = [
            camera(self.f[0], center) @ np.c_[np.eye(3), np.zeros(3)],
            camera(self.f[1], center) @ np.c_[self.R, self.t],
        ]
        self.scale = 1.0
        self.origin = np.zeros(3)
        pitch = np.arctan((center[1] - s["west_horizon_y"]) / self.f[0])
        self.basis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -np.sin(pitch), np.cos(pitch)],
                [0.0, -np.cos(pitch), -np.sin(pitch)],
            ]
        )
        # Same surviving asphalt/grass corner in both views. Adopt a nominal 1.60 m
        # camera height over its grade. There is no measured scale or surveyed slope.
        corner = s["far_asphalt_grass_corner"]
        grade = self.triangulate([corner["west"]], [corner["northwest"]])[0]
        self.scale = s["west_camera_height_above_near_grade_m"] / -grade[2]
        self.origin = np.array([0.0, 0.0, s["west_camera_height_above_near_grade_m"]])

    def world(self, optical):
        return optical @ self.basis.T * self.scale + self.origin

    def project(self, world, view):
        opt = (np.asarray(world) - self.origin) @ self.basis / self.scale
        if view:
            opt = opt @ self.R.T + self.t
        return self.f[view] * opt[..., :2] / opt[..., 2:] + self.center

    def triangulate(self, a, b):
        q = cv2.triangulatePoints(
            self.P[0], self.P[1], np.asarray(a, float).T, np.asarray(b, float).T
        ).T
        return self.world(q[:, :3] / q[:, 3:])

    def eye(self, view):
        return self.world(-self.R.T @ self.t if view else np.zeros(3))

    def plane(self, px, z, view=0):
        q = np.asarray(px, float)
        r = np.concatenate([(q - self.center) / self.f[view], np.ones((*q.shape[:-1], 1))], axis=-1)
        if view:
            r = r @ self.R
        r = r @ self.basis.T
        eye = self.eye(view)
        return eye + r * (np.asarray(z)[..., None] - eye[2]) / r[..., 2:]

    def report(self):
        controls = self.s["landmarks"]
        a = np.array([p["a"] for p in controls])
        b = np.array([p["b"] for p in controls])
        v = self.triangulate(a, b)
        errors = np.linalg.norm(np.stack([self.project(v, 0) - a, self.project(v, 1) - b]), axis=-1)
        return dict(
            focal_px=self.f.tolist(),
            baseline_model_m=self.scale,
            camera_eyes_m=[self.eye(k).tolist() for k in (0, 1)],
            training_matches=self.train_points,
            training_median_sampson_px=self.training_median,
            landmarks={
                p["name"]: dict(world_m=pt.tolist(), reprojection_px=e.tolist())
                for p, pt, e in zip(controls, v, errors.T)
            },
        )


def load():
    s = json.loads((HERE / "survey.json").read_text())
    return Pair(s, json.loads((HERE / "correspondences.json").read_text()))


if __name__ == "__main__":
    print(json.dumps(load().report(), indent=2))
