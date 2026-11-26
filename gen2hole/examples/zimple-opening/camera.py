"""Recover a local ground plane from the field rim, retaining native photograph pixels."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())
CROP = HERE / "references/M582-crop.png"


def photograph():
    if hashlib.sha256(CROP.read_bytes()).hexdigest() != S["crop_sha256"]:
        raise ValueError("Field crop checksum mismatch")
    return cv2.cvtColor(cv2.imread(str(CROP)), cv2.COLOR_BGR2RGB)


def circle(radius, n=256):
    t = np.arange(n) * (2 * np.pi / n)
    return radius * np.column_stack((np.cos(t), np.sin(t)))


class Camera:
    def __init__(self, diameter=None):
        self.radius = (diameter or S["assumed_outer_diameter_m"]) / 2
        self.pp = np.array(S["source_size"]) / 2
        self.f = S["exif_focal_mm"] * np.array(S["exif_sensor_density_per_inch"]) / 25.4
        points = np.asarray(S["rim_fit"])

        # Signed normal distance to the projected conic. The radial plane residual
        # alone overweights the nearly horizontal edges at this grazing incidence.
        def residual(k):
            self.install(k)

            def ring_distance(p):
                return np.linalg.norm(self.plane(p)[..., :2], axis=-1) - self.radius

            gx = ring_distance(points + [0.5, 0]) - ring_distance(points - [0.5, 0])
            gy = ring_distance(points + [0, 0.5]) - ring_distance(points - [0, 0.5])
            return ring_distance(points) / np.hypot(gx, gy)

        fit = least_squares(
            residual,
            [-0.09, -0.02, 1.15, -0.76, -5.9],
            bounds=([-0.35, -0.22, 0.4, -2, -9], [0.2, 0.22, 2.5, 0, -3]),
            loss="soft_l1",
            x_scale="jac",
            ftol=1e-12,
        )
        self.install(fit.x)
        curve = self.project(np.column_stack((circle(self.radius, 16000), np.zeros(16000))))
        tree = cKDTree(curve)
        self.fit_errors = tree.query(points)[0]
        self.hold_errors = tree.query(S["rim_heldout"])[0]

    def install(self, k):
        pitch, roll, height, x, y = k
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        self.R = np.array([[cr, -sr * sp, -sr * cp], [-sr, -cr * sp, -cr * cp], [0, cp, -sp]])
        self.eye = np.array([x, y, height])

    def ray(self, p):
        u = (np.asarray(p) - self.pp) / self.f
        return np.concatenate((u, np.ones((*u.shape[:-1], 1))), axis=-1) @ self.R

    def plane(self, p, z=0):
        d = self.ray(p)
        return self.eye + d * ((np.asarray(z) - self.eye[2]) / d[..., 2])[..., None]

    def project(self, v):
        d = (np.asarray(v) - self.eye) @ self.R.T
        return self.pp + self.f * d[..., :2] / d[..., 2, None]

    def upright_height(self, xy, py):
        a = (np.r_[xy, 0] - self.eye) @ self.R.T
        slope = (py - self.pp[1]) / self.f[1]
        return (slope * a[2] - a[1]) / (self.R[1, 2] - slope * self.R[2, 2])

    def cylinder(self, p, radius):
        d = self.ray(p)
        a = (d[..., :2] ** 2).sum(-1)
        b = d[..., :2] @ self.eye[:2]
        c = self.eye[:2] @ self.eye[:2] - radius**2
        disc = b * b - a * c
        if np.any(disc < 0):
            raise ValueError("Wall pick outside cylinder")
        return self.eye + d * ((-b + np.sqrt(disc)) / a)[..., None]
