"""Initial camera registrations from the two visible ellipses of the same round casting."""

from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())
PP = (np.array(S["size"]) - 1) / 2
FOCAL = S["focal_equivalent_mm"] * np.linalg.norm(S["size"]) / np.hypot(36, 24)


def photograph(name):
    p = HERE / "references" / (name + ".jpg")
    if hashlib.sha256(p.read_bytes()).hexdigest() != S["images"][name]["sha256"]:
        raise ValueError("Photograph checksum mismatch")
    return cv2.imread(str(p))


class Camera:
    def __init__(self, name):
        self.r = S["nominal_diameter_m"] / 2
        points = np.asarray(S["images"][name]["rim"])
        use = np.arange(len(points)) % 4 != 0

        def residual(k, pts):
            self.install(k)

            def conic(q):
                d = self.ray(q)
                h = d[:, 2, None] * self.eye[:2] - self.eye[2] * d[:, :2]
                return (h * h).sum(axis=1) - (d[:, 2] * self.r) ** 2

            gx = conic(pts + [0.5, 0]) - conic(pts - [0.5, 0])
            gy = conic(pts + [0, 0.5]) - conic(pts - [0, 0.5])
            return conic(pts) / np.maximum(np.hypot(gx, gy), 1e-14)

        center, axes, angle = cv2.fitEllipse(points.astype("f4"))
        angle_to_grade = np.arcsin(min(axes) / max(axes))
        pitch = angle_to_grade - np.arctan2(center[1] - PP[1], FOCAL)
        dist = FOCAL * self.r / (max(axes) * 0.5)
        self.install([pitch, 0, 1.0, 0, 0])
        vector = self.ray(center) * dist
        guess = [pitch, 0, -vector[2], -vector[0], -vector[1]]
        q = least_squares(
            lambda k: residual(k, points[use]),
            guess,
            bounds=([-0.32, -0.4, 0.18, -4, -12], [0.60, 0.4, 3, 5, -1]),
            loss="soft_l1",
            x_scale="jac",
            ftol=1e-12,
        )
        self.heldout = residual(q.x, points[~use])
        self.install(q.x)
        direction = self.plane(S["images"][name]["heading_anchor"])[:2]
        direction /= np.linalg.norm(direction)
        orient = np.eye(3)
        orient[:2, :2] = [[direction[0], -direction[1]], [direction[1], direction[0]]]
        self.eye = self.eye @ orient
        self.R = self.R @ orient

    def install(self, k):
        pitch, roll, h, x, y = k
        sp, cp = np.sin(pitch), np.cos(pitch)
        sr, cr = np.sin(roll), np.cos(roll)
        self.R = np.array([[cr, -sr * sp, -sr * cp], [-sr, -cr * sp, -cr * cp], [0, cp, -sp]])
        self.eye = np.array([x, y, h])

    def ray(self, p):
        q = (np.asarray(p) - PP) / FOCAL
        return np.concatenate([q, np.ones((*q.shape[:-1], 1))], axis=-1) @ self.R

    def plane(self, p, z=0):
        r = self.ray(p)
        return self.eye + r * ((np.asarray(z) - self.eye[2]) / r[..., 2])[..., None]

    def project(self, v):
        q = (v - self.eye) @ self.R.T
        return PP + FOCAL * q[..., :2] / q[..., 2, None]
