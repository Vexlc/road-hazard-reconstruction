"""Single-photo calibration from upright cone axes and foreshortened square feet."""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())


class Camera:
    def __init__(self, height=None, focal=None):
        self.f = focal or S["focal_px"]
        self.h = height or S["cone_height_m"]
        self.c = (np.array(S["view_size"]) - 1) / 2
        quads = np.array([r["base"] for r in S["upright_cones"]])
        tips = np.array([r["top"] for r in S["upright_cones"]])

        def pose(q):
            pitch, roll, altitude = q
            up = np.array(
                [np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch), -np.sin(pitch)]
            )
            right = np.cross([0.0, 0.0, 1.0], up)
            right /= np.linalg.norm(right)
            self.axes = np.column_stack([right, np.cross(up, right), up])
            self.eye = np.array([0.0, 0.0, altitude])

        def error(q):
            pose(q)
            corners = self.plane(quads, 0.018 * self.h / 0.71)
            centres = corners.mean(1)
            pred = self.project(centres + [0, 0, self.h - centres[0, 2]])
            edges = np.roll(corners, -1, axis=1) - corners
            sizes = np.linalg.norm(edges, axis=2)
            # Pixel-equivalent square-side and right-angle residuals.
            scale = 240 * 0.71 / self.h
            orth = np.sum(edges * np.roll(edges, -1, axis=1), axis=2) / sizes
            return np.r_[
                (pred - tips).ravel(),
                ((sizes - sizes.mean(1)[:, None]) * scale).ravel(),
                (orth * scale).ravel(),
            ]

        fit = least_squares(
            error,
            [0.35, -0.03, 1.8],
            bounds=([0.12, -0.3, 0.6], [0.85, 0.3, 4.5]),
            loss="soft_l1",
            f_scale=3.0,
        )
        pose(fit.x)
        self.pose = fit.x
        self.eye -= self.plane(S["origin_px"])
        self.feet = self.plane(quads, 0.018 * self.h / 0.71)

    def rays(self, p):
        uv = (np.asarray(p) - self.c) / self.f
        d = np.concatenate([uv, np.ones_like(uv[..., :1])], axis=-1) @ self.axes
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def plane(self, p, z=0):
        d = self.rays(p)
        return self.eye + d * ((np.asarray(z) - self.eye[2]) / d[..., 2])[..., None]

    def project(self, p):
        d = (p - self.eye) @ self.axes.T
        return self.c + self.f * d[..., :2] / d[..., 2:]

    def closest_height(self, p, xy):
        d = self.rays(p)
        t = np.sum((xy - self.eye[:2]) * d[..., :2], axis=-1) / np.sum(d[..., :2] ** 2, axis=-1)
        return self.eye[2] + t * d[..., 2]
