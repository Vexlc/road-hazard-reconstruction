"""Estimate road-plane pose from two equal-height cones and five building uprights."""

from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent


class Camera:
    def __init__(self, s, focal, cone_height):
        self.focal = focal
        self.centre = (np.array(s["size_px"]) - 1) / 2
        self.bases = np.array(s["cone_base_px"])
        self.tips = np.array(s["cone_tip_px"])

        def set_frame(q):
            pitch, roll, height = q
            up = np.array(
                [np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch), -np.sin(pitch)]
            )
            right = np.cross([0.0, 0.0, 1.0], up)
            right /= np.linalg.norm(right)
            self.axes = np.column_stack((right, np.cross(up, right), up))
            self.eye = np.array([0.0, 0.0, height])

        def residual(q):
            set_frame(q)
            top = self.plane(self.bases[:2]) + [0, 0, cone_height]
            # Point-to-image-line error for the five architecturally vertical edges.
            a, b = np.asarray(s["uprights_top_bottom_px"]).transpose(1, 0, 2)
            up = self.axes[:, 2]
            direction = up[:2] * focal - (b - self.centre) * up[2]
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            signed = (a - b)[:, 0] * direction[:, 1] - (a - b)[:, 1] * direction[:, 0]
            return np.r_[(self.project(top) - self.tips[:2]).ravel(), signed]

        fit = least_squares(
            residual,
            [0.12, 0, 1.5],
            bounds=([-0.1, -0.16, 0.45], [0.55, 0.16, 3.0]),
            loss="soft_l1",
        )
        set_frame(fit.x)
        self.upright_error_px = residual(fit.x)[4:].tolist()
        self.cone_error_px = np.linalg.norm(
            self.project(self.plane(self.bases) + [0, 0, cone_height]) - self.tips, axis=1
        ).tolist()
        self.eye -= self.plane(s["centre_px"])

    def rays(self, uv):
        p = np.asarray(uv, float)
        d = (
            np.concatenate(((p - self.centre) / self.focal, np.ones_like(p[..., :1])), -1)
            @ self.axes
        )
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def plane(self, uv, z=0):
        d = self.rays(uv)
        return self.eye + d * ((np.asarray(z) - self.eye[2]) / d[..., 2])[..., None]

    def project(self, p):
        d = (p - self.eye) @ self.axes.T
        return self.centre + self.focal * d[..., :2] / d[..., 2:]

    def lips(self, crest, toe):
        # Closest point on the vertical through each toe to its observed crest ray.
        ground = self.plane(toe)
        d = self.rays(crest)
        t = np.sum((ground[..., :2] - self.eye[:2]) * d[..., :2], axis=-1) / np.sum(
            d[..., :2] ** 2, axis=-1
        )
        return self.eye[2] + t * d[..., 2]


def load():
    s = json.loads((HERE / "survey.json").read_text())
    return s, Camera(s, s["working_focal_px"], s["working_cone_height_m"])
