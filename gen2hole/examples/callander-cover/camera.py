"""Fit the lid plane from the four field-image corners and a nominal face width."""

from pathlib import Path
import json
import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
UNIT = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])


class Camera:
    def __init__(self, survey, focal):
        self.focal = focal
        self.centre = (np.array(survey["size_px"]) - 1) / 2
        self.K = np.array([[focal, 0, self.centre[0]], [0, focal, self.centre[1]], [0, 0, 1.0]])
        uv = np.array(survey["face_corners_px"])
        width = survey["nominal_face_width_m"]

        def surface(length):
            return np.column_stack((UNIT * [width, length], np.zeros(4)))

        ok, r, t = cv2.solvePnP(surface(width), uv, self.K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise ValueError("Cannot initialize camera")

        def errors(q):
            px = cv2.projectPoints(surface(q[6]), q[:3], q[3:6], self.K, None)[0][:, 0]
            return (px - uv).ravel()

        fit = least_squares(
            errors,
            np.r_[r.ravel(), t.ravel(), width],
            bounds=([-np.inf] * 6 + [width * 0.65], [np.inf] * 6 + [width * 1.6]),
        )
        self.R = cv2.Rodrigues(fit.x[:3])[0]
        self.eye = -self.R.T @ fit.x[3:6]
        self.length = float(fit.x[6])
        self.error = float(np.sqrt(np.mean(np.sum(errors(fit.x).reshape(-1, 2) ** 2, axis=1))))
        if self.eye[2] <= 0:
            raise ValueError("Back-facing solution")

    def rays(self, uv):
        uv = np.asarray(uv)
        q = (
            np.concatenate(((uv - self.centre) / self.focal, np.ones_like(uv[..., :1])), -1)
            @ self.R
        )
        return q / np.linalg.norm(q, axis=-1, keepdims=True)

    def plane(self, uv, z=0):
        ray = self.rays(uv)
        return self.eye + ray * ((np.asarray(z) - self.eye[2]) / ray[..., 2])[..., None]

    def project(self, points):
        cam = (np.asarray(points) - self.eye) @ self.R.T
        return self.centre + self.focal * cam[..., :2] / cam[..., 2:]

    def drop(self, pair):
        top = self.plane(pair[0])
        fit = least_squares(lambda q: self.project(top - [0, 0, q[0]]) - pair[1], [0.018])
        return float(fit.x[0])


def load():
    s = json.loads((HERE / "survey.json").read_text())
    return s, Camera(s, s["focal_px"])
