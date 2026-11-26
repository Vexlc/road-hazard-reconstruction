"""Recover the local axes of an iron casting using its observed orthogonal stepped edges."""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent


class Camera:
    def __init__(self, s):
        self.s = s
        self.principal = np.array(s["size"]) / 2
        self.f = s["focal_mm_exif"] * s["size"][0] / s["focal_plane_width_model_mm"]
        self.height = s["camera_height_above_lid_model_m"]
        self.grade = s["reported_planned_overlay_m"]
        edge = np.asarray(s["right_outline_near_to_far"])
        lines = [(*s["front_edge"], 0), (*s["rear_edge_visible"], 0), (*s["near_left_visible"], 1)]
        lines.extend((a, b, 1 - i % 2) for i, (a, b) in enumerate(zip(edge, edge[1:])))

        def residual(t):
            camera_rotation, casting_axes = axes(t)
            errors = []
            for a, b, axis in lines:
                delta = np.diff(self.intersect([a, b], camera_rotation, self.height), axis=0)[0, :2]
                errors.append(
                    np.dot(delta / np.linalg.norm(delta), casting_axes[1 - axis])
                    * np.linalg.norm(np.subtract(a, b))
                )
            return np.r_[errors, (t[0] - 0.145) * 9, t[1] * 12]

        self.angles = least_squares(residual, [0.145, 0, -0.15], loss="soft_l1").x
        rotation, basis = axes(self.angles)
        self.rotation = rotation.copy()
        self.rotation[:, :2] = rotation[:, :2] @ basis.T
        self.eye = np.array([0.0, 0.0, self.grade + self.height])
        front = self.plane(s["front_edge"], self.grade)
        side = self.plane(edge, self.grade)
        self.width = np.linalg.norm(front[1] - front[0])
        self.length = side[-1, 1] - front[0, 1]
        self.eye[:2] -= front[0, :2] + [self.width / 2, self.length / 2]

    def intersect(self, pixels, rotation, height):
        rays = self.rays(pixels, rotation)
        return rays * (-height / rays[..., 2, None])

    def rays(self, pixels, rotation=None):
        pixels = np.asarray(pixels)
        optical = np.concatenate(
            [(pixels - self.principal) / self.f, np.ones((*pixels.shape[:-1], 1))], axis=-1
        )
        return optical @ (self.rotation if rotation is None else rotation)

    def plane(self, pixels, z):
        return self.eye + self.intersect(pixels, self.rotation, self.eye[2] - z)

    def project(self, world):
        optical = (world - self.eye) @ self.rotation.T
        return self.principal + self.f * optical[..., :2] / optical[..., 2, None]

    def profile(self):
        right = self.plane(self.s["right_outline_near_to_far"], self.grade)[:, :2]
        return np.vstack([right, right[::-1] * [-1, 1]])

    def ribs(self):
        # The observed rib feet extend forward of the face. Intersect the foot
        # rays with this vertical plane, not the surrounding pavement plane.
        y = -self.length / 2 - self.s["front_rib_base_projection_m"]
        foot = np.array(self.s["front_rib_pixels"])[:, 1, :]
        rays = self.rays(foot)
        return self.eye + rays * ((y - self.eye[1]) / rays[:, 1, None])


def axes(t):
    pitch, roll, yaw = t
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    right = np.array([1.0, 0.0, 0.0])
    down = np.array([0.0, -sp, -cp])
    rotation = np.array([cr * right + sr * down, -sr * right + cr * down, [0, cp, -sp]])
    return rotation, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])


def load():
    spec = json.loads((HERE / "survey.json").read_text())
    return spec, Camera(spec)
