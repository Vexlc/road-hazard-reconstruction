"""Calibrate one view from the two visible circular iron edges and the bearing ledge."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "survey.json").read_text())
SIZE = np.array(SPEC["working_size"])
RADIUS = SPEC["mouth_radius_nominal_m"]
THICKNESS = SPEC["lid_thickness_assumed_m"]


def axes(pitch, roll):
    a, b = np.cos(pitch), np.sin(pitch)
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[c, -s * b, -s * a], [-s, -c * b, -c * a], [0, a, -b]])


def rays(p, focal, rotation):
    xy = (np.asarray(p) - SIZE / 2) / focal
    return np.concatenate([xy, np.ones_like(xy[..., :1])], axis=-1) @ rotation


def at_height(p, k, z):
    ray = rays(p, k[0], axes(*k[1:3]))
    return ray[..., :2] * ((z - k[3]) / ray[..., 2, None])


def circle(center, radius, z, n=256):
    theta = np.arange(n) * 2 * np.pi / n
    return np.column_stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            np.broadcast_to(z, (n,)),
        ]
    )


class Calibration:
    def __init__(self, fixed_f=None):
        self.mouth, self.lid = [np.array(SPEC["outline_px"][key]) for key in ("mouth", "lid")]
        self.train = [np.arange(len(p)) % 3 != 0 for p in (self.mouth, self.lid)]
        m, l = [p[mask] for p, mask in zip((self.mouth, self.lid), self.train)]

        def residual(k):
            return (
                np.r_[
                    np.linalg.norm(at_height(m, k, 0) - k[4:6], axis=1) - RADIUS,
                    np.linalg.norm(at_height(l, k, THICKNESS) - k[6:8], axis=1) - k[8],
                ]
                * 630
            )

        lo, hi = (300, 2000) if fixed_f is None else (fixed_f - 0.001, fixed_f + 0.001)
        self.k = least_squares(
            residual,
            [fixed_f or 800, 1.05, 0, 1.15, -0.2, 0.65, 0.30, 0.58, 0.30],
            bounds=([lo, 0.4, -1, 0.3, -3, -3, -3, -3, 0.23], [hi, 1.55, 1, 4, 3, 3, 3, 3, 0.4]),
            loss="soft_l1",
        ).x
        self.f = self.k[0]
        self.rotation = axes(*self.k[1:3])
        self.eye = np.r_[-self.k[4:6], self.k[3]]
        self.lid_xy = self.k[6:8] - self.k[4:6]
        self.lid_radius = self.k[8]
        # Both the far and near ledge arcs constrain the depth/radius ambiguity.
        obs = np.array(SPEC["ledge_back_and_front_px"])

        def bearing(v):
            return (np.linalg.norm(self.plane(obs, v[1]), axis=1) - v[0]) * 630

        self.ledge_radius, self.ledge_z = least_squares(
            bearing, [0.28, -0.022], bounds=([0.24, -0.08], [0.299, -0.002]), loss="soft_l1"
        ).x
        # Intersect rays to the ragged bottom edge with the vertical bore cylinder.
        ray = self.rays(SPEC["lower_iron_edge_far_arc_px"])
        a = (ray[:, :2] ** 2).sum(1)
        b = ray[:, :2] @ self.eye[:2]
        c = self.eye[:2] @ self.eye[:2] - self.ledge_radius**2
        dist = (-b + np.sqrt(b * b - a * c)) / a
        self.wall_samples = self.eye + ray * dist[:, None]

    def rays(self, p):
        return rays(p, self.f, self.rotation)

    def plane(self, p, z):
        return at_height(p, self.k, z) - self.k[4:6]

    def project(self, xyz):
        q = (xyz - self.eye) @ self.rotation.T
        return SIZE / 2 + self.f * q[..., :2] / q[..., 2, None]

    def edge_errors(self):
        errors = {}
        for name, points, mask, center, r, z in zip(
            ("mouth", "lid"),
            (self.mouth, self.lid),
            self.train,
            ([0, 0], self.lid_xy),
            (RADIUS, self.lid_radius),
            (0, THICKNESS),
        ):
            projected = self.project(circle(center, r, z, 16384))
            distances = cKDTree(projected).query(points[~mask])[0]
            errors[name] = dict(
                count=len(distances),
                rms_px=float(np.sqrt(np.mean(distances**2))),
                max_px=float(distances.max()),
            )
        return errors
