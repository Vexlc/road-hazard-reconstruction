"""Fit the grate plane, visible edge thickness and aperture widths from one photograph."""

import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())


def rgb():
    path = HERE / "references/casting.png"
    if hashlib.sha256(path.read_bytes()).hexdigest() != S["crop_sha256"]:
        raise ValueError("Casting crop digest mismatch")
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


class Casting:
    def __init__(self, focal_scale=1.0, principal_shift=(0, 0)):
        photo = rgb()
        self.centres = (np.arange(8) - 3.5) * S["nominal_slot_pitch_m"]
        f = S["focal_mm"] * S["focal_plane_px_per_cm"] / 10 * focal_scale
        c = (
            (np.asarray(S["original_size"]) - 1) / 2
            - S["crop_xywh"][:2]
            + np.array(principal_shift)
        )
        self.K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
        px = np.vstack([S["top_px"], S["bottom_px"]]).astype(float)
        self.train = np.tile(~np.isin(np.arange(1, 9), S["heldout_slots_one_based"]), 2)

        def ends(span):
            return np.c_[
                np.tile(self.centres, 2), np.repeat([span / 2, -span / 2], 8), np.zeros(16)
            ]

        ok, r, t = cv2.solvePnP(
            ends(0.34)[self.train], px[self.train], self.K, None, flags=cv2.SOLVEPNP_IPPE
        )
        if not ok:
            raise ValueError("Grate pose initialization")

        def residual(q):
            p = cv2.projectPoints(ends(q[6]), q[:3], q[3:6], self.K, None)[0][:, 0]
            return (p[self.train] - px[self.train]).ravel()

        fit = least_squares(residual, np.r_[r.ravel(), t.ravel(), 0.34], x_scale="jac")
        self.r, self.t, self.span = fit.x[:3], fit.x[3:6], fit.x[6]
        self.R = cv2.Rodrigues(self.r)[0]
        self.H = self.K @ np.column_stack([self.R[:, :2], self.t])
        self.tip_errors = np.linalg.norm(self.project(ends(self.span)) - px, axis=1)
        self.outer = self.plane(S["outer_px"])
        traces = np.array(S["front_edge_traces_y_topx_toex"])
        use = ~np.isin(traces[:, 0], S["edge_holdout_rows"])

        def error(depth):
            top = self.plane(traces[:, [1, 0]])
            # Intersect each displaced local edge with the annotated image row.
            direction = self.plane(traces[:, [1, 0]] + [4.0, -3.2]) - top
            pairs = np.stack([top, top + direction], axis=1)
            v = self.project(np.c_[pairs.reshape(-1, 2), np.full(pairs.size // 2, -depth)]).reshape(
                -1, 2, 2
            )
            dxdy = np.diff(v, axis=1)[:, 0]
            pred = v[:, 0, 0] + (traces[:, 0] - v[:, 0, 1]) * dxdy[:, 0] / dxdy[:, 1]
            return pred - traces[:, 2]

        self.thickness = least_squares(
            lambda z: error(z[0])[use], [0.03], bounds=(0.005, 0.08), loss="soft_l1", f_scale=2.0
        ).x[0]
        self.edge_residual = error(self.thickness)
        self.half_width = []
        # Search outward across each edge in the rectified plane. Median across 21 cuts
        # reduces the influence of bright paint, a lodged stone and grain texture.
        gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype("f4")
        for x in self.centres:
            widths = []
            for side in (-1, 1):
                offsets = np.linspace(0.006, 0.016, 65)
                xx, yy = np.meshgrid(
                    x + side * offsets, np.linspace(-self.span * 0.38, self.span * 0.38, 21)
                )
                uv = (
                    self.project(np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)])
                    .reshape(*xx.shape, 2)
                    .astype("f4")
                )
                strip = cv2.remap(gray, uv, None, cv2.INTER_LINEAR)
                grad = gaussian_filter1d(strip, 3, axis=1, order=1)
                widths.append(np.median(offsets[grad[:, 4:-4].argmax(1) + 4]))
            self.half_width.append(widths)

    def project(self, v):
        return cv2.projectPoints(np.asarray(v, float), self.r, self.t, self.K, None)[0][:, 0]

    def plane(self, uv):
        return cv2.perspectiveTransform(
            np.asarray(uv, float).reshape(-1, 1, 2), np.linalg.inv(self.H)
        )[:, 0]

    def rings(self):
        rings = [self.outer]
        for x, (left, right) in zip(self.centres, self.half_width):
            r = (left + right) / 2
            cx = x + (right - left) / 2
            a = np.linspace(0, np.pi, 33)
            b = a + np.pi
            rings.append(
                np.vstack(
                    [
                        np.c_[cx + r * np.cos(a), self.span / 2 - r + r * np.sin(a)],
                        np.c_[cx + r * np.cos(b), -self.span / 2 + r + r * np.sin(b)],
                    ]
                )
            )
        for i, q in enumerate(rings):
            area = np.sum(q[:, 0] * np.roll(q[:, 1], -1) - q[:, 1] * np.roll(q[:, 0], -1))
            if (area > 0) != (i == 0):
                rings[i] = q[::-1]
        return rings
