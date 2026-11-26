"""Fit the open rim and the independent silhouettes of three traffic cones."""

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())
SOURCE = HERE / "references/M571.jpg"
SIZE = np.asarray(S["working_size"])
RADIUS = S["working_diameter_nominal_m"] / 2
p0 = SIZE / 2
f = S["focal_mm"] * S["focal_plane_px_per_inch"] / 25.4 * SIZE[0] / S["original_width"]


def axes(k):
    p, r = k[:2]
    ap, bp = np.cos(p), np.sin(p)
    ar, br = np.cos(r), np.sin(r)
    return np.array([[ar, -br * bp, -br * ap], [-br, -ar * bp, -ar * ap], [0, ap, -bp]])


def rays(p, k):
    return np.concatenate(
        [(np.asarray(p) - p0) / f, np.ones((*np.shape(p)[:-1], 1))], axis=-1
    ) @ axes(k)


def plane(p, k, z=0):
    q = rays(p, k)
    return q[..., :2] * ((z - k[2]) / q[..., 2, None]) - k[3:5]


def proj(v, k):
    q = (v - np.r_[-k[3:5], k[2]]) @ axes(k).T
    return p0 + f * q[..., :2] / q[..., 2, None]


def circle(c, r, z, n=256):
    t = np.arange(n) * np.pi * 2 / n
    return np.c_[c + r * np.stack([np.cos(t), np.sin(t)], axis=-1), np.full(n, z)]


def fit():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != S["source_sha256"]:
        raise ValueError("Photograph checksum mismatch")
    a = cv2.resize(cv2.imread(str(SOURCE)), tuple(SIZE), interpolation=cv2.INTER_AREA)
    # Trace the road-to-iron upper lip; exclude the far-side break and the foreground cone.
    theta = np.radians(np.arange(0, 360, 4))
    base = np.stack([644 + 128 * np.cos(theta), 625 + 43 * np.sin(theta)], -1)
    norm = np.stack([np.cos(theta) / 128, np.sin(theta) / 43], -1)
    norm /= np.linalg.norm(norm, axis=1, keepdims=True)
    step = np.arange(-3, 3.1, 0.25)
    q = base[:, None, :] + norm[:, None, :] * step[None, :, None]
    g = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(float), (3, 3), 0.7)
    value = map_coordinates(g, q[..., ::-1].transpose(2, 0, 1), order=1)
    i = np.argmax(abs(np.gradient(value, axis=1))[:, 1:-1], axis=1) + 1
    points = q[np.arange(len(q)), i]
    points = points[
        ~(
            ((points[:, 0] > 724) & (points[:, 1] > 630))
            | ((points[:, 0] > 575) & (points[:, 0] < 603) & (points[:, 1] < 596))
            | (points[:, 0] > 766)
        )
    ]
    bases = np.array(S["cone_feet_px"])
    tops = np.array(S["cone_centers_top_px"])
    take = np.arange(len(points)) % 3 != 0

    def residual(k):
        vals = list((np.linalg.norm(plane(points[take], k), axis=1) - RADIUS) * 430)
        for poly, top, h in zip(bases, tops, k[5:]):
            # Midpoint of opposing sides in the ground plane locates the upright axis.
            center = plane(poly, k).mean(axis=0)
            pix = proj(np.array([[*center, h]]), k)[0]
            vals.extend(pix - top)
        return vals

    k = least_squares(
        residual,
        [0.26, 0, 0.9, 0.0, 2.8, 0.48, 0.48, 0.48],
        bounds=(
            [0.06, -0.12, 0.45, -0.6, 1.2, 0.15, 0.15, 0.15],
            [0.50, 0.12, 1.8, 0.6, 5.8, 0.85, 0.85, 0.85],
        ),
        loss="soft_l1",
    ).x
    edges = proj(circle([0, 0], RADIUS, 0, 12000), k)
    err = cKDTree(edges).query(points[~take])[0]
    hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (((h < 17) | (h > 172)) & (s > 110) & (v > 85)).astype(np.uint8)
    cones = []
    for j, (poly, top, height) in enumerate(zip(bases, tops, k[5:])):
        xy = plane(poly, k)
        center = xy.mean(0)
        # Independent silhouette rows above the base; guard against the left metal fragment.
        rows = np.arange(*[(446, 565, 5), (402, 525, 5), (469, 621, 5)][j])
        bounds = [(343, 431), (608, 683), (771, 859)][j]
        sides = np.array(
            [
                [np.flatnonzero(mask[y, bounds[0] : bounds[1]])[t] + bounds[0] for t in [0, -1]]
                for y in rows
            ]
        )
        y = np.r_[rows, rows]
        x = np.r_[sides[:, 0], sides[:, 1]]

        def outline(par):
            rb, rt, dx, dy, dz = par
            return cv2.convexHull(
                proj(
                    np.vstack(
                        [
                            circle(center, rb, 0.024, 128),
                            circle(center + [dx, dy], rt, height + dz, 128),
                        ]
                    ),
                    k,
                ).astype("f4")
            )[:, 0]

        def distance(par):
            poly = outline(par)
            return np.array(
                [cv2.pointPolygonTest(poly, (float(xx), float(yy)), True) for xx, yy in zip(x, y)]
            )

        training = np.tile(np.arange(len(rows)) % 3 != 0, 2)

        def body_fit(x):
            # The near cones are scuffed and lean away from their base centres. The visible
            # top ellipse constrains the horizontal bend; soft centring limits depth ambiguity.
            z = height + x[4]
            tip = proj(np.array([[*(center + x[2:4]), z]]), k)[0]
            return np.r_[distance(x)[training], (tip - top) * 2.5, x[2:4] * 24, x[4] * 24]

        result = least_squares(
            body_fit,
            [0.09, 0.024, 0.0003, 0.0003, 0.0003],
            bounds=([0.04, 0.005, -0.08, -0.08, -0.04], [0.20, 0.06, 0.08, 0.08, 0.04]),
            loss="soft_l1",
            diff_step=0.001,
        )
        tip_xy = center + result.x[2:4]
        z = height + result.x[4]
        hold = distance(result.x)[~training]
        side = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
        cones.append(
            dict(
                center_xy_m=center.tolist(),
                height_m=z,
                top_xy_m=tip_xy.tolist(),
                top_radius_m=result.x[1],
                base_body_radius_m=result.x[0],
                foot_length_m=side.tolist(),
                silhouette_held_out_px=int(len(hold)),
                silhouette_rms_px=float(np.sqrt(np.mean(hold**2))),
                top_reprojection_px=float(
                    np.linalg.norm(proj(np.array([[*tip_xy, z]]), k)[0] - top)
                ),
            )
        )
    return (
        k,
        cones,
        dict(
            aperture_held_out=len(err),
            aperture_rms_px=float(np.sqrt(np.mean(err**2))),
            aperture_max_px=float(err.max()),
        ),
    )
