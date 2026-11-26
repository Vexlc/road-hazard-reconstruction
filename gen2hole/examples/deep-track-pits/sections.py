"""Fit local section coordinates to the labelled depression and neighbouring raised boundary."""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components


def main_component(xy):
    tree = cKDTree(xy)
    graph = tree.sparse_distance_matrix(tree, 0.059, output_type="coo_matrix")
    _, labels = connected_components(graph, directed=False)
    return labels == np.bincount(labels).argmax()


class Section:
    def __init__(self, points, labels, kind):
        self.kind = kind
        bed = points[labels]
        if kind == "curved":
            # Mid-height returns isolate the exposed arc from its top and the pavement below.
            face = points[~labels & (points[:, 2] > 0.14) & (points[:, 2] < 0.21), :2]

            def residual(q):
                return np.linalg.norm(face - q[:2], axis=1) - q[2]

            fit = least_squares(residual, [0, 1, 0.9], loss="cauchy", f_scale=0.005)
            self.center = fit.x[:2]
            self.radius = fit.x[2]
        s, t = self.coordinates(bed[:, :2])
        bins = np.linspace(s.min(), s.max() + 1e-9, 16)
        xs = []
        ys = []
        for left, right in zip(bins[:-1], bins[1:]):
            use = (s >= left) & (s < right)
            if use.sum() >= 4:
                xs.append(s[use].mean())
                ys.append(np.percentile(t[use], [12, 88]).mean())
        self.s = np.asarray(xs)
        self.middle = PchipInterpolator(self.s, gaussian_filter1d(ys, 1.25, mode="nearest"))

    def coordinates(self, xy):
        if self.kind == "straight":
            return xy[:, 0], xy[:, 1]
        dx, dy = (xy - self.center).T
        return self.radius * np.arctan2(dx, -dy), np.hypot(dx, dy) - self.radius

    def xy(self, s, t):
        if self.kind == "straight":
            return np.c_[s, t]
        angle = s / self.radius
        return self.center + (self.radius + t)[:, None] * np.stack(
            [np.sin(angle), -np.cos(angle)], 1
        )

    def centerline(self, s):
        return self.middle(s.clip(self.s[0], self.s[-1]))

    def chart(self, v, h):
        s, t = self.coordinates(v[:, :2])
        return np.c_[s, (t - self.centerline(s)) * h / (h - v[:, 2])]

    def unproject(self, uv, z, h):
        s, p = uv.T
        t = self.centerline(s) + p * (h - z) / h
        return np.c_[self.xy(s, t), z]
