"""Register the ground plane and estimate the camera centre from the visible wall pairs."""
import cv2
import numpy as np
from scipy.optimize import least_squares


def transform(points, matrix):
    points = np.asarray(points, dtype=float)
    h = np.column_stack([points.reshape(-1, 2), np.ones(points.size // 2)]) @ matrix.T
    return (h[:, :2] / h[:, 2:]).reshape(points.shape)


class Projection:
    def __init__(self, spec):
        self.size = np.asarray(spec["extent_px"])
        corners = np.array([[-1,1],[1,1],[1,-1],[-1,-1]]) * np.array(spec["grade_extent_m"]) / 2
        self.forward = cv2.getPerspectiveTransform(np.float32(corners), np.float32(spec["grade_corners_px"]))
        self.inverse = np.linalg.inv(self.forward)
        pairs = transform(spec["wall_pairs_px"], self.inverse)
        upper, lower = pairs[:, 0], pairs[:, 1]
        def residual(params):
            ratio, x, y = params
            return (ratio * upper + (1-ratio) * np.array([x,y]) - lower).ravel()
        fit = least_squares(residual, [.8,0,0], bounds=([.50,-.4,-.4],[.99,.4,.4]),
                            loss="soft_l1", f_scale=.002)
        ratio, x, y = fit.x
        height = (ratio*spec["floor_depth_m"] - spec["seat_drop_m"]) / (1-ratio)
        self.eye = np.array([x,y,height])
        self.fit_errors_m = np.linalg.norm(residual(fit.x).reshape(-1,2), axis=1)

    def world(self, pixels, z):
        grade = transform(pixels, self.inverse)
        z = np.broadcast_to(z, grade.shape[:-1])
        xy = self.eye[:2] + (grade-self.eye[:2]) * (1-z[..., None]/self.eye[2])
        return np.concatenate([xy, z[..., None]], axis=-1)

    def pixels(self, points):
        p = np.asarray(points)
        grade = self.eye[:2] + (p[..., :2]-self.eye[:2]) / (1-p[..., 2:]/self.eye[2])
        return transform(grade, self.forward)

    def uv(self, points):
        return (.5+self.pixels(points))/self.size * [1,-1] + [0,1]
