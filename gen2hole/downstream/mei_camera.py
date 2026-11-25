"""Unproject native omnidirectional MEI calibrations in raw sensor pixel coordinates."""

import json
from pathlib import Path

import cv2
import numpy as np


class MeiCamera:
    def __init__(self, size, projection, distortion, xi):
        self.size = tuple(size)
        if (
            len(size) != 2
            or any(type(x) is not int or x <= 0 for x in size)
            or not np.isfinite([*projection, *distortion, xi]).all()
            or min(projection[:2]) <= 0
            or xi < 0
        ):
            raise ValueError(
                "Invalid MEI image size, focal length or calibration coefficient"
            )
        self.fx, self.fy, self.cx, self.cy = projection
        self.k1, self.k2, self.p1, self.p2 = distortion
        self.xi = xi

    @classmethod
    def from_spec(cls, spec):
        p = spec["projection_parameters"]
        d = spec["distortion_parameters"]
        if spec["model_type"] != "MEI":
            raise ValueError("Expected the MEI projection model")
        if any(
            v != 0 for k, v in p.items() if k not in ("gamma1", "gamma2", "u0", "v0")
        ):
            raise ValueError("Expected a zero-skew MEI calibration")
        if any(v != 0 for k, v in d.items() if k not in ("k1", "k2", "p1", "p2")):
            raise ValueError("MEI distortion supports k1, k2, p1 and p2")
        return cls(
            (spec["image_width"], spec["image_height"]),
            [p[k] for k in ("gamma1", "gamma2", "u0", "v0")],
            [d[k] for k in ("k1", "k2", "p1", "p2")],
            spec["mirror_parameters"]["xi"],
        )

    @classmethod
    def read(cls, path):
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not storage.isOpened():
                raise ValueError("Cannot open MEI calibration")
            spec = {"model_type": storage.getNode("model_type").string()}
            for dim in ("image_width", "image_height"):
                node = storage.getNode(dim)
                if not node.isInt():
                    raise ValueError(f"Expected integer {dim}")
                spec[dim] = int(node.real())
            for name in (
                "projection_parameters",
                "mirror_parameters",
                "distortion_parameters",
            ):
                group = storage.getNode(name)
                if not group.isMap():
                    raise ValueError(f"Missing calibration section {name}")
                spec[name] = {}
                for key in group.keys():  # noqa: SIM118 - cv2.FileNode exposes keys() without iteration.
                    node = group.getNode(key)
                    if not (node.isReal() or node.isInt()):
                        raise ValueError(f"Expected numeric {key}")
                    spec[name][key] = node.real()
            return cls.from_spec(spec)
        finally:
            storage.release()

    def _distort(self, x, y):
        r2 = x * x + y * y
        radial = 1 + r2 * (self.k1 + self.k2 * r2)
        u = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        v = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return u, v, r2, radial

    def unproject(self, left, top, width, height, dx=0.0, dy=0.0):
        y, x = np.mgrid[top : top + height, left : left + width]
        u = (x + dx - self.cx) / self.fx
        v = (y + dy - self.cy) / self.fy
        x, y = u.copy(), v.copy()
        # Newton's method in the normalized mirror plane. A symmetric Brown
        # Jacobian avoids the outer-image fixed-point oscillation on KITTI lenses.
        for _ in range(20):
            px, py, r2, radial = self._distort(x, y)
            ex, ey = px - u, py - v
            slope = 2 * (self.k1 + 2 * self.k2 * r2)
            a = radial + x * x * slope + 2 * self.p1 * y + 6 * self.p2 * x
            b = x * y * slope + 2 * self.p1 * x + 2 * self.p2 * y
            c = radial + y * y * slope + 6 * self.p1 * y + 2 * self.p2 * x
            det = a * c - b * b
            stable = (det > 1e-12) & (a > 0)
            step_x = np.divide(c * ex - b * ey, det, out=np.zeros_like(x), where=stable)
            step_y = np.divide(a * ey - b * ex, det, out=np.zeros_like(y), where=stable)
            x -= step_x
            y -= step_y
            if max(np.max(abs(step_x)), np.max(abs(step_y))) < 1e-12:
                break
        px, py, r2, _ = self._distort(x, y)
        disc = 1 + (1 - self.xi * self.xi) * r2
        valid = (disc >= 0) & (np.hypot((px - u) * self.fx, (py - v) * self.fy) <= 1e-5)
        # Certify the inner, axis-connected distortion branch up to this radius.
        # The two radial Jacobian eigenvalues are quadratics in squared radius.
        lower = np.ones_like(r2)
        for linear, quadratic in ((self.k1, self.k2), (3 * self.k1, 5 * self.k2)):
            end = 1 + r2 * (linear + quadratic * r2)
            lower = np.minimum(lower, end)
            if quadratic > 0 and linear < 0:
                at = np.minimum(r2, -linear / (2 * quadratic))
                lower = np.minimum(lower, 1 + at * (linear + quadratic * at))
        valid &= lower > 6 * np.hypot(self.p1, self.p2) * np.sqrt(r2)
        scale = (self.xi + np.sqrt(np.maximum(disc, 0))) / (1 + r2)
        rays = np.stack([scale * x, scale * y, scale - self.xi], axis=-1)
        valid &= np.isfinite(rays).all(-1)
        rays[~valid] = [0, 0, 1]
        return rays.astype("f4"), valid


def calibrated_samples(path, packet):
    """Read a raw calibration and retain pixel registration through crops and subrays."""
    from mesh_transfer import pixel_rays

    if Path(path).suffix.lower() in (".yml", ".yaml"):
        lens = MeiCamera.read(path)
        size = lens.size
    else:
        lens = None
        params = json.loads(Path(path).read_text())
        size = tuple(params[1:3])
        if len(size) != 2 or any(
            type(dim) not in (int, float)
            or not np.isfinite(dim)
            or dim <= 0
            or dim != int(dim)
            for dim in size
        ):
            raise ValueError("Expected positive integer sensor dimensions")
        size = tuple(map(int, size))
    window = np.asarray(packet.get("sensor_window_xywh", [0, 0, *size]))
    if window.shape != (4,) or window.dtype.kind not in "iu":
        raise ValueError("sensor_window_xywh must contain four integer sensor pixels")
    left, top, width, height = map(int, window)
    if (
        min(left, top) < 0
        or min(width, height) < 1
        or left + width > size[0]
        or top + height > size[1]
        or ("rgb" in packet and packet["rgb"].shape[:2] != (height, width))
    ):
        raise ValueError(
            "Exposure and calibration window differ or lie outside the sensor"
        )
    offsets = np.asarray(packet.get("sample_offsets_px", [[0.0, 0.0]]))
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 2
        or not len(offsets)
        or offsets.dtype.kind not in "fiu"
        or not np.isfinite(offsets).all()
        or (np.abs(offsets.astype(float)) > 0.5).any()
    ):
        raise ValueError(
            "sample_offsets_px must give finite (dx,dy) within each native pixel"
        )
    rays, masks = [], []
    for dx, dy in offsets:
        if lens:
            direction, valid = lens.unproject(left, top, width, height, dx, dy)
        else:
            model, _, _, fx, fy, cx, cy, *dist = params
            cropped = [
                model,
                width,
                height,
                fx,
                fy,
                cx - left - dx,
                cy - top - dy,
                *dist,
            ]
            direction, valid = pixel_rays(cropped, return_valid=True)
        rays.append(direction)
        masks.append(valid)
    return np.stack(rays), np.stack(masks)
