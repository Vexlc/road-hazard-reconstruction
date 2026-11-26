"""Full-sphere longitude/latitude cameras, with windows expressed in native pixel centres."""

import json
from pathlib import Path

import numpy as np


def spherical_rays(size, window, offset=(0.0, 0.0)):
    width, height = size
    left, top, cols, rows = window
    x = np.arange(cols, dtype=float) + left + 0.5 + offset[0]
    y = np.arange(rows, dtype=float) + top + 0.5 + offset[1]
    longitude = 2 * np.pi * ((x % width) / width - 0.5)
    latitude = np.pi * (0.5 - y / height)
    strip = np.cos(latitude)[:, None]
    rays = np.empty((rows, cols, 3), "f4")
    rays[..., 0] = strip * np.sin(longitude)
    rays[..., 1] = -np.sin(latitude)[:, None]
    rays[..., 2] = strip * np.cos(longitude)
    return rays, np.ones((rows, cols), bool)


def longitude_period(view, calibration):
    """Only a complete calibrated longitude sweep has adjacent left and right edges."""
    if calibration is None or Path(calibration).suffix.lower() in (".yaml", ".yml"):
        return None
    lens = json.loads(Path(calibration).read_text())
    if lens[0] == "EQUIRECTANGULAR":
        w = lens[1]
        if view.get("sensor_window_xywh", [0, 0, *lens[1:]])[2] == w:
            return int(w)
    return None
