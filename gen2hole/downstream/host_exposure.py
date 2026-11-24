"""Resolve calibrated, independently colored camera subrays in linear sRGB."""

import numpy as np

from prop_lighting import linear, srgb


def weights(view, shape):
    if "sample_weights" not in view:
        return None
    if "sample_rgb" not in view:
        raise ValueError("sample_weights requires registered sample_rgb")
    w = np.asarray(view["sample_weights"])
    if w.shape == (shape[0],):
        w = w[:, None, None]
    elif w.shape != shape:
        raise ValueError("Use S or S×H×W camera sampling weights")
    if w.dtype.kind not in "fiu" or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Camera weights must be finite and nonnegative")
    # Normalize relative to the largest value before summing, including for very large weights.
    peak = w.max(axis=0)
    if (peak == 0).any():
        raise ValueError("Every pixel must have at least one contributing optical sample")
    w = w.astype(np.float64) / peak
    return w / w.sum(axis=0)


def integrate(colors, normalized):
    if normalized is None:
        return srgb(linear(colors).mean(axis=0))
    return srgb((linear(colors) * normalized[..., None]).sum(axis=0))


def exposure(view, shape):
    normalized = weights(view, shape)
    if "sample_rgb" not in view:
        return None, normalized
    colors = np.asarray(view["sample_rgb"])
    rgb = np.asarray(view["rgb"])
    if (
        colors.shape != (*shape, 3)
        or colors.dtype != np.uint8
        or rgb.shape != colors.shape[1:]
        or rgb.dtype != np.uint8
    ):
        raise ValueError("Use S×H×W×3 uint8 sRGB samples aligned with the resolved uint8 RGB image")
    if np.max(np.abs(integrate(colors, normalized).astype(np.int16) - rgb)) > 1:
        raise ValueError("Resolved RGB disagrees with the weighted linear-sRGB exposure")
    return colors, normalized
