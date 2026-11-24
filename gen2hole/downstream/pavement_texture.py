"""Filter photographed pavement in the footprint of each calibrated pixel on the hit triangle."""

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def pyramid(rgb):
    levels = [rgb]
    while min(levels[-1].shape[:2]) > 1:
        h, w = levels[-1].shape[:2]
        levels.append(cv2.resize(levels[-1], ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA))
    return levels


def ray_derivatives(directions):
    return [
        np.gradient(directions, axis=d) if directions.shape[d] > 1 else np.zeros_like(directions)
        for d in (1, 0)
    ]


def projected_axes(xyz, texcoords, rays, distance, derivatives, size, pixel_width, origin_deltas=None):
    e1, e2 = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    normal = np.cross(e1, e2)
    squared = np.maximum((normal * normal).sum(1), 1e-20)
    bary1 = np.cross(e2, normal) / squared[:, None]
    bary2 = np.cross(normal, e1) / squared[:, None]
    denominator = (rays * normal).sum(1)
    safe = np.where(abs(denominator) > 1e-12, denominator, np.copysign(1e-12, denominator))
    du1 = (texcoords[:, 1] - texcoords[:, 0]) * size
    du2 = (texcoords[:, 2] - texcoords[:, 0]) * size
    axes = []
    for axis, derivative in enumerate(derivatives):
        # Differentiate the intersection of a ray with this specific triangle's tangent plane.
        dt = (normal * derivative).sum(1) / safe
        delta = (distance * pixel_width)[:, None] * (derivative - rays * dt[:, None])
        if origin_deltas is not None:
            shift = origin_deltas[axis]
            dt_origin = (normal * shift).sum(1) / safe
            delta += pixel_width * (shift - rays * dt_origin[:, None])
        axes.append((delta * bary1).sum(1)[:, None] * du1 + (delta * bary2).sum(1)[:, None] * du2)
    return axes


def filtered(levels, uv, dx, dy, *, wrap=False):
    # Eigenaxes of the texture-space footprint. Cap anisotropy at four taps.
    a = dx[:, 0] ** 2 + dy[:, 0] ** 2
    c = dx[:, 1] ** 2 + dy[:, 1] ** 2
    b = dx[:, 0] * dx[:, 1] + dy[:, 0] * dy[:, 1]
    gap = np.hypot(a - c, 2 * b)
    major = np.sqrt(np.maximum((a + c + gap) / 2, 0))
    minor = np.sqrt(np.maximum((a + c - gap) / 2, 0))
    angle = np.arctan2(2 * b, a - c) / 2
    span = np.minimum(major, 2 * max(levels[0].shape[:2]))
    delta = np.column_stack((np.cos(angle), np.sin(angle))) * span[:, None] / levels[0].shape[1::-1]
    diameter = np.maximum.reduce([minor, major / 4, np.ones_like(major)])
    lod = np.clip(np.log2(diameter), 0, len(levels) - 1)
    low = np.floor(lod).astype(int)
    taps = np.clip(np.ceil(major / diameter), 1, 4).astype(int)
    total = np.zeros((len(uv), 3), "f4")
    for offset in (0, 1):
        indices = np.minimum(low + offset, len(levels) - 1)
        weights = (lod - low) if offset else (1 - (lod - low))
        for level in np.unique(indices[weights > 0]):
            selected = (indices == level) & (weights > 0)
            tex = levels[level]
            for tap in range(int(taps[selected].max())):
                active = selected & (taps > tap)
                shift = ((tap + 0.5) / taps[active] - 0.5)[:, None] * delta[active]
                coords = ((uv[active] + shift) * tex.shape[1::-1] - 0.5)[:, ::-1].T
                values = np.column_stack(
                    [
                        map_coordinates(ch, coords, order=1, mode="grid-wrap" if wrap else "nearest", output="f4")
                        for ch in tex.transpose(2, 0, 1)
                    ]
                )
                total[active] += values * (weights[active] / taps[active])[:, None]
    return np.rint(total).clip(0, 255).astype("u1")
