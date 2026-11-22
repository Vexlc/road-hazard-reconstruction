"""Fit one RGB-D capture while reserving spatial blocks of its original ranges for testing."""

from dataclasses import replace

import cv2
import numpy as np
from scipy import ndimage

from pit_captures import project

RIM = 'Fracture inset rim'
DEBRIS_PREFIX = 'Fracture debris '


def split_depth(view):
    # Each 12x12 sensor tile withholds a 4x4 block. Vary its in-tile position to avoid
    # locking the holdout to a repeated image feature or to the reconstruction lattice.
    y, x = np.indices(view.depth.shape)
    tx, ty = x // 12, y // 12
    dx = (3 * tx + 7 * ty + 1) % 9
    dy = (5 * tx + 2 * ty + 4) % 9
    test = (x % 12 >= dx) & (x % 12 < dx + 4) & (y % 12 >= dy) & (y % 12 < dy + 4) & (view.depth > 0)
    return replace(view, depth=np.where(test, 0, view.depth)), test


def annotated_fragments(spec, view):
    if spec['image_size_px'] != list(view.rgb.shape[1::-1]):
        raise ValueError('Fragment annotations and the fitted camera must have matching dimensions')
    parts = spec['fragment_polygons_px']
    if not 1 <= len(parts) <= 254:
        raise ValueError('Expected 1 to 254 visible fragment outlines')
    labels = np.zeros(view.mask.shape, 'u1')
    for code, points in enumerate(parts, 1):
        poly = np.asarray(points)
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.issubdtype(poly.dtype, np.integer):
            raise ValueError('Expected a polygon in source pixel coordinates')
        if np.any(poly < 0) or np.any(poly >= view.rgb.shape[1::-1]):
            raise ValueError('Fragment outside the fitted image')
        area = cv2.fillPoly(np.zeros_like(labels), [poly.astype('i4')], 1).astype(bool)
        if area.sum() < 12 or np.any(area & ~view.mask) or np.any(labels[area]):
            raise ValueError(f'Fragment {code} is tiny, outside the pit, or overlaps another annotation')
        labels[area] = code
    return labels


def surface_regions(spec, view, field, frame):
    if 'rim_width_m' in spec:
        width = float(spec['rim_width_m'])
        if not np.isfinite(width) or width <= 0 or 'fragment_polygons_px' in spec:
            raise ValueError('Expected a positive rim width and one region definition')
        distance = ndimage.distance_transform_edt(field['defect_mask']) * frame['grid_pitch_m']
        labels = ((distance > 0) & (distance <= width)).astype('u1')
        names = [RIM]
    else:
        pixels = annotated_fragments(spec, view)
        height = field['height']
        axis = np.linspace(-0.5, 0.5, len(height)) * field['width']
        x, y = np.meshgrid(axis, -axis)
        world = (
            np.dstack([x, y, height]) @ np.array(frame['local_axes_in_anchor_camera']).T
            + frame['local_origin_in_anchor_m']
        )
        uv = project(world, view.K).astype('f4')
        labels = cv2.remap(pixels, uv, None, cv2.INTER_NEAREST)
        labels[~field['defect_mask']] = 0
        names = [DEBRIS_PREFIX + f'{i:02}' for i in range(1, int(pixels.max()) + 1)]
    if set(np.unique(labels)) != set(range(len(names) + 1)):
        raise ValueError('A labelled region did not reach the reconstructed surface')
    return labels, names
