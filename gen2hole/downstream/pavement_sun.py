"""Register sunlight on the intact road and trace it through the replacement opening."""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from prop_lighting import blocked_sun, linear, srgb


def reference(survey, basis):
    world = np.asarray(survey["sun_world"], float)
    endpoints = np.array([survey["shade_srgb"], survey["lit_srgb"]], float)
    if (
        world.shape != (3,)
        or endpoints.shape != (2, 3)
        or not np.isfinite(world).all()
        or not np.isfinite(endpoints).all()
        or np.linalg.norm(world) < 1e-6
        or endpoints.min() < 0
        or endpoints.max() > 255
    ):
        raise ValueError(
            "Expected a world-space direction towards the sun and two finite road sRGB triplets"
        )
    sun = world @ basis / np.linalg.norm(world)
    shade, lit = linear(endpoints)
    if sun[2] < 0.22 or (lit - shade).min() < 0.025:
        raise ValueError(
            "Sun survey needs sufficient elevation and two distinct samples of the same asphalt"
        )
    return sun, shade, lit


def smooth_grid(grid, observed):
    weights = gaussian_filter(observed.astype(float), 1.1, mode="constant")
    signal = gaussian_filter(np.where(observed, grid, 0.0), 1.1, mode="constant")
    return np.divide(signal, weights, out=np.zeros_like(signal), where=weights > 1e-8)


def fit_sun(survey, anchor, extent, source_rgb, batches):
    sun, shade, lit = reference(survey, anchor["basis_columns"])
    pitch = 0.05
    # The border adds space for the sun ray leaving a shaft at a different XY.
    half = np.ceil((extent / 2 + 1.2) / pitch).astype(int)
    shape = tuple((half * 2 + 1)[::-1])
    xy, visibility = [], []
    for points, rgb, direct in batches:
        if direct is None:
            delta = lit - shade
            # Project linear RGB onto the measured light/shade segment. Reject distinctly different materials.
            raw = (linear(rgb) - shade) @ delta / (delta @ delta)
            residual = linear(rgb) - shade - raw[:, None] * delta
            valid = np.abs(residual).max(axis=1) <= 0.13
            points, direct = points[valid], np.clip(raw[valid], 0, 1)
        xy.append(points)
        visibility.append(direct)
    points = np.rint(np.concatenate(xy) / pitch).astype(int) + half
    values = np.concatenate(visibility)
    keys = np.ravel_multi_index(points.T[::-1], shape)
    order = np.argsort(keys)
    unique, starts, counts = np.unique(
        keys[order], return_index=True, return_counts=True
    )
    grid = np.full(shape, np.nan)
    sorted_values = values[order]
    for key, start, count in zip(unique, starts, counts):
        if count >= 2:
            grid.flat[key] = np.median(sorted_values[start : start + count])
    observed = np.isfinite(grid)
    if observed.sum() < 20:
        raise ValueError("Not enough measured road cells to register sunlight")
    y, x = np.indices(shape)
    held = observed & ((x + y * 3) % 5 == 0)
    train = observed & ~held
    check = held & (distance_transform_edt(~train) <= 2.5)
    if not check.any():
        raise ValueError(
            "Sun survey lacks neighbouring road cells for a held-out check"
        )
    constant_error = float(np.mean(abs(grid[check] - np.median(grid[train]))))
    spatial_error = float(np.mean(abs(grid[check] - smooth_grid(grid, train)[check])))
    if spatial_error + 0.005 < constant_error:
        field = smooth_grid(grid, observed)
    else:
        field = np.full(shape, np.median(grid[observed]))
    source = np.median(linear(source_rgb), axis=0)
    if source.min() < 0.005:
        raise ValueError(
            "Intact reference apron too dark to determine daylight exposure"
        )
    return dict(
        method="surveyed-road-sun-v1",
        pitch_m=pitch,
        sun_local=sun.tolist(),
        ambient_rgb=(shade / lit).tolist(),
        daylight_gain_rgb=(lit / source).tolist(),
        visibility_u8=np.rint(255 * field).astype("u1").tolist(),
        confidence_u8=np.rint(
            255 * np.clip(1 - distance_transform_edt(~observed) / 4, 0, 1)
        )
        .astype("u1")
        .tolist(),
        cells=int(observed.sum()),
        heldout_cells=int(check.sum()),
        constant_mae=constant_error,
        spatial_mae=spatial_error,
    )


def shade_hits(mesh, source, xyz, normals, sky, model, fallback):
    sun = np.asarray(model["sun_local"])
    ambient = np.asarray(model["ambient_rgb"])
    gain = np.asarray(model["daylight_gain_rgb"])
    grid = np.asarray(model["visibility_u8"], "f4") / 255
    confidence = np.asarray(model["confidence_u8"], "f4") / 255
    pitch = float(model["pitch_m"])
    if (
        model.get("method") != "surveyed-road-sun-v1"
        or grid.ndim != 2
        or grid.shape != confidence.shape
        or sun.shape != (3,)
        or ambient.shape != (3,)
        or gain.shape != (3,)
        or pitch <= 0
        or not np.isfinite([*sun, *ambient, *gain, pitch]).all()
        or not np.isclose(np.linalg.norm(sun), 1)
        or sun[2] < 0.22
        or ambient.min() < 0
        or ambient.max() >= 1
        or gain.min() <= 0
        or not np.isfinite(grid).all()
        or grid.min() < 0
        or grid.max() > 1
        or not np.isfinite(confidence).all()
        or confidence.min() < 0
        or confidence.max() > 1
    ):
        raise ValueError("Invalid registered pavement sunlight")
    xy = xyz[:, :2] - xyz[:, 2:] * sun[:2] / sun[2]
    coords = (xy / pitch + (np.array(grid.shape[::-1]) - 1) / 2).T[::-1]
    weight = map_coordinates(confidence, coords, order=1, mode="constant")[:, None]
    direct = map_coordinates(grid, coords, order=1, mode="constant")
    if not mesh.measured_rim:
        cosine = np.maximum(normals @ sun, 0)
        active = (cosine > 0) & (direct > 0) & (weight[:, 0] > 0)
        if active.any():
            direct[active] *= 1 - blocked_sun(
                mesh, xyz[active] + 0.0004 * normals[active], sun
            )
        direct *= cosine / sun[2]
    light = ambient * sky[:, None] + (1 - ambient) * direct[:, None]
    return srgb(
        (1 - weight) * linear(fallback) + weight * linear(source) * gain * light
    )
