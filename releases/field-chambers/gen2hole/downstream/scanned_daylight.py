"""Relight a de-lit pavement scan from the measured host sun and adjacent road."""

import numpy as np
from prop_lighting import blocked_sun, linear, srgb


def irradiance(mesh, points, normals, anchor):
    fit = anchor["relief_daylight"]
    sun = np.asarray(fit["sun_world"]) @ np.asarray(anchor["basis_columns"])
    sun /= np.linalg.norm(sun)
    fraction = fit["direct_fraction"]
    incident = mesh.sky_gain(points, normals)
    if fraction:
        clear = 1 - blocked_sun(mesh, points + normals * .0004, sun)
        direct = clear * np.maximum(0, normals @ sun) / sun[2]
        incident = (1 - fraction) * incident + fraction * direct
    return incident


def apply(mesh, source, road, target, points, normals, anchor, local_light):
    incident = irradiance(mesh, points, normals, anchor)
    donor = np.maximum(np.median(linear(road), axis=0), 0.004)
    host = np.median(linear(target), axis=0)
    return srgb(linear(source) * (host / donor) * (incident * local_light)[:, None])
