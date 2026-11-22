"""Miter constant-width bands around a traced convex access opening and pose its rigid lid."""

import numpy as np


def corners_local(profile):
    return (np.asarray(profile["rim_uv"]) - 0.5) * np.asarray(profile["size_m"]) * [1, -1]


def edges(profile):
    xy = corners_local(profile)
    if np.cross(xy, np.roll(xy, -1, axis=0)).sum() < 0:
        xy = xy[::-1]
    tangents = np.roll(xy, -1, axis=0) - xy
    normals = tangents[:, [1, 0]] * [1, -1]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    distances = (normals * xy).sum(1)
    if min(distances) <= 0 or np.any(xy @ normals.T - distances > 0.000001):
        raise ValueError("Seat requires a convex contour around its origin")
    return normals, distances


def rings(profile, offsets):
    normals, distances = edges(profile)
    angles = [0, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
    for offset in offsets:
        limits = distances + offset
        if min(limits) <= 0:
            raise ValueError("Inward shelf closes the opening")
        for i in range(len(normals)):
            k = [i - 1, i]
            xy = np.linalg.solve(normals[k], limits[k])
            angles.append(np.arctan2(xy[1], xy[0]) % (2 * np.pi))
    theta = np.r_[np.unique(np.round(angles, 10)), 2 * np.pi]
    ray = np.c_[np.cos(theta), np.sin(theta)]
    dot = ray @ normals.T
    result = {}
    for offset in offsets:
        span = np.where(dot > 1e-10, (distances + offset) / np.maximum(dot, 1e-10), np.inf).min(axis=1)
        result[offset] = ray * span[:, None]
    return theta, result


def field_distance(profile, x, y):
    normals, distances = edges(profile)
    return np.maximum.reduce([x * n[0] + y * n[1] - b for n, b in zip(normals, distances)])


def panel_pose(profile, state):
    tilt = float(state.get("angle_deg", 0))
    yaw = float(state.get("yaw_deg", 0))
    if (
        not np.isfinite([tilt, yaw]).all()
        or not 0 <= tilt <= 90
        or any(state.get("offset_m", []))
        or (tilt and yaw)
    ):
        raise ValueError("Use a seated lid, planar rotation or one edge tilt")
    matrix = np.eye(4)
    if yaw % 360:
        a = np.deg2rad(yaw % 360)
        matrix[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        # The underside rests on the shelf until a corner reaches the raised frame.
        lid = (np.asarray(profile["lid_boundary_uv"]) - 0.5) * profile["size_m"] * [1, -1]
        moved = lid @ matrix[:2, :2].T
        if field_distance(profile, moved[:, 0], moved[:, 1]).max() > 0.002:
            matrix[2, 3] = profile["thickness_m"] + 0.0015
    elif tilt:
        xy = corners_local(profile)
        i = state["pivot_edge"]
        if not isinstance(i, int) or not 0 <= i < len(xy):
            raise ValueError("Invalid pivot edge")
        p, q = xy[i], xy[(i + 1) % len(xy)]
        axis = np.r_[q - p, 0.0]
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(tilt) * np.sign(np.cross(q - p, xy.mean(0) - p))
        skew = np.array([[0.0, 0.0, axis[1]], [0.0, 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        matrix[:3, :3] = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        pivot = np.r_[p, 0.0]
        matrix[:3, 3] = pivot - matrix[:3, :3] @ pivot
    return matrix
