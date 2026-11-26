"""Check the gravity projection of a shell-centroid proxy against its surveyed support polygon."""

import numpy as np
from scipy.spatial import ConvexHull

from mesh_transfer import fit_footprint, fit_road, footprint_size
from road_geometry import glb_triangles


def bearing_face(path):
    tris = np.concatenate([xyz[faces] for _, xyz, faces in glb_triangles(path, names=True)]).astype(float)
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    centroid = np.average(tris.mean(1), weights=area, axis=0)
    vertices = tris.reshape(-1, 3)
    feet = vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.0008]
    hull = ConvexHull(np.unique(feet[:, :2], axis=0))
    return centroid, hull.equations


def fit_balanced(folder, seed, points, heading, spec):
    centroid, edges = bearing_face(folder / "asset.glb")
    up = np.array([0.0, 0.0, 1.0])
    size = np.maximum(footprint_size(folder), 0.5)
    seat = fit_road(points, seed, heading, up_world=up, radius_m=1.05)
    basis = np.asarray(seat["basis_columns"])
    grade = (-up) @ basis
    angle = 0.0
    if spec.get("align_bore_to_grade") and np.linalg.norm(grade[:2]) > 0.005:
        axis = np.asarray(spec["pipe_ends"][0]["outward_local"])
        if abs(axis[2]) > 0.1:
            raise ValueError("Expected a horizontal pipe axis")
        angle = (
            np.arctan2(grade[1], grade[0]) - np.arctan2(axis[1], axis[0]) + np.pi / 2
        ) % np.pi - np.pi / 2
    forward = basis[:, :2] @ [np.cos(angle), np.sin(angle)]
    seat = fit_road(points, seed, forward, up_world=up, radius_m=1.05)
    seat = fit_footprint(seat, points, size)
    gravity = -np.asarray(seat["basis_columns"])[2]
    xy = centroid[:2] - centroid[2] * gravity[:2] / gravity[2]
    margin = float(-np.max(edges[:, :2] @ xy + edges[:, 2]))
    if margin <= 0:
        raise ValueError(f"{folder.name}: gravity projection outside support polygon ({margin:.4f} m)")
    evidence = dict(
        shell_centroid_local_m=centroid.tolist(),
        gravity_local=gravity.tolist(),
        projected_xy_m=xy.tolist(),
        support_margin_m=margin,
        yaw_adjustment_deg=np.degrees(angle),
        grade_ratio=float(np.linalg.norm(gravity[:2]) / abs(gravity[2])),
    )
    if spec.get("align_bore_to_grade"):
        axis = np.asarray(spec["pipe_ends"][0]["outward_local"])[:2]
        axis /= np.linalg.norm(axis)
        cross = abs(gravity[:2] @ [-axis[1], axis[0]])
        evidence["cross_axis_grade_ratio"] = float(cross / abs(gravity[2]))
    seat["gravity_support"] = evidence
    return seat
