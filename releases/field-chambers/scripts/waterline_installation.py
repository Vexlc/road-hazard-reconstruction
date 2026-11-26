"""Seat a fixed observed shoreline on a matching measured grade and level its free surface."""

import numpy as np
from scipy.spatial.transform import Rotation

from pavement_layout import PavementLayout
from road_geometry import sha256


def install(recipe, models, key, base):
    layout = PavementLayout(recipe, models)
    ((row, path),) = layout.members
    if (
        row["asset"] != key
        or not (row.get("observed_water_interface") or row.get("terrain_pool_stage"))
        or any(row["xy_m"])
        or row["heading_deg"]
    ):
        raise ValueError("Use a single centred photographic pool")
    normal = np.asarray(layout.water_states[row["id"]]["up_local"])
    basis = np.asarray(base["basis_columns"])
    up = np.array([0.0, 0.0, 1.0])
    target = up @ basis
    yaw = np.arctan2(target[1], target[0]) - np.arctan2(normal[1], normal[0])
    basis = basis @ Rotation.from_rotvec(up * yaw).as_matrix()
    correction, _ = Rotation.align_vectors([up], [basis @ normal])
    installed = correction.as_matrix() @ basis
    delta = abs(installed[2, :2] - basis[2, :2]) @ (layout.meshes[key].extent / 2)
    if delta > 0.006:
        raise ValueError("Host grade is incompatible with the photographed shoreline")
    seat = dict(
        base,
        basis_columns=installed.tolist(),
        source_grade_rigid_adjustment_m=float(delta),
        surveyed_basis_columns=base["basis_columns"],
        gravity_up_world=up.tolist(),
        source_normal_world=(installed @ normal).tolist(),
        in_plane_yaw_deg=float(np.rad2deg(yaw)),
    )
    anchor = dict(
        seat,
        layout_sha256=layout.digest,
        instances=[
            dict(
                id=row["id"],
                asset=key,
                mesh_sha256=layout.meshes[key].digest,
                surface_sha256=sha256(path / "surface.npz"),
                anchor=seat,
            )
        ],
    )
    layout.bind(anchor)
    return layout, layout.meshes[key], anchor
