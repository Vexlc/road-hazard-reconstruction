"""Check each object's contact geometry against an independent patch of measured pavement."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mesh_transfer import fit_footprint
from road_geometry import glb_triangles


def coherent_step(local, low, high):
    """Height jump repeated on two parallel, adjacent 10 cm cell edges."""
    patch = local[((local[:, :2] >= low - .1) & (local[:, :2] <= high + .1)).all(1)]
    xy, group, counts = np.unique(np.floor(patch[:, :2] / .10).astype(int), axis=0,
                                  return_inverse=True, return_counts=True)
    heights = {tuple(xy[i]): float(np.median(patch[group == i, 2])) for i in np.flatnonzero(counts >= 3)}
    maximum = 0.
    for x, y in heights:
        corners = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
        if all(c in heights for c in corners):
            a, b, c, d = (heights[k] for k in corners)
            for first, second in ((b - a, d - c), (c - a, d - b)):
                if first * second > 0:
                    maximum = max(maximum, min(abs(first), abs(second)))
    return maximum


def check_members(folder, anchor, points):
    folder = Path(folder)
    path = folder / "metadata.json"
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text())
    if not metadata.get("member_ground_check"):
        return None
    members = metadata["instances"]
    contacts = {row["id"]: [] for row in members}
    geometry_instances = []
    ids = list(contacts)
    for name, vertices, _ in glb_triangles(folder / "asset.glb", names=True):
        member = name.rsplit(" @", 1)[-1]
        contacts[member].append(vertices[vertices[:, 2] <= .006])
        geometry_instances.append(ids.index(member) + 1)
    basis = np.asarray(anchor["basis_columns"])
    origin = np.asarray(anchor["center_world"])
    # The full drive may contain millions of points. Only the local registered survey is needed.
    local = (points - origin) @ basis
    all_contacts = np.concatenate([v for vs in contacts.values() for v in vs])
    if not len(all_contacts):
        raise ValueError("This assembly has no ground contact geometry")
    low, high = all_contacts.min(0), all_contacts.max(0)
    nearby = ((local[:, :2] >= low[:2] - .45) & (local[:, :2] <= high[:2] + .45)).all(1)
    points = points[nearby]
    road = local[nearby & (abs(local[:, 2]) < .24)]
    jump = coherent_step(road, low[:2], high[:2])
    if jump > .040:
        raise ValueError(f"A coherent road step of {jump:.3f} m crosses the assembly")
    tree = cKDTree(road[:, :2])
    checked = []
    for row in members:
        vertices = np.concatenate(contacts[row["id"]])
        xy = vertices[:, :2]
        if len(xy) < 3:
            raise ValueError("Missing support vertices for " + row["id"])
        # Test the lowest load-bearing vertices themselves, including narrow feet.
        feet = vertices[vertices[:, 2] <= .00081, :2]
        _, unique = np.unique(np.rint(feet / .035), axis=0, return_index=True)
        returns = tree.query_ball_point(feet[unique], .105)
        if not len(returns) or any(len(group) < 3 for group in returns):
            raise ValueError(f'{row["id"]}: missing measured road beside a load-bearing contact')
        measured_gap = max(abs(float(np.median(road[group, 2]))) for group in returns)
        if measured_gap > .020:
            raise ValueError(f'{row["id"]}: measured contact height differs by {measured_gap:.3f} m')
        lo, hi = xy.min(0), xy.max(0)
        center = origin + basis[:, :2] @ ((lo + hi) / 2)
        seed = dict(anchor, center_world=center.tolist())
        patch = fit_footprint(seed, points, np.maximum(hi - lo, [.45, .45]))
        normal = np.asarray(patch["basis_columns"])[:, 2]
        actual = origin + xy @ basis[:, :2].T
        error = np.abs((actual - patch["center_world"]) @ normal)
        support = patch["footprint_support"]
        if error.max() > .020 or support["cell_residual_p90_m"] > .020:
            raise ValueError(f'{row["id"]}: shared support exceeds 20 mm at the contact or in the road survey')
        checked.append(dict(id=row["id"], asset=row["asset"], contact_vertices=len(xy),
                            contact_sites=len(unique), max_measured_gap_m=measured_gap,
                            max_plane_gap_m=float(error.max()), center_world=patch["center_world"],
                            normal_world=normal.tolist(), **support))
    return dict(instances=checked, geometry_instances=geometry_instances, max_adjacent_step_m=jump, tolerance_m=.020)
