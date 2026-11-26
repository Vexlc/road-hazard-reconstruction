"""Register an inlet to the observed curb toe after fitting grade on the lane side."""

import numpy as np
from mesh_transfer import fit_road, unit


def anchor_from_curb(raw, shot, model, poses, drive, record):
    if (
        raw.name != record["scene"]
        or shot != record["reference_frame"]
        or list(model[0]) != record["image_size"]
    ):
        raise ValueError("Curb marks must match their source exposure and image size")
    _, project, unproject = model
    seed, road, heading = drive.host_patch(
        raw, shot, project, poses, record["road_seed_px"]
    )
    fit = fit_road(road, seed, heading, up_world=[0, 0, 1], radius_m=1.05)
    origin = np.asarray(fit["center_world"])
    normal = np.asarray(fit["basis_columns"])[:, 2]
    camera = drive.pose_matrix(poses[shot])
    eye = camera[:3, 3]

    def at_grade(pixel):
        x, y = pixel
        rays, valid = unproject(x, y, 1, 1)
        if not valid[0, 0]:
            raise ValueError("Curb point outside source camera")
        d = camera[:3, :3] @ rays[0, 0]
        return eye + d * (normal @ (origin - eye)) / (normal @ d)

    a, b = map(at_grade, record["curb_toe_segment_px"])
    along = unit(b - a)
    side = unit(np.cross(along, normal))
    toe = a + along * ((at_grade(record["curb_toe_center_px"]) - a) @ along)
    if (at_grade(record["lane_side_witness_px"]) - toe) @ side < 0:
        side = -side
    basis = np.column_stack([side, np.cross(normal, side), normal])
    center = toe - record["curb_x_m"] * side
    # Laser profile on the two sides of the curb, measured in the local grade frame.
    points = np.concatenate(
        [
            drive.cloud(raw, i)[::2]
            for i in range(max(0, shot - 4), min(80, shot + 5), 2)
        ]
    )
    local = (points - center) @ basis
    ranges = {}
    strips = record.get(
        "profile_x_ranges_m", {"gutter": [-0.24, 0.2], "sidewalk": [-0.90, -0.47]}
    )
    for name, (x0, x1) in strips.items():
        region = (
            (local[:, 0] >= x0)
            & (local[:, 0] <= x1)
            & (abs(local[:, 1]) < 0.48)
            & (abs(local[:, 2]) < 0.35)
        )
        z = local[region, 2]
        if len(z) < 10:
            raise ValueError("Insufficient curb returns")
        ranges[name] = dict(
            returns=len(z),
            median_m=float(np.median(z)),
            p10_m=float(np.percentile(z, 10)),
            p90_m=float(np.percentile(z, 90)),
        )
    if not (
        0.05 < ranges["sidewalk"]["median_m"] - ranges["gutter"]["median_m"] < 0.26
    ):
        raise ValueError(
            f"Observed elevation profile does not support a curb: {ranges}"
        )
    return dict(
        fit,
        center_world=center.tolist(),
        basis_columns=basis.tolist(),
        curb_toe_world=toe.tolist(),
        curb_source=record,
        laser_profile=ranges,
    )


def replaced_support(points, labels, codes, anchor):
    """Identify the measured face and tread inside the replacement receiver."""
    local = (points - anchor["center_world"]) @ np.asarray(anchor["basis_columns"])
    record = anchor["curb_source"]
    half = np.asarray(record["receiver_extent_m"]) / 2
    foot = (abs(local[:, :2]) <= half).all(1)
    x, _, z = local.T
    toe = record["curb_x_m"]
    top = anchor["laser_profile"]["sidewalk"]["median_m"]
    # Only the original support facing the sensor is replaced. Cars, vegetation and
    # street furniture retain their measured positions even inside this rectangle.
    tread = (x <= toe + 0.035) & (abs(z - top) < 0.052)
    riser = (abs(x - toe) < 0.072) & (z >= -0.035) & (z <= top + 0.052)
    return (
        np.isin(labels, [codes[k] for k in ("Ground", "Road", "Sidewalk")])
        & foot
        & (tread | riser)
    )


def registered_curb_cloud(raw, scan, drive, anchor, reflections):
    """Replace measured curb/sidewalk support only inside the modeled receiver apron."""
    import pandas as pd

    points = drive.cloud(raw, scan)
    sem = raw / "annotations/semseg"
    codes = {name: int(code) for code, name in drive.read(sem / "classes.json").items()}
    labels = pd.read_pickle(sem / f"{scan:02}.pkl.gz")["class"].to_numpy()
    support = replaced_support(points, labels, codes, anchor)
    keep = ~support
    if reflections:
        keep &= labels != codes["Reflection"]
    return points[keep], int(support.sum())
