"""Register an inset paving stone to labelled driveway returns from measured scans."""

import numpy as np
import pandas as pd
from mesh_transfer import fit_road
from traffic_safe_drive import digest, read


def anchor_from_driveway(raw, drive, record):
    if raw.name != record["scene"]:
        raise ValueError("Ground survey and source sequence differ")
    sem = raw / "annotations/semseg"
    support_class = record.get("support_surface", "Driveway")
    label = next(
        int(code)
        for code, name in read(sem / "classes.json").items()
        if name == support_class
    )
    frame, row = record["seed_scan_row"]
    cloud_path = raw / "lidar" / f"{frame:02}.pkl.gz"
    if digest(cloud_path) != record["seed_scan_sha256"]:
        raise ValueError("Changed source of surveyed support point")
    points = drive.cloud(raw, frame)
    if pd.read_pickle(sem / cloud_path.name)["class"].iloc[row] != label:
        raise ValueError("Installation seed is outside the specified support class")
    seed = points[row]
    support = []
    for f in range(*record["support_scan_range"]):
        path = f"{f:02}.pkl.gz"
        pts = drive.cloud(raw, f)
        labels = pd.read_pickle(sem / path)["class"].to_numpy()
        good = (labels == label) & (np.linalg.norm(pts - seed, axis=1) < 1.4)
        support.append(pts[good])
    pts = np.concatenate(support)
    pose = drive.pose_matrix(read(raw / "camera/front_camera/poses.json")[frame])
    anchor = fit_road(
        pts, seed, pose[:3, 2], up_world=[0, 0, 1], radius_m=record["radius_m"]
    )
    prefix = "driveway" if support_class == "Driveway" else "ground"
    return {**anchor, prefix+"_source":record, prefix+"_support_returns":len(pts)}
