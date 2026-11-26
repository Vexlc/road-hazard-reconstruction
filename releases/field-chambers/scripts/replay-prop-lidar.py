"""Intersect measured dual-LiDAR beams with a traffic-screened, stationary PandaSet prop."""

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from traffic_safe_drive import ROOT, digest, pandaset, read, write
from traffic_clearance import world_bounds

sys.path.insert(0, str(ROOT / "gen2hole/downstream/sensors"))
from prepare_scans import RIG, calibrate, interpolate, possible
from road_geometry import Surface


@lru_cache(maxsize=4)
def scan(path):
    raw = path.parents[1]
    times, centers, _ = calibration(raw)
    cloud = pd.read_pickle(path)
    xyz = cloud[["x", "y", "z"]].to_numpy(dtype=float)
    t = cloud["t"].to_numpy(dtype=float)
    sensor = cloud["d"].to_numpy()
    if not (np.array_equal(np.unique(sensor), [0, 1]) and np.isfinite(xyz).all() and np.isfinite(t).all()):
        raise ValueError(f"Invalid calibrated dual-LiDAR source: {path}")
    origins = np.empty_like(xyz)
    for emitter in (0, 1):
        mask = sensor == emitter
        origins[mask] = interpolate(times, centers[emitter], t[mask])
    delta = xyz - origins
    ranges = np.linalg.norm(delta, axis=1)
    return origins, delta / np.maximum(ranges[:, None], 1e-12), ranges, t, sensor, digest(path)


@lru_cache(maxsize=2)
def calibration(raw):
    return calibrate(raw)


def replay(args, dest):
    survey = pandaset(args.raw)[0]
    rig_hash = digest(RIG)
    rows = {}
    for place in sorted(args.views.glob("*/placement.json")):
        record = read(place)
        key = record["asset"]
        glb = args.models / key / "asset.glb"
        if (
            record["scene"] != args.raw.name
            or key != place.parent.name
            or digest(glb) != record["glb_sha256"]
        ):
            raise ValueError(f"{key}: host or geometry does not match the camera exposure")
        mesh = Surface(glb)
        anchor = record["anchor"]
        center = np.array(anchor["center_world"])
        basis = np.array(anchor["basis_columns"])
        bounds = world_bounds(anchor, [mesh.low, mesh.high])
        if not np.allclose(bounds, record["world_xy_bounds_m"], atol=1e-6, rtol=0):
            raise ValueError(f"{key}: placement geometry disagrees with reported footprint")
        exposures = {row["lidar_frame"] for row in record["frames"] if "visible_px" in row}
        folder = dest / key
        folder.mkdir()
        total = Counter()
        shots = []
        for frame in sorted(exposures):
            path = args.raw / "lidar" / f"{frame:02}.pkl.gz"
            origin, direction, before, t, sensor, identity = scan(path)
            frame_row = dict(frame=frame, source_sha256=identity)
            shots.append(frame_row)
            try:
                clearance = survey.require_window({key: bounds}, float(t.min()), float(t.max()), 0.25)
            except ValueError as error:
                frame_row["rejected"] = str(error)
                continue
            local_o = (origin - center) @ basis
            local_d = direction @ basis
            cand = np.flatnonzero(
                (before > 0.05) & possible(local_o, local_d, (mesh.low - 0.001, mesh.high + 0.001))
            )
            if len(cand):
                hit, part, _ = mesh.cast(local_o[cand], local_d[cand])
            else:
                hit, part = np.empty(0, "f4"), np.empty(0, "u1")
            # A nearer measured return occludes the inserted object. Preserve millimetre ties.
            nearer = (hit > 0) & (hit < before[cand] - 0.003)
            after = np.where(nearer, hit, before[cand])
            points = origin[cand] + after[:, None] * direction[cand]
            counts = dict(
                candidate_beams=len(cand),
                changed=int(nearer.sum()),
                main_hits=int(np.sum(nearer & (sensor[cand] == 0))),
                forward_hits=int(np.sum(nearer & (sensor[cand] == 1))),
                through_box_without_mesh=int(np.isinf(hit).sum()),
                measured_return_in_front=int((np.isfinite(hit) & ~nearer).sum()),
            )
            total.update(counts)
            out = folder / f"{frame:02}.npz"
            np.savez_compressed(
                out,
                row=cand.astype("i4"),
                sensor=sensor[cand].astype("u1"),
                time_s=t[cand],
                origin_world_m=origin[cand],
                direction_world=direction[cand],
                range_before_m=before[cand],
                mesh_first_m=hit,
                range_after_m=after,
                xyz_after_world_m=points,
                hit_part=np.where(nearer, part, 0).astype("u1"),
            )
            frame_row.update(counts, traffic=clearance, packet_sha256=digest(out))
        if not any("traffic" in shot for shot in shots):
            raise ValueError(f"{key}: no complete traffic-clear scan")
        rows[key] = dict(
            totals=dict(total),
            frames=shots,
            anchor=center.tolist(),
            basis=basis.tolist(),
            glb_sha256=record["glb_sha256"],
            placement_sha256=digest(place),
        )
        print(key, dict(total), flush=True)
    if not rows:
        raise ValueError("No placements found")
    write(
        dest / "report.json",
        dict(
            scene=args.raw.name,
            rig_sha256=rig_hash,
            cross_camera_origin_residual=calibration(args.raw)[2],
            assets=rows,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "views", "raw", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("Choose a fresh output directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "scans"
        stage.mkdir()
        replay(args, stage)
        stage.rename(args.out)
