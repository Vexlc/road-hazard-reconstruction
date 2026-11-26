"""Replay vehicle beams with an explicit unresolved state for singly observed debris depths."""

import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from camera import HERE

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import LIB, Surface

sys.path.insert(0, str(LIB.parent / "scripts"))
driver = import_module("replay-prop-lidar")
from prepare_scans import ROAD_NAMES, possible, RIG
from traffic_safe_drive import read, write, digest


def replay(models, drive, raw, out):
    out.mkdir(parents=True, exist_ok=True)
    path = models / "model/series/assets/pothole/P584/asset.glb"
    mesh = Surface(path)
    poses = [read(drive / s / "P584/placement.json") for s in ("front", "back")]
    anchor = poses[0]["anchor"]
    if not all(
        p["anchor"] == anchor and p["glb_sha256"] == digest(path) and p["scene"] == raw.name
        for p in poses
    ):
        raise ValueError("Different source mesh, host scene or installation")
    bounds = driver.world_bounds(anchor, [mesh.low, mesh.high])
    survey = driver.pandaset(raw)[0]
    R = np.asarray(anchor["basis_columns"])
    center = np.asarray(anchor["center_world"])
    roads = [
        int(i) for i, n in read(raw / "annotations/semseg/classes.json").items() if n in ROAD_NAMES
    ]
    frames = []
    total = Counter()
    diff = []
    for i in sorted({row["scan"] for p in poses for row in p["exposures"]}):
        origin, v, r, t, sensor, source = driver.scan(raw / "lidar" / f"{i:02}.pkl.gz")
        row = dict(frame=i, source_sha256=source)
        frames.append(row)
        try:
            row["traffic"] = survey.require_window(
                {"P584": bounds}, float(t.min()), float(t.max()), 0.25
            )
        except ValueError as error:
            row["excluded"] = str(error)
            continue
        o = (origin - center) @ R
        v = v @ R
        ids = np.flatnonzero((r > 0.05) & possible(o, v, [mesh.low - 0.006, mesh.high + 0.006]))
        o, v, r, t, sensor = o[ids], v[ids], r[ids], t[ids], sensor[ids]
        scan = raw / "annotations/semseg" / f"{i:02}.pkl.gz"
        data = pd.read_pickle(scan)
        if not np.array_equal(data.index, np.arange(len(data))):
            raise ValueError("Semantic row order")
        label = data["class"].to_numpy()[ids]
        plane = -o[:, 2] / np.where(v[:, 2] < -1e-5, v[:, 2], np.nan)
        hit, part, _ = mesh.cast(o, v)
        # Replace only a measured near-grade road return. Scanned objects and
        # significant foreground range occluders retain their original distances.
        road = np.isin(label, roads) & (abs(o[:, 2] + r * v[:, 2]) < 0.04) & (plane > 0)
        visible = r >= np.minimum(plane, hit) - 0.10
        use = road & visible & np.isin(part, [25, 180, 181, 182, 183, 184])
        constrained = use & np.isin(part, [25, 180])
        unknown = use & ~constrained
        # Never see through a hypothesised occluder to a farther constrained wall.
        bounded_range = np.where(unknown, np.nan, np.where(constrained, hit, r))
        diff.extend((hit[constrained] - plane[constrained]).tolist())
        packet = out / f"{i:02}.npz"
        np.savez_compressed(
            packet,
            source_row=ids.astype("i4"),
            sensor=sensor,
            time_s=t,
            semantic=label,
            origin_local_m=o.astype("f4"),
            unit_ray_local=v.astype("f4"),
            host_range_m=r.astype("f4"),
            grade_plane_range_m=plane.astype("f4"),
            visual_range_m=np.where(use, hit, r).astype("f4"),
            bounded_range_m=bounded_range.astype("f4"),
            unresolved=unknown,
            inserted_part=np.where(use, part, 0).astype("u1"),
        )
        stats = dict(
            candidate_beams=len(ids),
            paired=int(use.sum()),
            kept_host=int((~use).sum()),
            upright_cones=int((constrained & (part == 25)).sum()),
            exposed_wall=int((constrained & (part == 180)).sum()),
            unknown=int(unknown.sum()),
            continuation=int((use & (part == 181)).sum()),
            bag=int((use & (part == 182)).sum()),
            cardboard=int((use & (part == 183)).sum()),
            trapped_cone=int((use & (part == 184)).sum()),
        )
        row.update(stats, packet_sha256=digest(packet), semantics_sha256=digest(scan))
        total.update(stats)
    if min(total["upright_cones"], total["unknown"]) <= 0:
        raise ValueError(total)
    result = dict(
        scene=raw.name,
        glb_sha256=digest(path),
        rig_sha256=digest(RIG),
        constrained_range_minus_plane_m=dict(
            zip(("p10", "median", "p90"), np.percentile(diff, [10, 50, 90]).tolist())
        ),
        accepted_scans=sum("excluded" not in f for f in frames),
        counts=dict(total),
        frames=frames,
    )
    write(out / "report.json", result)
    print(total, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "drive", "raw", "out"):
        p.add_argument(name, type=Path)
    replay(**vars(p.parse_args()))
