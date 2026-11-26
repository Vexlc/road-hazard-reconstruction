"""Compare first returns on the uplifted sheets and their host road plane along real LiDAR rays."""

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
    path = models / "model/series/assets/pothole/P581/asset.glb"
    full = Surface(path)
    emerging = Surface(path, exclude={1, 175})
    shots = [read(drive / side / "P581/placement.json") for side in ("front", "back")]
    anchor = shots[0]["anchor"]
    if any(
        r["glb_sha256"] != digest(path) or r["anchor"] != anchor or r["scene"] != raw.name
        for r in shots
    ):
        raise ValueError("Different mesh or installation")
    bounds = driver.world_bounds(anchor, [full.low, full.high])
    survey = driver.pandaset(raw)[0]
    R = np.array(anchor["basis_columns"])
    centre = np.array(anchor["center_world"])
    roads = [
        int(i) for i, n in read(raw / "annotations/semseg/classes.json").items() if n in ROAD_NAMES
    ]
    counts = Counter()
    frames = []
    ranges = []
    for i in sorted({s["scan"] for shot in shots for s in shot["exposures"]}):
        origin, v, r, t, sensor, identity = driver.scan(raw / "lidar" / f"{i:02}.pkl.gz")
        row = dict(frame=i, source_sha256=identity)
        frames.append(row)
        try:
            row["traffic"] = survey.require_window(
                {"P581": bounds}, float(t.min()), float(t.max()), 0.25
            )
        except ValueError as e:
            row["excluded"] = str(e)
            continue
        o = (origin - centre) @ R
        v = v @ R
        ids = np.flatnonzero(
            (r > 0.05) & possible(o, v, [emerging.low - 0.002, emerging.high + 0.002])
        )
        o, v, r = o[ids], v[ids], r[ids]
        scan = raw / "annotations/semseg" / f"{i:02}.pkl.gz"
        labels = pd.read_pickle(scan)
        if not np.array_equal(labels.index, np.arange(len(t))):
            raise ValueError("Source row mismatch")
        label = labels["class"].to_numpy()[ids]
        ground = np.divide(-o[:, 2], v[:, 2], out=np.full(len(r), np.nan), where=v[:, 2] < -1e-5)
        hit, part, _ = full.cast(o, v)
        # Compare only identified ground returns. A closer recorded obstruction has precedence.
        use = (
            np.isin(part, [13, 14])
            & np.isin(label, roads)
            & (ground > 0)
            & (abs(o[:, 2] + r * v[:, 2]) < 0.06)
            & (hit <= r + 0.003)
        )
        separation = ground[use] - hit[use]
        if (separation < -0.003).any():
            raise ValueError("Positive-height surface behind road")
        ranges.extend(separation.tolist())
        packet = out / f"{i:02}.npz"
        np.savez_compressed(
            packet,
            source_row=ids.astype("i4"),
            time_s=t[ids],
            sensor=sensor[ids],
            semantic=label,
            origin_local_m=o.astype("f4"),
            unit_ray_local=v.astype("f4"),
            host_range_m=r.astype("f4"),
            grade_plane_range_m=ground.astype("f4"),
            replay_range_m=np.where(use, hit, r).astype("f4"),
            inserted_part=np.where(use, part, 0).astype("u1"),
        )
        numbers = dict(
            candidates=len(ids),
            paired=int(use.sum()),
            top=int((use & (part == 13)).sum()),
            face=int((use & (part == 14)).sum()),
            kept_host=int((~use).sum()),
        )
        row.update(**numbers, labels_sha256=digest(scan), packet_sha256=digest(packet))
        counts.update(numbers)
    if not counts["face"] or not counts["top"]:
        raise ValueError(counts)
    write(
        out / "report.json",
        dict(
            scene=raw.name,
            asset_sha256=digest(path),
            rig_sha256=digest(RIG),
            accepted_scans=sum("excluded" not in r for r in frames),
            totals=dict(counts),
            frames=frames,
            plane_minus_insert_range_m=dict(
                zip(["p50", "p90", "p99"], np.percentile(ranges, [50, 90, 99]).tolist())
            ),
        ),
    )
    print(dict(counts))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("models", "drive", "raw", "out"):
        p.add_argument(k, type=Path)
    replay(**vars(p.parse_args()))
