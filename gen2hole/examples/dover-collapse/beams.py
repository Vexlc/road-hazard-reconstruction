"""Replay real vehicle ray directions; keep a separate range stream for stereo-supported pavement."""

import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from cameras import HERE

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import LIB, Surface

sys.path.insert(0, str(LIB.parent / "scripts"))
driver = import_module("replay-prop-lidar")
from prepare_scans import ROAD_NAMES, possible, RIG
from traffic_safe_drive import read, write, digest


def replay(models, drive, raw, out):
    out.mkdir(parents=True, exist_ok=True)
    path = models / "model/series/assets/pothole/P590/asset.glb"
    mesh = Surface(path)
    poses = [read(drive / s / "P590/placement.json") for s in ("front", "back")]
    anchor = poses[0]["anchor"]
    if not all(
        p["anchor"] == anchor and p["glb_sha256"] == digest(path) and p["scene"] == raw.name
        for p in poses
    ):
        raise ValueError("Different source mesh, host scene or installation")
    with np.load(path.parent / "surface.npz") as field:
        extent = np.array([field["width"], field["length"]])
        supports = field["stereo_valid"].copy()
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
                {"P590": bounds}, float(t.min()), float(t.max()), 0.25
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
        use = road & visible & (part == 2)
        xy = o[:, :2] + np.where(use, hit, 0)[:, None] * v[:, :2]
        grid = (0.5 + xy / extent * [1, -1]) * (np.array(supports.shape[::-1]) - 1)
        ij = np.rint(grid).astype(int).clip([0, 0], np.array(supports.shape[::-1]) - 1)
        reliable = use & supports[ij[:, 1], ij[:, 0]]
        if np.any(~np.isfinite(hit[use])):
            raise ValueError("Invalid deck intersection")
        delta = hit[reliable] - plane[reliable]
        diff.extend(delta.tolist())
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
            supported_range_m=np.where(reliable, hit, r).astype("f4"),
            stereo_supported=reliable,
            inserted_part=np.where(use, part, 0).astype("u1"),
        )
        stats = dict(
            candidate_beams=len(ids),
            paired=int(use.sum()),
            kept_host=int((~use).sum()),
            optical_face_kept=int((road & visible & (part == 3)).sum()),
            stereo_supported=int(reliable.sum()),
            interpolated=int((use & ~reliable).sum()),
        )
        row.update(stats, packet_sha256=digest(packet), semantics_sha256=digest(scan))
        total.update(stats)
    if total["stereo_supported"] <= 0:
        raise ValueError(total)
    result = dict(
        scene=raw.name,
        glb_sha256=digest(path),
        rig_sha256=digest(RIG),
        supported_range_minus_plane_m=dict(
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
