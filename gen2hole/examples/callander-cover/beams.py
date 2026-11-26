"""Compare flat and relief geometry on identical, road-labelled PandaSet emission directions."""

import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import LIB, Surface

sys.path.insert(0, str(LIB.parent / "scripts"))
driver = import_module("replay-prop-lidar")
from prepare_scans import ROAD_NAMES, possible, RIG
from traffic_safe_drive import read, write, digest


def replay(models, drive, raw, out):
    out.mkdir(parents=True, exist_ok=True)
    folder = models / "model/series/assets/raised/A580"
    model = folder / "asset.glb"
    flat = folder / "flush.glb"
    records = [read(drive / side / "A580/placement.json") for side in ("front", "back")]
    anchor = records[0]["anchor"]
    if any(
        r["anchor"] != anchor or r["scene"] != raw.name or r["glb_sha256"] != digest(model)
        for r in records
    ):
        raise ValueError("Inconsistent camera placement")
    pose = np.array(anchor["basis_columns"])
    centre = np.array(anchor["center_world"])
    surface = Surface(model)
    control = Surface(flat)
    installed = Surface(model, exclude={1})
    bounds = driver.world_bounds(anchor, [installed.low, installed.high])
    survey = driver.pandaset(raw)[0]
    road = [
        int(k) for k, v in read(raw / "annotations/semseg/classes.json").items() if v in ROAD_NAMES
    ]
    frames = []
    totals = Counter()
    deltas = []
    for i in sorted({f["scan"] for r in records for f in r["exposures"]}):
        origin, direction, original, t, sensor, source = driver.scan(
            raw / "lidar" / f"{i:02}.pkl.gz"
        )
        row = dict(frame=i, source_sha256=source)
        frames.append(row)
        try:
            row["traffic"] = survey.require_window(
                {"A580": bounds}, float(t.min()), float(t.max()), 0.25
            )
        except ValueError as e:
            row["excluded"] = str(e)
            continue
        o = (origin - centre) @ pose
        v = direction @ pose
        ids = np.flatnonzero(
            (original > 0.05) & possible(o, v, [installed.low - 0.004, installed.high + 0.004])
        )
        o, v, host = o[ids], v[ids], original[ids]
        labels_path = raw / "annotations/semseg" / f"{i:02}.pkl.gz"
        sem = pd.read_pickle(labels_path)
        if not np.array_equal(sem.index, np.arange(len(original))):
            raise ValueError("Point row mismatch")
        labels = sem["class"].to_numpy()[ids]
        ground, fp, _ = control.cast(o, v)
        fitted, rp, _ = surface.cast(o, v)
        old = o + host[:, None] * v
        _, old_part, _ = control.cast(
            np.column_stack([old[:, :2], np.ones(len(old))]),
            np.broadcast_to([0.0, 0.0, -1.0], old.shape),
        )
        # Compare geometry only when both the recorded road return and the grade-plane
        # entry lie on the installation. Other candidates remain original in both replays.
        used = (
            np.isin(labels, road) & (abs(old[:, 2]) < 0.04) & (old_part > 1) & (fp > 1) & (rp > 0)
        )
        fitted_out = np.where(used, fitted, host)
        flat_out = np.where(used, ground, host)
        diff = (fitted_out[used] - flat_out[used]) * 1000
        deltas.extend(diff.tolist())
        stationary = used & (rp == 9) & (fp == 9)
        assert np.allclose(fitted_out[stationary], flat_out[stationary], atol=2e-5, rtol=0)
        counts = dict(
            candidates=len(ids),
            paired=int(used.sum()),
            kept_host=int((~used).sum()),
            broken_asphalt_hits=int((used & (rp == 2)).sum()),
            iron_hits=int((used & (rp == 5)).sum()),
            face_hits=int((used & (rp == 9)).sum()),
            above_3mm=int((abs(diff) > 3).sum()),
        )
        packet = out / f"{i:02}.npz"
        np.savez_compressed(
            packet,
            source_row=ids.astype("i4"),
            sensor=sensor[ids],
            time_s=t[ids],
            semantic=labels.astype("u1"),
            origin_local_m=o.astype("f4"),
            unit_ray_local=v.astype("f4"),
            host_range_m=host.astype("f4"),
            relief_range_m=fitted_out.astype("f4"),
            flush_range_m=flat_out.astype("f4"),
            paired=used,
            relief_part=np.where(used, rp, 0),
            flush_part=np.where(used, fp, 0),
        )
        row.update(**counts, labels_sha256=digest(labels_path), packet_sha256=digest(packet))
        totals.update(counts)
    if not totals["face_hits"] or not totals["broken_asphalt_hits"]:
        raise ValueError(totals)
    write(
        out / "report.json",
        dict(
            scene=raw.name,
            model_sha256=digest(model),
            flush_sha256=digest(flat),
            rig_sha256=digest(RIG),
            accepted_scans=sum("excluded" not in r for r in frames),
            totals=dict(totals),
            absolute_paired_delta_mm_percentiles=dict(
                zip(
                    ["p50", "p90", "p99", "max"],
                    np.percentile(abs(np.asarray(deltas)), [50, 90, 99, 100]).tolist(),
                )
            ),
            frames=frames,
        ),
    )
    print(dict(totals), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("models", "drive", "raw", "out"):
        p.add_argument(key, type=Path)
    replay(**vars(p.parse_args()))
