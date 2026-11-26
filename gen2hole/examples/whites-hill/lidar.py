"""Replay two explicit water-return hypotheses on the original PandaSet beam directions."""

import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lane-damage"))
from photo_mesh import LIB, Surface

sys.path.insert(0, str(LIB.parent / "scripts"))
driver = import_module("replay-prop-lidar")
from prepare_scans import ROAD_NAMES, possible, RIG
from traffic_safe_drive import read, write, digest


def update(scene, o, v, original, entry, replaces):
    hit, part, _ = scene.cast(o, v)
    valid = np.isfinite(hit) & (hit > 0.01)
    above = o[:, 2] + np.where(valid, hit, 0) * v[:, 2] >= -0.002
    valid &= above | entry
    closer = valid & (hit < original - 0.003)
    state = np.zeros(len(o), "u1")
    state[closer] = 1
    state[replaces & ~closer & valid] = 2
    state[replaces & ~valid] = 3
    updated = original.copy()
    used = (state == 1) | (state == 2)
    updated[used] = hit[used]
    updated[state == 3] = np.nan
    return updated, np.where(used, part, 0).astype("u1"), state


def check(models, drive, raw, out):
    out.mkdir(exist_ok=True, parents=True)
    f = models / "model/series/assets/missing/M576/asset.glb"
    placements = [read(drive / side / "M576/placement.json") for side in ["front", "back"]]
    assert all(p["glb_sha256"] == digest(f) and p["scene"] == raw.name for p in placements)
    anchor = placements[0]["anchor"]
    assert anchor == placements[1]["anchor"]
    R = np.array(anchor["basis_columns"])
    centre = np.array(anchor["center_world"])
    meta = read(f.parent / "metadata.json")
    half = np.array(meta["camera_fit"]["opening_xy_m"]) / 2
    optical = Surface(f, exclude={1})
    solid = Surface(f, exclude={1, 4})
    bounds = driver.world_bounds(anchor, [optical.low, optical.high])
    survey = driver.pandaset(raw)[0]
    roads = [
        int(i) for i, k in read(raw / "annotations/semseg/classes.json").items() if k in ROAD_NAMES
    ]
    totals = Counter()
    frames = []
    for frame in sorted({x["scan"] for p in placements for x in p["exposures"]}):
        path = raw / "lidar" / f"{frame:02}.pkl.gz"
        origin, direction, measured, t, sensor, identity = driver.scan(path)
        row = dict(frame=frame, source_sha256=identity)
        frames.append(row)
        try:
            row["traffic"] = survey.require_window(
                {"M576": bounds}, float(t.min()), float(t.max()), 0.25
            )
        except ValueError as error:
            row["excluded"] = str(error)
            continue
        o = (origin - centre) @ R
        v = direction @ R
        ids = np.flatnonzero(
            (measured > 0.05) & possible(o, v, [optical.low - 0.002, optical.high + 0.002])
        )
        o, v, before = o[ids], v[ids], measured[ids]
        labels_path = raw / "annotations/semseg" / f"{frame:02}.pkl.gz"
        sem = pd.read_pickle(labels_path)
        assert np.array_equal(sem.index, np.arange(len(measured)))
        labels = sem["class"].to_numpy()[ids]
        plane = -o[:, 2] / np.where(abs(v[:, 2]) > 1e-8, v[:, 2], np.nan)
        entry = (plane > 0) & np.all(abs(o[:, :2] + plane[:, None] * v[:, :2]) < half, axis=1)
        old = o + before[:, None] * v
        replaces = (
            entry
            & np.isin(labels, roads)
            & (abs(old[:, 2]) < 0.04)
            & np.all(abs(old[:, :2]) < half, axis=1)
        )
        water, wp, ws = update(optical, o, v, before, entry, replaces)
        dry, sp, ss = update(solid, o, v, before, entry, replaces)
        common = (wp != 4) & (wp == sp)
        assert np.allclose(water[common], dry[common], equal_nan=True) and np.all(sp != 4)
        different = ~np.isclose(water, dry, equal_nan=True, atol=0.001, rtol=0)
        counts = dict(
            candidates=len(ids),
            unchanged=int((ws == 0).sum()),
            closer=int((ws == 1).sum()),
            replaced_road=int((ws == 2).sum()),
            unknown=int((ws == 3).sum()),
            guard_hits=int((wp == 174).sum()),
            slab_hits=int(np.isin(wp, [9, 11]).sum()),
            water_hits=int((wp == 4).sum()),
            different_by_water_response=int(different.sum()),
        )
        packet = out / f"{frame:02}.npz"
        np.savez_compressed(
            packet,
            source_row=ids.astype("i4"),
            sensor=sensor[ids],
            time_s=t[ids],
            origin_local_m=o.astype("f4"),
            direction_local=v.astype("f4"),
            semantic=labels.astype("u1"),
            host_range_m=before.astype("f4"),
            water_return_range_m=water.astype("f4"),
            no_water_return_range_m=dry.astype("f4"),
            water_return_part=wp,
            no_water_return_part=sp,
            water_return_state=ws,
            no_water_return_state=ss,
        )
        row.update(**counts, labels_sha256=digest(labels_path), packet_sha256=digest(packet))
        totals.update(counts)
    probes = np.array([[-0.23, -0.11, 1.0], [-0.11, 0.05, 1.0], [-0.16, 0.18, 1.0]])
    down = np.tile([0.0, 0.0, -1.0], (len(probes), 1))
    water, part, _ = optical.cast(probes, down)
    bed, _, _ = solid.cast(probes, down)
    assert (
        np.all(part == 4) and np.all(np.isinf(bed)) and totals["guard_hits"] and totals["slab_hits"]
    )
    write(
        out / "report.json",
        dict(
            scene=raw.name,
            model_sha256=digest(f),
            rig_sha256=digest(RIG),
            states=["unchanged", "closer_inserted_surface", "replaced_host_road", "unknown"],
            vertical_water_return_m=water.tolist(),
            vertical_solid_only="no return",
            totals=dict(totals),
            accepted_scans=sum("excluded" not in row for row in frames),
            frames=frames,
        ),
    )
    print(totals)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("models", "drive", "raw", "out"):
        p.add_argument(key, type=Path)
    check(**vars(p.parse_args()))
