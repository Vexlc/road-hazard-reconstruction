"""Replay recorded PandaSet beams against the calibrated cones and an open-bottom chamber."""

import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import json

HERE = Path(__file__).resolve().parent
RADIUS = json.loads((HERE / "survey.json").read_text())["working_diameter_nominal_m"] / 2

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import LIB, Surface

sys.path.insert(0, str(LIB.parent / "scripts"))
driver = import_module("replay-prop-lidar")
from prepare_scans import ROAD_NAMES, possible, RIG
from traffic_safe_drive import read, write, digest


def check(
    models,
    drive,
    raw,
    out,
    *,
    asset_id="M571",
    radius=RADIUS,
    extra_parts=None,
    replacement_parts=(5, 6),
    replacement_state="replaced_road_with_sleeve_or_seat",
):
    folder = models / "model/series/assets/missing" / asset_id
    mesh = Surface(folder / "observed.glb", exclude={1})
    if extra_parts is None:
        extra_parts = {"cone_hits": 25, "arc_hits": 18}
    placements = [read(drive / side / asset_id / "placement.json") for side in ("front", "back")]
    assert placements[0]["anchor"] == placements[1]["anchor"]
    assert all(
        p["glb_sha256"] == digest(folder / "asset.glb") and p["scene"] == raw.name
        for p in placements
    )
    anchor = placements[0]["anchor"]
    center = np.asarray(anchor["center_world"])
    basis = np.asarray(anchor["basis_columns"])
    full = Surface(folder / "asset.glb")
    bounds = driver.world_bounds(anchor, [full.low, full.high])
    survey = driver.pandaset(raw)[0]
    road_ids = [
        int(c)
        for c, label in read(raw / "annotations/semseg/classes.json").items()
        if label in ROAD_NAMES
    ]
    assert len(road_ids) == 4
    frames = sorted({e["scan"] for p in placements for e in p["exposures"]})
    outcomes = [
        "unchanged",
        "nearer_native_surface",
        replacement_state,
        "open_depth_unknown",
    ]
    out.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    by_sensor = {0: Counter(), 1: Counter()}
    records = []
    for frame in frames:
        f = raw / "lidar" / f"{frame:02}.pkl.gz"
        origin, direction, measured, time, sensor, identity = driver.scan(f)
        row = dict(frame=frame, cloud_sha256=identity)
        records.append(row)
        try:
            row["traffic"] = survey.require_window(
                {asset_id: bounds}, float(time.min()), float(time.max()), 0.25
            )
        except ValueError as error:
            row["excluded"] = str(error)
            continue
        o = (origin - center) @ basis
        v = direction @ basis
        ids = np.flatnonzero(
            (measured > 0.05) & possible(o, v, [mesh.low - 0.001, mesh.high + 0.001])
        )
        o, v, before = o[ids], v[ids], measured[ids]
        labels_path = raw / "annotations/semseg" / f"{frame:02}.pkl.gz"
        labels = pd.read_pickle(labels_path)
        assert len(labels) == len(measured) and np.array_equal(
            labels.index, np.arange(len(measured))
        )
        semantic = labels["class"].to_numpy()[ids]
        row["labels_sha256"] = digest(labels_path)
        hit, part, _ = mesh.cast(o, v)
        t = -o[:, 2] / np.where(abs(v[:, 2]) > 1e-8, v[:, 2], np.nan)
        crosses = (t > 0) & (np.linalg.norm(o[:, :2] + t[:, None] * v[:, :2], axis=1) < radius)
        # Ignore the outside of the sleeve on rays whose entry at grade is on intact road.
        below = o[:, 2] + np.where(np.isfinite(hit), hit, 0) * v[:, 2] < -0.002
        valid = np.isfinite(hit) & (hit > 0) & (~below | crosses)
        orig_position = o + v * before[:, None]
        local_road = np.isin(semantic, road_ids) & (abs(orig_position[:, 2]) < 0.04)
        replaces = crosses & local_road & (np.linalg.norm(orig_position[:, :2], axis=1) < radius)
        kind = np.zeros(len(ids), "u1")
        nearer = valid & (hit < before - 0.003)
        kind[nearer] = 1
        kind[replaces & ~nearer & valid] = 2
        kind[replaces & ~valid] = 3
        after = before.copy()
        after[(kind == 1) | (kind == 2)] = hit[(kind == 1) | (kind == 2)]
        after[kind == 3] = np.nan
        assert np.array_equal(after[kind == 0], before[kind == 0])
        assert np.all(after[kind == 1] < before[kind == 1] - 0.003) and np.all(
            after[kind == 2] >= before[kind == 2] - 0.003
        )
        if not np.all(np.isin(part[kind == 2], replacement_parts)):
            bad = (kind == 2) & ~np.isin(part, replacement_parts)
            raise ValueError(
                (frame, part[bad].tolist(), (o[bad] + v[bad] * hit[bad, None]).tolist())
            )
        assert np.all(np.isnan(after[kind == 3]))
        assert not np.any(np.isin(part, [3, 7]))
        packet = out / f"{frame:02}.npz"
        np.savez_compressed(
            packet,
            source_row=ids.astype("i4"),
            time_s=time[ids],
            sensor=sensor[ids].astype("u1"),
            origin_local_m=o.astype("f4"),
            direction_local=v.astype("f4"),
            host_semantic=semantic.astype("u1"),
            range_host_m=before.astype("f4"),
            range_model_m=after.astype("f4"),
            state=kind,
            part=np.where((kind == 1) | (kind == 2), part, 0).astype("u1"),
        )
        counts = {name: int((kind == j).sum()) for j, name in enumerate(outcomes)}
        counts.update(
            candidate_beams=len(kind),
            **{
                name: int(((kind == 1) & (part == code)).sum())
                for name, code in extra_parts.items()
            },
        )
        row.update(counts, packet_sha256=digest(packet))
        totals.update(counts)
        for i, sensor_counts in by_sensor.items():
            sensor_counts.update(
                {
                    name: int(((kind == j) & (sensor[ids] == i)).sum())
                    for j, name in enumerate(outcomes)
                }
            )
    # A shallow vehicle beam can hit the sleeve even when a vertical ray is open.
    # Validate the vertical control separately from the measured directions.
    down = mesh.cast(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]]))[0]
    assert (
        np.isinf(down[0]) and totals[next(iter(extra_parts))] > 0 and totals[replacement_state] > 0
    )
    report = dict(
        scene=raw.name,
        visual_model_sha256=placements[0]["glb_sha256"],
        open_model_sha256=digest(folder / "observed.glb"),
        rig_sha256=digest(RIG),
        road_semantic_ids=road_ids,
        host_grade_tolerance_m=0.04,
        states=outcomes,
        anchor=anchor,
        vertical_center_control="no observed return",
        totals=dict(totals),
        per_sensor={str(i): dict(v) for i, v in by_sensor.items()},
        frames=records,
    )
    write(out / "report.json", report)
    print(totals, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("models", "drive", "raw", "out"):
        p.add_argument(key, type=Path)
    check(**vars(p.parse_args()))
