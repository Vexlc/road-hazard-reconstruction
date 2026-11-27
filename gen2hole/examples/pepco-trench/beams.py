"""Replay identical time-stamped vehicle beams against three placements of one casting."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from camera import HERE, S
from casting import visible_face

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import LIB, Surface

sys.path.insert(0, str(LIB.parent / "scripts"))
driver = import_module("replay-prop-lidar")
from prepare_scans import ROAD_NAMES, possible, RIG
from traffic_safe_drive import read, write, digest


def replay(models, drive, raw, out):
    out.mkdir(parents=True, exist_ok=True)
    states = S["states"]
    meshs = []
    hashes = {}
    scans = set()
    anchor = None
    for state in states:
        path = models / state / "model/series/assets/raised/A591/asset.glb"
        hashes[state] = digest(path)
        meshs.append(Surface(path))
        for side in ("front", "back"):
            rec = read(drive / state / side / "A591/placement.json")
            if anchor is None:
                anchor = rec["anchor"]
            if (
                rec["anchor"] != anchor
                or rec["glb_sha256"] != hashes[state]
                or rec["scene"] != raw.name
            ):
                raise ValueError("Conflicting installation or exposure")
            scans.update(e["scan"] for e in rec["exposures"])
    low = np.min([m.low for m in meshs], 0)
    high = np.max([m.high for m in meshs], 0)
    bounds = driver.world_bounds(anchor, [low, high])
    survey = driver.pandaset(raw)[0]
    R = np.array(anchor["basis_columns"])
    center = np.array(anchor["center_world"])
    road_classes = [
        int(i) for i, n in read(raw / "annotations/semseg/classes.json").items() if n in ROAD_NAMES
    ]
    registration = read(models / "registration.json")["calibration"]
    total_bracket = np.zeros(len(states), int)
    bracket_widths = [[] for state in states]
    counts = np.zeros((len(states), 3), int)
    changed = np.zeros((len(states), len(states)), int)
    frames = []
    for ix in sorted(scans):
        o, v, r, t, sensor, sha = driver.scan(raw / "lidar" / f"{ix:02}.pkl.gz")
        frame = dict(frame=ix, scan_sha256=sha)
        frames.append(frame)
        try:
            frame["traffic"] = survey.require_window(
                {"A591": bounds}, float(t.min()), float(t.max()), 0.25
            )
        except ValueError as error:
            frame["excluded"] = str(error)
            continue
        o = (o - center) @ R
        v = v @ R
        subset = np.flatnonzero((r > 0.05) & possible(o, v, [low - 0.006, high + 0.006]))
        o, v, r, t, sensor = o[subset], v[subset], r[subset], t[subset], sensor[subset]
        file = raw / "annotations/semseg" / f"{ix:02}.pkl.gz"
        data = pd.read_pickle(file)
        if not np.array_equal(data.index, np.arange(len(data))):
            raise ValueError("Semantic row order")
        label = data["class"].to_numpy()[subset]
        plane = -o[:, 2] / np.where(v[:, 2] < -1e-5, v[:, 2], np.nan)
        crossing = o[:, :2] + plane[:, None] * v[:, :2]
        enters = np.linalg.norm(crossing, axis=1) < S["nominal_near_diameter_m"] / 2 + 0.020
        envelopes = []
        road = np.isin(label, road_classes) & (abs(o[:, 2] + r * v[:, 2]) < 0.04) & (plane > 0)
        parts = []
        visual = []
        bounded = []
        unknown = []
        for k, m in enumerate(meshs):
            hit, part, _ = m.cast(o, v)
            z = o[:, 2] + v[:, 2] * np.where(np.isfinite(hit), hit, 0)
            use = (
                road & (r >= np.minimum(plane, hit) - 0.10) & (part > 1) & ((z >= -0.0015) | enters)
            )
            shift = np.array(S["offset_xy_m"]) if states[k] == "offset" else np.zeros(2)

            def source_part(orig, dist, labels):
                xy = orig[:, :2] + np.where(np.isfinite(dist), dist, 0)[:, None] * v[:, :2]
                return (labels == 9) & visible_face(registration, xy - shift)

            recorded = use & source_part(o, hit, part)
            unresolved = use & ~recorded
            # Independently trace at both author-estimated rise limits. Withhold a range
            # interval if the ray changes surfaces or exits the photographed footprint.
            bracket = []
            good = recorded.copy()
            for inches in S["reported_approximate_raise_inches"]:
                dz = 0.0254 * inches - S["model_raise_m"]
                eye = o - [0, 0, dz]
                alternative, labels, _ = m.cast(eye, v)
                good &= source_part(eye, alternative, labels)
                bracket.append(alternative)
            br = np.stack([np.minimum(*bracket), np.maximum(*bracket)], -1)
            br[~good] = np.nan
            envelopes.append(br)
            total_bracket[k] += good.sum()
            bracket_widths[k].extend(np.diff(br[good], axis=1).ravel().tolist())
            pred = np.where(use, hit, r)
            visual.append(pred)
            bounded.append(np.where(unresolved, np.nan, pred))
            parts.append(np.where(use, part, 0))
            unknown.append(unresolved)
            counts[k] += [(~use).sum(), recorded.sum(), unresolved.sum()]
        parts = np.array(parts)
        changed += (parts[:, None] != parts[None, :]).sum(axis=2)
        packet = out / f"{ix:02}.npz"
        np.savez_compressed(
            packet,
            source_row=subset.astype("i4"),
            sensor=sensor,
            time_s=t,
            semantic=label,
            states=states,
            origin_local_m=o.astype("f4"),
            unit_ray_local=v.astype("f4"),
            host_range_m=r.astype("f4"),
            grade_plane_range_m=plane.astype("f4"),
            visual_range_m=np.float32(visual),
            plate_constrained_range_m=np.float32(bounded),
            rise_bracket_range_m=np.float32(envelopes),
            unresolved=np.array(unknown),
            inserted_part=parts.astype("u1"),
        )
        frame.update(
            packet_sha256=digest(packet), semantics_sha256=digest(file), candidate_rows=len(subset)
        )
    if (
        not (total_bracket[:2] > 0).all()
        or counts[2, 1] != 0
        or min(changed[np.triu_indices(3, 1)]) <= 0
    ):
        raise ValueError("Insufficient comparable returns")
    write(
        out / "report.json",
        dict(
            states=states,
            rig_sha256=digest(RIG),
            glb_sha256=hashes,
            counts={
                s: dict(zip(["kept_host", "photographed_face", "unresolved"], map(int, count)))
                for s, count in zip(states, counts)
            },
            changed_part_rows=changed.tolist(),
            height_bracket_beams=total_bracket.tolist(),
            bracket_width_median_m=[float(np.median(w)) if w else None for w in bracket_widths],
            frames=frames,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("models", "drive", "raw", "out"):
        p.add_argument(arg, type=Path)
    replay(**vars(p.parse_args()))
