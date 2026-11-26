"""Replay recorded camera and LiDAR bearings on a corroded lid and its sealed control."""
import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from road_geometry import Surface, read_json, save_json, sha256
from traffic_clearance import world_bounds

camera = import_module("check-pavement-pass")
lidar = import_module("replay-prop-lidar")


def compare(packet, dest):
    np.savez_compressed(dest, **packet)
    part = packet["part"]
    delta = (packet["corroded_range_m"] - packet["sealed_range_m"]) * 1000
    return dict(packet_sha256=sha256(dest), samples=len(part),
                first_parts={str(p): int((part == p).sum()) for p in np.unique(part)},
                deeper_than_2mm=int((delta > 2).sum()), deeper_than_50mm=int((delta > 50).sum()),
                maximum_extra_range_mm=float(delta.max(initial=0)))


def check(native, control, raw, views, report):
    mesh, sealed = Surface(native), Surface(control)
    digest = sha256(native)
    records = []
    result = dict(glb_sha256=digest, sealed_sha256=sha256(control), scene=raw.name, cameras={})
    out = views / "bearings"
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(views.glob("*/M544/placement.json")):
        record = read_json(path)
        if records and record["anchor"] != records[0]["anchor"]:
            raise ValueError("Vehicle cameras use different anchor placements")
        records.append(record)
        exp = camera.exposure(path.parent, raw, mesh, digest)
        transform = exp["matrix"]
        eye = (exp["pose"][:3, 3] - transform[:3, 3]) @ transform[:3, :3]
        vec = exp["xyz"] - eye
        distance = np.linalg.norm(vec, axis=1)
        direction = vec / distance[:, None]
        baseline, label, _ = sealed.cast(np.broadcast_to(eye, vec.shape), direction)
        part = exp["parts"][exp["row"], exp["col"]]
        keep = (label == 163) & (part > 1)
        packet = dict(uv=exp["uv"][keep], local_hit=exp["xyz"][keep], part=part[keep],
                      corroded_range_m=distance[keep], sealed_range_m=baseline[keep])
        result["cameras"][record["camera"]] = dict(frame=exp["frame"], sensor_frames=len(record["exposures"]),
                subpixel_error_mm=1000 * exp["error_m"], **compare(packet, out / (record["camera"] + ".npz")))
    anchor = records[0]["anchor"]
    basis, center = np.array(anchor["basis_columns"]), np.array(anchor["center_world"])
    bounds = world_bounds(anchor, (mesh.low, mesh.high))
    road = next(int(i) for i, name in read_json(raw/"annotations/semseg/classes.json").items() if name == "Road")
    traffic = lidar.pandaset(raw)[0]
    chunks, hashes, excluded = [], {}, {}
    for index in sorted({e["scan"] for r in records for e in r["exposures"]}):
        path = raw/"lidar"/f"{index:02}.pkl.gz"
        origin, bearing, prior, times, emitters, signature = lidar.scan(path)
        hashes[index] = signature
        try:
            traffic.require_window({"M544":bounds}, float(times.min()), float(times.max()), .25)
        except ValueError as exc:
            excluded[index] = str(exc)
            continue
        local, ray = (origin-center) @ basis, bearing @ basis
        tags = pd.read_pickle(raw/"annotations/semseg"/path.name)["class"].to_numpy()
        ids = np.flatnonzero((tags==road) & (ray[:,2]<0) & (abs(local[:,2]+ray[:,2]*prior)<.055) &
                             lidar.possible(local, ray, (mesh.low-.001, mesh.high+.001)))
        if not len(ids):
            continue
        distance, part, _ = mesh.cast(local[ids], ray[ids])
        baseline, label, _ = sealed.cast(local[ids], ray[ids])
        keep = (label==163) & (part>1)
        ids = ids[keep]
        xyz = pd.read_pickle(path).iloc[ids][["x","y","z"]].to_numpy()
        np.testing.assert_allclose(origin[ids]+prior[ids,None]*bearing[ids], xyz, atol=1e-7, rtol=0)
        chunks.append(dict(scan=np.full(len(ids),index), raw_row=ids, time_s=times[ids], emitter=emitters[ids],
                           origin_world=origin[ids], direction_world=bearing[ids], original_range_m=prior[ids],
                           part=part[keep], corroded_range_m=distance[keep], sealed_range_m=baseline[keep]))
    packet = {k:np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
    result["lidar"] = dict(scans_sha256=hashes, excluded_scans=excluded, **compare(packet,out/"lidar.npz"))
    save_json(report, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native","control","raw","views","report"):
        parser.add_argument(name,type=Path)
    check(**vars(parser.parse_args()))
