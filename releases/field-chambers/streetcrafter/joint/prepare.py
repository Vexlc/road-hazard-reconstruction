"""Seed the StreetGS scene graph from training LiDAR and calibrated rig images."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


def load_rig(bundle):
    split = json.loads((bundle / "all/split.json").read_text())
    cameras = {}
    for line in (bundle / "eval/sparse/0/images.txt").read_text().splitlines():
        if not line:
            continue
        _, *values, cid, name = line.split()
        qw, qx, qy, qz, *trans = map(float, values)
        w2c = np.eye(4)
        w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, 3] = trans
        cameras[name] = w2c
    return split, cameras


def projection(points, w2c, calibration):
    _, w, h, fx, fy, cx, cy = calibration
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uv = cam[:, :2] / np.maximum(z[:, None], 0.01) * [fx, fy] + [cx, cy]
    ids = np.flatnonzero((z > 1.0) & (uv >= 0).all(1) & (uv < [w, h]).all(1))
    u, v = uv[ids].astype(int).T
    return ids, u, v, z[ids]


def unique_cloud(samples, cell):
    cloud = np.asarray(np.concatenate(samples), np.float32)
    _, ids = np.unique(np.floor(cloud[:, :3] / cell), axis=0, return_index=True)
    return cloud[ids]


def untracked(bundle, split, cams):
    # Re-observe each training LiDAR seed from several exposures of the same lens.
    # A depth buffer suppresses points hidden by a nearer surface. High temporal
    # variance can also reflect lighting or calibration, so the group has no class label.
    cloud = np.loadtxt(bundle / "gut_train/sparse/0/points3D.txt", usecols=range(1, 7), dtype=np.float32)
    xyz = cloud[:, :3]
    score = np.zeros(len(cloud))
    supported = np.zeros(len(cloud), bool)
    for sensor in split["sensors"]:
        names = [
            n
            for n in split["train"]
            if split["views"][n]["sensor"] == sensor and not split["views"][n]["tile"]
        ]
        seen = np.zeros(len(cloud))
        total = np.zeros((len(cloud), 3))
        squares = np.zeros_like(total)
        for n in names[:: max(1, len(names) // 15)]:
            cal = split["projections"]["eval"][split["views"][n]["projection"]]
            ids, u, v, z = projection(xyz, cams[n], cal)
            bins = (v // 3) * ((cal[1] + 2) // 3) + u // 3
            depth = np.full(((cal[1] + 2) // 3) * ((cal[2] + 2) // 3), np.inf)
            np.minimum.at(depth, bins, z)
            keep = z <= depth[bins] + 0.25
            ids = ids[keep]
            color = cv2.imread(str(bundle / "eval/images" / n))[v[keep], u[keep]].astype(float) / 255
            seen[ids] += 1
            total[ids] += color
            squares[ids] += color**2
        count = np.maximum(seen[:, None], 1)
        var = np.maximum(0, squares / count - (total / count) ** 2).mean(1)
        score = np.maximum(score, np.where(seen >= 3, np.sqrt(var), 0))
        supported |= seen >= 3
    threshold = max(0.055, np.quantile(score[supported], 0.84))
    mask = supported & (score >= threshold)
    if mask.sum() < 64 or (~mask).sum() < 64:
        raise ValueError("Insufficient multi-exposure support for the residual field")
    actor = cloud[mask].copy()
    center = np.median(actor[:, :3], axis=0)
    actor[:, :3] -= center
    pose = np.array([*center, 1, 0, 0, 0, 1], dtype=np.float32)
    tracks = np.tile(pose, (len(split["sensors"]), len(split["frames"]), 1, 1))
    info = {
        0: object_meta(0, 0, "untracked residual", 0, len(split["frames"]) - 1, np.ptp(actor[:, :3], axis=0))
    }
    return {"background": cloud[~mask], "obj_000": actor}, tracks, info


def object_meta(oid, slot, label, start, end, size):
    return dict(
        id=slot,
        object_id=oid,
        track_id=str(oid),
        **{"class": label},
        class_label=0,
        deformable=True,
        length=float(size[0]),
        width=float(size[1]),
        height=float(size[2]),
        start_frame=start,
        end_frame=end,
    )


def pandaset(root, scene, bundle, split, cams):
    import pandas as pd

    data = root / "work/data"
    processed = data / "processed" / scene
    annotations = json.loads((processed / "instances/instances_info.json").read_text())
    times = json.loads((processed / "timestamps.json").read_text())
    names_by_frame = {}
    for n in split["train"]:
        v = split["views"][n]
        if not v["tile"]:
            names_by_frame.setdefault(v["group"], []).append(n)
    groups = sorted(names_by_frame)
    sensors = split["sensors"]
    rng = np.random.default_rng(42)
    objects = {}
    for key, a in annotations.items():
        ann = a["frame_annotations"]
        idx = np.flatnonzero(np.isin(ann["frame_idx"], groups))
        if len(idx) < 2 or np.asarray(ann["stationary"])[idx].all():
            continue
        frames = np.asarray(ann["frame_idx"])[idx]
        matrices = np.asarray(ann["obj_to_world"])[idx]
        ts = np.asarray(times["front_camera"])[frames]
        poses = np.zeros((len(sensors), len(split["frames"]), 8), np.float32)
        rot = Slerp(ts, Rotation.from_matrix(matrices[:, :3, :3]))
        for j, sensor in enumerate(sensors):
            stamps = np.clip(times[sensor], ts.min(), ts.max())
            poses[j, :, :3] = np.column_stack([np.interp(stamps, ts, matrices[:, k, 3]) for k in range(3)])
            poses[j, :, 3:7] = rot(stamps).as_quat()[:, [3, 0, 1, 2]]
            poses[j, frames.min() : frames.max() + 1, 7] = 1
        objects[int(key)] = dict(
            label=a["class_name"],
            poses=poses,
            frames=frames,
            sizes=np.asarray(ann["box_size"])[idx],
            matrices=matrices,
            points=[],
        )
    background = []
    for frame in groups:
        scan = pd.read_pickle(data / "raw" / scene / "lidar" / f"{frame:02}.pkl.gz")
        world = scan[["x", "y", "z"]].to_numpy(dtype=float)[::2]
        tree = cKDTree(world)
        owner = np.full(len(world), -1)
        local = np.zeros_like(world)
        for oid, o in objects.items():
            where = np.flatnonzero(o["frames"] == frame)
            if not len(where):
                continue
            k = where[0]
            mat = o["matrices"][k]
            half = o["sizes"][k] / 2
            ids = np.asarray(tree.query_ball_point(mat[:3, 3], np.linalg.norm(half)), int)
            pts = (world[ids] - mat[:3, 3]) @ mat[:3, :3]
            inside = (np.abs(pts) <= half).all(1)
            owner[ids[inside]] = oid
            local[ids[inside]] = pts[inside]
        for n in names_by_frame[frame]:
            cal = split["projections"]["eval"][split["views"][n]["projection"]]
            ids, u, v, z = projection(world, cams[n], cal)
            good = z < 95
            ids, u, v = ids[good], u[good], v[good]
            img = cv2.imread(str(bundle / "eval/images" / n))
            for oid in np.unique(owner[ids]):
                choices = np.flatnonzero(owner[ids] == oid)
                ii = rng.choice(choices, min(len(choices), 1100 if oid < 0 else 220), replace=False)
                xyz = world if oid < 0 else local
                pts = np.column_stack((xyz[ids[ii]], img[v[ii], u[ii], ::-1]))
                (background if oid < 0 else objects[oid]["points"]).append(pts)
    layers = {"background": unique_cloud(background, 0.15)}
    tracks = []
    info = {}
    for oid, o in objects.items():
        if not o["points"]:
            continue
        cloud = unique_cloud(o["points"], 0.06)
        if len(cloud) < 24:
            continue
        layers[f"obj_{oid:03d}"] = cloud
        info[oid] = object_meta(
            oid, len(tracks), o["label"], int(min(o["frames"])), int(max(o["frames"])), o["sizes"].max(0)
        )
        tracks.append(o["poses"])
    if not tracks:
        raise ValueError("No supported moving PandaSet tracks")
    return layers, np.stack(tracks, axis=2), info


def prepare(root, scene):
    bundle = root / "work/streetcrafter-rig" / scene
    dest = root / "work/streetgs-joint" / scene
    if dest.is_dir():
        return
    split, cams = load_rig(bundle)
    if scene in ("055", "056"):
        layers, tracks, info = pandaset(root, scene, bundle, split, cams)
    else:
        layers, tracks, info = untracked(bundle, split, cams)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "data"
        stage.mkdir()
        np.savez(stage / "seeds.npz", **layers)
        np.save(stage / "tracks.npy", tracks)
        (stage / "objects.json").write_text(json.dumps(info, indent=2))
        stage.rename(dest)
    print(scene, {k: len(v) for k, v in layers.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("scene")
    a = p.parse_args()
    prepare(a.root, a.scene)
