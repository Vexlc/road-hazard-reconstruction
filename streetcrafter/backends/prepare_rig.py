"""Prepare all physical cameras for one driving clip in a common metric world."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
import yaml

from fisheye_bundle import held_out
from rig_bundle import Bundle, pin_params, transform
from prepare_kitti import homogeneous, project_mei, remap_grid

PANDA = [
    "front_camera",
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_camera",
]
SIZE = (480, 300)


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def perspective_uv(cam, k):
    norm = cam[:, :2] / np.maximum(cam[:, 2:], 1e-8)
    return norm * [k[0, 0], k[1, 1]] + k[:2, 2]


def pandaset(scene, data, dest):
    root = data / "processed" / scene
    times = json.loads((root / "timestamps.json").read_text())
    frames = range(len(times["frame"]))
    bundle = Bundle(dest, scene, PANDA, frames)
    for i in frames:
        if not held_out(i):
            df = pd.read_pickle(data / "raw" / scene / "lidar" / f"{i:02}.pkl.gz")
            world = df[["x", "y", "z"]].to_numpy()[::2]
        for j, sensor in enumerate(PANDA):
            img = read_image(data / "raw" / scene / "camera" / sensor / f"{i:02}.jpg")
            fx, fy, cx, cy = np.loadtxt(root / "intrinsics" / f"{j}.txt")[:4]
            k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            pose = np.loadtxt(root / "extrinsics" / f"{i:03}_{j}.txt")
            size = img.shape[1::-1]
            target = (480, round(480 * size[1] / size[0]))
            bundle.meta["source_calibration"][sensor] = dict(
                model="PINHOLE",
                size=size,
                K=k.tolist(),
                camera_to_world=f"work/data/processed/{scene}/extrinsics/{{frame:03}}_{j}.txt",
            )
            bundle.add(
                i,
                j,
                times[sensor][i] - times["frame"][0],
                i,
                pose,
                cv2.resize(img, target, interpolation=cv2.INTER_AREA),
                pin_params(k, size, target),
            )
            if not held_out(i):
                cam = transform(world, np.linalg.inv(pose))
                bundle.seed(sensor, cam, perspective_uv(cam, k), img, pose)
    bundle.finish()


def view_grids(project, size, yaws, fov):
    # Find calibrated perspective crops entirely inside the recorded sensor.
    w, h = SIZE
    y, x = np.mgrid[:h, :w]
    result = []
    for yaw in yaws:
        rel = np.eye(4)
        rel[:3, :3] = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
        for aperture in range(fov, 19, -2):
            f = w / (2 * np.tan(np.deg2rad(aperture / 2)))
            rays = np.stack(((x + 0.5 - w / 2) / f, (y + 0.5 - h / 2) / f, np.ones_like(x)), axis=-1)
            mapping = project(rays @ rel[:3, :3].T)
            if np.isfinite(mapping).all() and ((mapping >= 0) & (mapping < np.array(size) - 1)).all():
                result.append((mapping.astype("float32"), [f, f, w / 2, h / 2], rel))
                break
        else:
            raise ValueError(f"No calibrated crop for yaw {yaw}")
    return result


def pin_views(img, grids, test):
    views = [
        (cv2.remap(img, grid, None, cv2.INTER_LINEAR), params, rel)
        for grid, params, rel in (grids[:1] if test else grids)
    ]
    return views[0][0], views[0][1], views[1:]


def kitti(scene, data, dest):
    short = scene == "kitti360-0000"
    raw = data / ("kitti360-0000-0200" if short else scene)
    start, count = (200, 48) if short else (480, 64)
    frames = np.arange(start, start + count)
    sensors = [f"image_0{i}" for i in range(4)]
    bundle = Bundle(dest, scene, sensors, frames.tolist())
    folder = raw / "calibration"

    def numbers(path):
        return {
            k: np.fromstring(v, sep=" ")
            for k, v in (
                line.split(":", 1)
                for line in path.read_text().splitlines()
                if ":" in line and not line.startswith("calib_time")
            )
        }

    rig = {c: homogeneous(a) for c, a in numbers(folder / "calib_cam_to_pose.txt").items()}
    velo = rig["image_00"] @ np.linalg.inv(homogeneous(np.loadtxt(folder / "calib_cam_to_velo.txt")))
    stereo = numbers(folder / "perspective.txt")
    poses = np.loadtxt(raw / "poses.txt")
    stamps = {
        c: np.char.replace(
            np.loadtxt(raw / c / "timestamps.txt", delimiter="\t", dtype=str)[frames], " ", "T"
        ).astype("datetime64[ns]")
        for c in sensors
    }
    epoch = stamps[sensors[0]][0]
    seconds = {c: (t - epoch) / np.timedelta64(1, "s") for c, t in stamps.items()}
    clock = np.char.replace(
        np.loadtxt(raw / sensors[0] / "timestamps.txt", delimiter="\t", dtype=str), " ", "T"
    ).astype("datetime64[ns]")
    tref = (clock[poses[:, 0].astype(int)] - epoch) / np.timedelta64(1, "s")
    slerp = Slerp(tref, Rotation.from_matrix(poses[:, 1:].reshape(-1, 3, 4)[:, :, :3]))

    def body(t):
        t = float(np.clip(t, tref[0], tref[-1]))
        return homogeneous(
            np.column_stack(
                (
                    slerp(t).as_matrix(),
                    [np.interp(t, tref, poses[:, j]) - np.interp(0, tref, poses[:, j]) for j in (4, 8, 12)],
                )
            )
        )

    calibs, grids, fish_grids, ks = {}, {}, {}, {}
    for j, c in enumerate(sensors):
        if j < 2:
            n = c[-2:]
            k = stereo["P_rect_" + n].reshape(3, 4)[:, :3]
            ks[c] = k
            rect = np.eye(4)
            rect[:3, :3] = stereo["R_rect_" + n].reshape(3, 3).T
            rig[c] = rig[c] @ rect
            rec = dict(model="PINHOLE rectified", K=k.tolist(), size=stereo["S_rect_" + n].tolist())
        else:
            spec = yaml.safe_load((folder / (c + ".yaml")).read_text().split("\n", 1)[1])
            calibs[c] = spec
            size = (spec["image_width"], spec["image_height"])
            grids[c] = view_grids(lambda ray: project_mei(ray, spec), size, [0, -35, 35], 92)
            fish_grids[c] = remap_grid((400, 400), 400 / np.deg2rad(125), True, spec)
            rec = dict(model="MEI", parameters=spec, size=size)
        bundle.meta["source_calibration"][c] = dict(**rec, camera_to_pose=rig[c].tolist())
    for i, frame in enumerate(frames):
        test = held_out(i)
        if not test:
            scan = np.fromfile(raw / "velodyne" / f"{frame:010}.bin", "<f4").reshape(-1, 4)[:, :3]
            world = transform(scan, body(seconds[sensors[0]][i]) @ velo)
        for j, c in enumerate(sensors):
            img = read_image(raw / c / f"{frame:010}.png")
            t = seconds[c][i]
            pose = body(t) @ rig[c]
            gut, extra = None, []
            if j < 2:
                w, h = img.shape[1::-1]
                target = (640, round(640 * h / w))
                pin = cv2.resize(img, target, interpolation=cv2.INTER_AREA)
                params = pin_params(ks[c], (w, h), target)
            else:
                pin, params, extra = pin_views(img, grids[c], test)
                f = 400 / np.deg2rad(125)
                gut = (
                    cv2.remap(img, fish_grids[c], None, cv2.INTER_LINEAR),
                    "OPENCV_FISHEYE",
                    [f, f, 200, 200, 0, 0, 0, 0],
                )
            bundle.add(i, j, t, frame, pose, pin, params, gut, extra)
            if not test:
                cam = transform(world, np.linalg.inv(pose))
                cam = cam[cam[:, 2] > 1.5]
                uv = perspective_uv(cam, ks[c]) if j < 2 else project_mei(cam, calibs[c])
                bundle.seed(c, cam, uv, img, pose)
    bundle.finish()


def audi(scene, work, dest):
    raw = work / "data" / scene
    cameras = json.loads((raw / "cams_lidars.json").read_text())["cameras"]
    index = json.loads((raw / "rig_frames.json").read_text())
    sensors = list(cameras)
    frames = index["front_left"]
    bundle = Bundle(dest, scene, sensors, frames)
    ref = work / "streetcrafter-backends" / scene / "fish_all/sparse/0/images.txt"
    ref_poses = []
    for line in ref.read_text().splitlines():
        if line.strip():
            _, qw, qx, qy, qz, tx, ty, tz, _, name = line.split()
            inv = homogeneous(
                np.column_stack(
                    (
                        Rotation.from_quat(np.array([qx, qy, qz, qw], float)).as_matrix(),
                        np.array([tx, ty, tz], float),
                    )
                )
            )
            if int(Path(name).stem) != len(ref_poses):
                raise ValueError("Unexpected reference pose order")
            ref_poses.append(np.linalg.inv(inv))
    models, projectors, grids, rig = {}, {}, {}, {}
    for sensor, spec in cameras.items():
        k = np.array(spec["CamMatrixOriginal"])
        d = np.ravel(spec["Distortion"])
        fish = spec["Lens"] == "Fisheye"

        def project(rays, k=k, d=d, fish=fish):
            shape = rays.shape
            fn = cv2.fisheye.projectPoints if fish else cv2.projectPoints
            return fn(
                np.ascontiguousarray(rays.reshape(-1, 1, 3), dtype=float), np.zeros(3), np.zeros(3), k, d
            )[0].reshape(*shape[:-1], 2)

        grids[sensor] = view_grids(
            project, spec["Resolution"], [0, -24, 24] if fish else [0], 92 if fish else 60
        )
        projectors[sensor] = project
        models[sensor] = (
            "OPENCV_FISHEYE" if fish else "FULL_OPENCV",
            pin_params(k, spec["Resolution"], SIZE) + (list(d) if fish else list(d) + [0.0, 0.0, 0.0]),
        )
        view = spec["view"]
        x = np.array(view["x-axis"])
        x /= np.linalg.norm(x)
        y = np.array(view["y-axis"]) - np.dot(x, view["y-axis"]) * x
        y /= np.linalg.norm(y)
        rig[sensor] = homogeneous(np.column_stack((-y, -np.cross(x, y), x, view["origin"])))
        bundle.meta["source_calibration"][sensor] = dict(
            model=spec["Lens"],
            size=spec["Resolution"],
            K=k.tolist(),
            D=d.tolist(),
            optical_to_body=rig[sensor].tolist(),
        )
    refs = np.array(
        [json.loads((raw / f"{f:09}.json").read_text())["cam_tstamp"] for f in frames], dtype="int64"
    )
    times = (refs - refs[0]) / 1e6
    bodies = np.array(ref_poses) @ np.linalg.inv(rig["front_left"])
    slerp = Slerp(times, Rotation.from_matrix(bodies[:, :3, :3]))
    for i, frame in enumerate(frames):
        test = held_out(i)
        for j, c in enumerate(sensors):
            fid = index[c][i]
            folder = raw if c == "front_left" else raw / c
            meta = json.loads((folder / f"{fid:09}.json").read_text())
            if meta["cam_name"] != c or meta["image_zoom"] != 1:
                raise ValueError(f"Unexpected sensor or zoom for {c}/{fid}")
            t = (meta["cam_tstamp"] - refs[0]) / 1e6
            tc = float(np.clip(t, times[0], times[-1]))
            body = homogeneous(
                np.column_stack(
                    (slerp(tc).as_matrix(), [np.interp(tc, times, bodies[:, a, 3]) for a in range(3)])
                )
            )
            pose = body @ rig[c]
            img = read_image(folder / f"{fid:09}.png")
            pin, params, extra = pin_views(img, grids[c], test)
            gut = (cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA), *models[c])
            bundle.add(i, j, t, fid, pose, pin, params, gut, extra)
            if not test:
                with np.load(folder / f"{fid:09}.npz") as scan:
                    forward, left, up = scan["pcloud_points"][scan["pcloud_attr.valid"]].T
                cam = np.column_stack((-left, -up, forward))
                cam = cam[cam[:, 2] > 1.5]
                bundle.seed(c, cam, projectors[c](cam), img, pose)
    bundle.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("work", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.dest.parent) as tmp:
        dest = Path(tmp) / "bundle"
        if args.scene in ("055", "056"):
            pandaset(args.scene, args.work / "data", dest)
        elif args.scene.startswith("kitti360-"):
            kitti(args.scene, args.work / "data", dest)
        elif args.scene == "a2d2-frontleft":
            audi(args.scene, args.work, dest)
        else:
            raise ValueError(args.scene)
        dest.rename(args.dest)


if __name__ == "__main__":
    main()
