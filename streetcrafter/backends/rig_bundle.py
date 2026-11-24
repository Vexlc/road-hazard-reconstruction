"""Build shared, timestamp-grouped COLMAP scenes for both StreetCrafter backends."""

import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fisheye_bundle import held_out


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def pin_params(k, size, target):
    sx, sy = np.array(target) / size
    return [k[0, 0] * sx, k[1, 1] * sy, (k[0, 2] + 0.5) * sx, (k[1, 2] + 0.5) * sy]


class Bundle:
    def __init__(self, root, scene, sensors, frames):
        self.root, self.sensors = root, sensors
        self.meta = dict(
            scene=scene,
            sensors=sensors,
            frames=list(frames),
            train=[],
            test=[],
            times={},
            views={},
            source_calibration={},
            projections={},
            seed_points_by_sensor={},
        )
        self.rows, self.cameras = {}, {}
        for part in ("all", "gut_train", "eval"):
            (root / part / "images").mkdir(parents=True)
            (root / part / "sparse/0").mkdir(parents=True)
            self.rows[part] = []
            self.cameras[part] = {}
        self.cloud = []
        self.rng = np.random.default_rng(42)

    def add(self, i, cam, time, source_frame, pose, pinhole, k, gut=None, extras=()):
        sensor = self.sensors[cam]
        prefix = i * 100 + cam * 10
        test = held_out(i)

        def write(part, tile, rgb, model, params, c2w, link=None):
            name = f"{prefix + tile:05}.png"
            path = self.root / part / "images" / name
            if link is not None:
                os.link(self.root / link / "images" / name, path)
            elif not cv2.imwrite(str(path), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                raise IOError(path)
            width, height = rgb.shape[1::-1]
            cid = cam * 10 + tile + 1
            calib = [model, width, height, *map(float, params)]
            if self.cameras[part].setdefault(cid, calib) != calib:
                raise ValueError(f"Calibration changed for {sensor}")
            w2c = np.linalg.inv(c2w)
            q = Rotation.from_matrix(w2c[:3, :3]).as_quat()[[3, 0, 1, 2]]
            self.rows[part].append(
                " ".join(map(str, [prefix + tile + 1, *q, *w2c[:3, 3], cid, name])) + "\n\n"
            )
            if part == "all":
                self.meta["times"][name] = float(time)
                self.meta["test" if test else "train"].append(name)
                self.meta["views"][name] = dict(
                    sensor=sensor, group=i, source_frame=int(source_frame), tile=tile, projection=str(cid)
                )
            self.meta["projections"].setdefault(part, {})[str(cid)] = calib

        write("all", 0, pinhole, "PINHOLE", k, pose)
        write("eval", 0, pinhole, "PINHOLE", k, pose, "all")
        if not test:
            if gut is None:
                write("gut_train", 0, pinhole, "PINHOLE", k, pose, "all")
            else:
                image, model, params = gut
                write("gut_train", 0, image, model, params, pose)
            for tile, (image, params, relative) in enumerate(extras, 1):
                write("all", tile, image, "PINHOLE", params, pose @ relative)

    def seed(self, sensor, cam, uv, img, pose):
        w, h = img.shape[1::-1]
        valid = np.flatnonzero(
            (cam[:, 2] > 1.5)
            & (np.linalg.norm(cam, axis=1) < 100)
            & ((uv >= 0) & (uv < [w - 1, h - 1])).all(axis=1)
        )
        if not len(valid):
            raise ValueError(f"Missing LiDAR coverage for {sensor}")
        ids = self.rng.choice(valid, min(1000, len(valid)), replace=False)
        u, v = np.rint(uv[ids]).astype(int).T
        self.cloud.append(np.column_stack((transform(cam[ids], pose), img[v, u, ::-1])))
        counts = self.meta["seed_points_by_sensor"]
        counts[sensor] = counts.get(sensor, 0) + len(ids)

    def finish(self):
        groups = {key: {self.meta["views"][n]["group"] for n in self.meta[key]} for key in ("train", "test")}
        if groups["test"] & groups["train"]:
            raise ValueError("Timestamp leaked across splits")
        for i in range(len(self.meta["frames"])):
            observed = [
                v["sensor"] for v in self.meta["views"].values() if v["group"] == i and v["tile"] == 0
            ]
            if sorted(observed) != sorted(self.sensors):
                raise ValueError(f"Incomplete rig at group {i}")
        cloud = np.concatenate(self.cloud)
        _, idx = np.unique(np.floor(cloud[:, :3] / 0.15), axis=0, return_index=True)
        cloud = cloud[idx]
        self.meta["seed_points"] = len(cloud)
        path = self.root / "all/sparse/0/points3D.txt"
        np.savetxt(
            path, np.column_stack((np.arange(1, len(cloud) + 1), cloud)), fmt="%d %.6f %.6f %.6f %d %d %d 0"
        )
        for part, rows in self.rows.items():
            sparse = self.root / part / "sparse/0"
            (sparse / "images.txt").write_text("".join(rows))
            (sparse / "cameras.txt").write_text(
                "".join(" ".join(map(str, [cid, *c])) + "\n" for cid, c in sorted(self.cameras[part].items()))
            )
            if part != "all":
                os.link(path, sparse / path.name)
        (self.root / "all/split.json").write_text(json.dumps(self.meta, indent=2))
