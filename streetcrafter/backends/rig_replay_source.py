"""Export calibrated fisheye targets and matching pinhole cameras for rig replay."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from prepare_kitti import remap_grid


def export(bundle, data):
    target = bundle / "sensor_eval"
    if target.exists():
        return
    meta = json.loads((bundle / "all/split.json").read_text())
    scene = meta["scene"]
    calibs = meta["projections"]["gut_train"]
    grids = {}
    with TemporaryDirectory(dir=bundle) as tmp:
        stage = Path(tmp) / "sensor_eval"
        sparse = stage / "sparse/0"
        sparse.mkdir(parents=True)
        (stage / "images").mkdir()
        for name in ["points3D.txt", "images.txt"]:
            os.link(bundle / "eval/sparse/0" / name, sparse / name)
        rows = []
        for cid, cal in calibs.items():
            row = cal if cal[0] == "OPENCV_FISHEYE" else meta["projections"]["eval"][cid]
            rows.append(" ".join(map(str, [cid, *row])))
        (sparse / "cameras.txt").write_text("\n".join(rows) + "\n")
        for name, v in meta["views"].items():
            if v["tile"]:
                continue
            path = stage / "images" / name
            cid = v["projection"]
            if calibs[cid][0] != "OPENCV_FISHEYE":
                os.link(bundle / "eval/images" / name, path)
                continue
            original = bundle / "gut_train/images" / name
            if original.exists():
                os.link(original, path)
                continue
            sensor, frame = v["sensor"], v["source_frame"]
            _, w, h, fx, *_ = calibs[cid]
            if scene.startswith("kitti"):
                raw = data / ("kitti360-0000-0200" if scene == "kitti360-0000" else scene)
                source = raw / sensor / f"{frame:010}.png"
                if cid not in grids:
                    grids[cid] = remap_grid(
                        (w, h), fx, True, meta["source_calibration"][sensor]["parameters"]
                    )
                img = cv2.remap(cv2.imread(str(source)), grids[cid], None, cv2.INTER_LINEAR)
            else:
                raw = data / scene
                source = (raw if sensor == "front_left" else raw / sensor) / f"{frame:09}.png"
                img = cv2.resize(cv2.imread(str(source)), (w, h), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(path), img):
                raise IOError(path)
        stage.rename(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    export(args.bundle, args.data)
