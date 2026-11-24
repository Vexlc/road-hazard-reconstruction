"""Complete the synchronized A2D2 and KITTI-360 camera rigs for the local clips."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import time
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

from download_a2d2 import BASE, DRIVE, FRAMES
from download_kitti import BASE as KITTI_BASE, fetch


def audi(raw):
    sensors = json.loads((raw / "cams_lidars.json").read_text())["cameras"]
    refs = [json.loads((raw / f"{i:09d}.json").read_text())["cam_tstamp"] for i in FRAMES]

    def get(sensor, frame, ext):
        root = raw if sensor == "front_left" else raw / sensor
        path = root / f"{frame:09d}.{ext}"
        if path.exists():
            return path
        root.mkdir(exist_ok=True)
        kind = "lidar" if ext == "npz" else "camera"
        name = f"{DRIVE.replace('_', '')}_{kind}_{sensor.replace('_', '')}_{frame:09d}.{ext}"
        with requests.Session() as s:
            s.mount(
                "https://",
                HTTPAdapter(
                    max_retries=Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
                ),
            )
            for attempt in range(6):
                try:
                    r = s.get(f"{BASE}camera_lidar/{DRIVE}/{kind}/cam_{sensor}/{name}", timeout=90)
                    break
                except requests.RequestException:
                    if attempt == 5:
                        raise
                    time.sleep(1 + attempt)
            r.raise_for_status()
            with NamedTemporaryFile(dir=root, delete=False) as f:
                f.write(r.content)
            Path(f.name).replace(path)
        return path

    index = {}
    with ThreadPoolExecutor(16) as pool:
        for sensor in sensors:
            if sensor == "front_left":
                index[sensor] = list(FRAMES)
                continue

            def stamp(frame):
                return json.loads(get(sensor, frame, "json").read_text())["cam_tstamp"]

            t0, t1 = pool.map(stamp, [FRAMES.start, FRAMES.start + 1])
            offset = round((refs[0] - t0) / (t1 - t0))
            candidates = list(range(FRAMES.start + offset - 4, FRAMES.stop + offset + 4))
            stamps = np.asarray(list(pool.map(stamp, candidates)))
            nearest = np.abs(stamps[:, None] - refs).argmin(axis=0)
            err = np.max(np.abs(stamps[nearest] - refs))
            if err > 17_000 or len(set(nearest)) != len(refs):
                raise ValueError(f"{sensor}: no unique synchronized observation, skew {err} us")
            ids = [candidates[j] for j in nearest]
            index[sensor] = ids
            print(sensor, ids[0], ids[-1], err, flush=True)
            jobs = [(sensor, i, ext) for i in ids for ext in ["png", "npz"]]
            list(pool.map(lambda t: get(*t), jobs))
    (raw / "rig_frames.json").write_text(json.dumps(index, indent=2))


def kitti(raw, drive_id, start, count):
    raw.mkdir(parents=True, exist_ok=True)
    drive = f"2013_05_28_drive_{drive_id}_sync"
    cams = [f"image_0{i}" for i in range(4)]

    def camera(cam):
        folder = "data_rect" if cam in cams[:2] else "data_rgb"
        fetch(
            raw,
            f"data_2d_raw/{drive}_{cam}.zip",
            {
                f"{drive}/{cam}/{folder}/{i:010}.png": f"{cam}/{i:010}.png"
                for i in range(start, start + count)
            },
        )

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(camera, cams))
    fetch(
        raw,
        f"data_3d_raw/{drive}_velodyne.zip",
        {
            f"{drive}/velodyne_points/data/{i:010}.bin": f"velodyne/{i:010}.bin"
            for i in range(start, start + count)
        },
    )

    def metadata(archive, members):
        if all((raw / path).exists() for path in members.values()):
            return
        for attempt in range(6):
            try:
                response = requests.get(KITTI_BASE + archive, timeout=60)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for key, dest in members.items():
                        path = raw / dest
                        path.parent.mkdir(parents=True, exist_ok=True)
                        payload = z.read(key)
                        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                            tmp.write(payload)
                        Path(tmp.name).replace(path)
                return
            except (requests.RequestException, zipfile.BadZipFile):
                if attempt == 5:
                    raise
                time.sleep(1 + attempt)

    metadata("89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip", {f"{drive}/poses.txt": "poses.txt"})
    for group, cameras in [("perspective", cams[:2]), ("fisheye", cams[2:])]:
        metadata(
            f"data_2d_raw/data_timestamps_{group}.zip",
            {f"{drive}/{c}/timestamps.txt": f"{c}/timestamps.txt" for c in cameras},
        )
    names = [
        "perspective.txt",
        "image_02.yaml",
        "image_03.yaml",
        "calib_cam_to_pose.txt",
        "calib_cam_to_velo.txt",
    ]
    metadata(
        "384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip",
        {f"calibration/{n}": f"calibration/{n}" for n in names},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", choices=["a2d2-frontleft", "kitti360-0000", "kitti360-0003-right"])
    p.add_argument("data", type=Path)
    a = p.parse_args()
    if a.scene == "a2d2-frontleft":
        audi(a.data / a.scene)
    elif a.scene == "kitti360-0000":
        kitti(a.data / "kitti360-0000-0200", "0000", 200, 48)
    else:
        kitti(a.data / a.scene, "0003", 480, 64)
