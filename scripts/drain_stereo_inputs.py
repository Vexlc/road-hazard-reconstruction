"""Make two image-matched road observations and independent geometric fisheye controls."""

from importlib import import_module

import cv2
import numpy as np
from PIL import Image

from prop_registration import rectified_disparities

cameras = import_module("check-mei-pits")
ROOT, read, save = cameras.ROOT, cameras.read, cameras.save
EXAMPLES = ROOT / "gen2hole/examples/stereo-drains"


def pack_view(folder, name, pose, rgb, lens, xy, protected, **labels):
    h, w = protected.shape
    save(folder / (name + ".json"), lens)
    np.savez_compressed(
        folder / (name + ".npz"),
        camera_to_world=pose,
        rgb=rgb,
        sensor_window_xywh=[*xy, w, h],
        sample_offsets_px=[[0.0, 0.0]],
        protected=protected,
        **labels,
    )


def pair_maps(folder, xy, reverse, valid):
    np.savez_compressed(
        folder / "matches.npz",
        right_xy=xy.astype("f4"),
        left_xy=reverse.astype("f4"),
        match_valid=valid,
    )


def road(views, dest, pairs):
    drive = import_module("render-mixed-road")
    for site, spec in read(EXAMPLES / "road-windows.json").items():
        raw = ROOT / "work/data" / spec["raw"]
        data, cams = drive.load_rig(ROOT / "work/streetcrafter-rig" / spec["scene"])
        frame = spec["frame"]
        g = data["frames"].index(frame)
        d = dest / site
        d.mkdir(parents=True, exist_ok=True)
        path = raw / "image_00" / f"{frame:010}.png"
        dl, dr = rectified_disparities(path)
        h, w = dl.shape
        top = 205
        y, x = np.indices((h - top, w))
        poly = cv2.fillPoly(
            np.zeros((h, w), "u1"), [np.array(spec["road"], "i4")], 1
        ).astype(bool)[top:]
        poses = [np.linalg.inv(cams[f"{g:03}{i}0.png"]) for i in range(2)]
        for i, name in enumerate(("primary", "secondary")):
            sensor = f"image_0{i}"
            k = np.array(data["source_calibration"][sensor]["K"])
            rgb = np.array(Image.open(raw / sensor / path.name).convert("RGB"))[top:]
            lens = ["PINHOLE", w, h, k[0, 0], k[1, 1], k[0, 2] + 0.5, k[1, 2] + 0.5]
            pack_view(
                d,
                name,
                poses[i],
                rgb,
                lens,
                [0, top],
                np.zeros_like(poly),
                **({"receiver_mask": poly, "sensor_name": sensor} if i == 0 else {}),
            )
        reverse = np.stack([x - dr[top:], y], -1)
        reverse[(dr[top:] >= 0) | (dr[top:] < -159)] = np.nan
        pair_maps(d, np.stack([x - dl[top:], y], -1), reverse, dl[top:] > 1)
        xyz = drive.scans(raw, data, cams)[g]
        q = (xyz - poses[0][:3, 3]) @ poses[0][:3, :3]
        q = q[(q[:, 2] > 1) & (np.linalg.norm(q, axis=1) < 25)]
        k = np.array(data["source_calibration"]["image_00"]["K"])
        u, v = np.rint(((q @ k.T) / q[:, 2, None])[:, :2]).astype(int).T
        use = (v >= top) & (v < h) & (u >= 0) & (u < w)
        truth = np.full(poly.shape, np.inf, "f4")
        np.minimum.at(truth, (v[use] - top, u[use]), np.linalg.norm(q[use], axis=1))
        truth[~poly] = np.inf
        np.savez_compressed(d / "truth.npz", road_lidar_range=truth)
        save(
            d / "survey.json",
            {p: read(views / p / site / "anchor.json") for p in pairs},
        )
        save(
            d / "source.json",
            dict(
                **spec,
                raw_sha256={
                    s: cameras.digest(raw / s / path.name)
                    for s in ("image_00", "image_01")
                },
                crop=[0, top, w, h - top],
            ),
        )


def project(xyz, camera):
    pose = np.asarray(camera["camera_to_world"])
    q = (xyz - pose[:3, 3]) @ pose[:3, :3]
    rho = np.linalg.norm(q[..., :2], axis=-1)
    theta = np.arctan2(rho, q[..., 2])
    w = camera["size"]
    return (w - 1) / 2 + q[..., :2] * (
        w / np.deg2rad(camera["fov_deg"]) * theta / np.maximum(rho, 1e-12)
    )[..., None]


def controls(views, dest, pairs):
    for pair in pairs:
        for shot in ("side", "rear"):
            folder = views / "cycles" / pair / shot
            d = dest / pair / shot
            d.mkdir(exist_ok=True, parents=True)
            hosts = [folder / side / "host" for side in ("left", "right")]
            cams = [read(p / "camera.json") for p in hosts]
            ids = [np.load(p / "id.npy") for p in hosts]
            pos = [np.load(p / "position.npy") for p in hosts]
            for i, name in enumerate(("primary", "secondary")):
                w = cams[i]["size"]
                f = w / np.deg2rad(cams[i]["fov_deg"])
                pack_view(
                    d,
                    name,
                    cams[i]["camera_to_world"],
                    np.array(Image.open(hosts[i] / "rgb.png").convert("RGB")),
                    ["OPENCV_FISHEYE", w, w, f, f, w / 2, w / 2, 0, 0, 0, 0],
                    [0, 0],
                    ids[i] == 0,
                    **(
                        {"receiver_mask": ids[i] == 1, "occluder_mask": ids[i] == 2}
                        if i == 0
                        else {}
                    ),
                )
            reverse = project(pos[1], cams[0])
            reverse[ids[1] == 0] = np.nan
            pair_maps(d, project(pos[0], cams[1]), reverse, ids[0] > 0)
            native = folder / "left/native"
            np.savez_compressed(
                d / "truth.npz",
                host_id=ids[0],
                host_range=np.load(hosts[0] / "range.npy"),
                native_id=np.load(native / "id.npy"),
                native_range=np.load(native / "range.npy"),
            )
            save(d / "ground.json", cams[0]["reference_ground_to_world"])


def prepare(views, dest, pairs):
    road(views, dest, pairs)
    controls(views, dest, pairs)
