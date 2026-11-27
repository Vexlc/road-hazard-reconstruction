"""Replay successive raised covers at their photo-derived spacing on two independently fitted road planes."""

import argparse
from contextlib import closing
from importlib import import_module
from pathlib import Path
import sys
import imageio_ffmpeg
import numpy as np
from PIL import Image
from build import HERE, KEY
from photo_mesh import LIB

sys.path.insert(0, str(LIB.parent / "scripts"))
drv = import_module("render-road-distress")
from camera_geometry import CAMERAS
from mesh_transfer import RoadMesh, fit_road
from traffic_safe_drive import read, write, digest
from traffic_clearance import world_bounds


def install(raw, models, drive):
    near = read(drive / "raised/front" / KEY / "placement.json")["anchor"]
    cal = read(models / "registration.json")["calibration"]
    front = raw / "camera/front_camera"
    poses = read(front / "poses.json")
    heading = drv.drive.pose_matrix(poses[15])[:3, 2]
    norm = np.asarray(near["basis_columns"])[:, 2]
    forward = heading - norm * (heading @ norm)
    forward /= np.linalg.norm(forward)
    seed = np.array(near["center_world"]) + cal["centre_separation_m"] * forward
    points = []
    for i in range(0, len(poses), 3):
        road = drv.drive.road_points(raw, i)
        points.append(road[np.linalg.norm(road - seed, axis=1) < 2.0])
    far = fit_road(
        np.concatenate(points),
        seed,
        np.array(near["basis_columns"])[:, 0],
        up_world=[0, 0, 1],
        radius_m=1.05,
    )
    far["relief_daylight"] = near["relief_daylight"]
    return [near, far], cal["centre_separation_m"]


def replay(raw, masks, models, drive, out):
    out.mkdir(parents=True, exist_ok=True)
    anchors, spacing = install(raw, models, drive)
    folder = models / "raised/model/series/assets/raised" / KEY
    mesh = RoadMesh(folder, filter_pavement=True, measured_rim=True)
    survey = drv.drive.pandaset(raw)[0]
    track = survey.evaluate({str(i): world_bounds(a, mesh.bounds) for i, a in enumerate(anchors)})
    times = np.asarray(read(raw / "lidar/timestamps.json"))
    result = dict(
        covers="Two instances of the near casting",
        nominal_spacing_m=spacing,
        actual_spacing_3d_m=float(
            np.linalg.norm(np.subtract(*[a["center_world"] for a in anchors]))
        ),
        anchors=anchors,
        mesh_sha256=digest(folder / "asset.glb"),
        cameras={},
    )
    for side in ["front", "back"]:
        camera = side + "_camera"
        cam = raw / "camera" / camera
        ts = read(cam / "timestamps.json")
        poses = read(cam / "poses.json")
        model = drv.pinhole(cam, read(cam / "intrinsics.json"), 15)
        (w, h), proj, _ = model
        window = drv.drive.renderer.RayWindow(model)
        cutoff = round(0.73 * h) if side == "back" else h
        ego = dict(grid=[w, h], edge=[[0, cutoff], [w, cutoff]])
        index = CAMERAS.index(camera)
        frames = []
        best = 0
        dest = out / side
        dest.mkdir(exist_ok=True)
        with closing(
            imageio_ffmpeg.write_frames(
                str(dest / "drive.mp4"),
                (960, 540),
                fps=10,
                codec="libx264",
                macro_block_size=1,
                output_params=["-threads", "2", "-crf", "16"],
            )
        ) as writer:
            writer.send(None)
            for frame, t in enumerate(ts):
                clear = survey.exposure(track, t)
                if not clear["clear"]:
                    continue
                pose = drv.drive.pose_matrix(poses[frame])
                centres = (np.array([a["center_world"] for a in anchors]) - pose[:3, 3]) @ pose[
                    :3, :3
                ]
                possible = (centres[:, 2] > 1) & (centres[:, 2] < 40)
                if not possible.any():
                    continue
                scan = int(np.argmin(abs(times - t)))
                path = cam / f"{frame:02}.jpg"
                im = np.asarray(Image.open(path).convert("RGB"))
                after = im.copy()
                distance = np.full((h, w), np.inf, "f4")
                which = np.zeros((h, w), "u1")
                part = which.copy()
                dynamic = np.asarray(Image.open(masks / f"{frame:03}_{index}.png").convert("L"))
                for i, (a, ok) in enumerate(zip(anchors, possible), 1):
                    if not ok:
                        continue
                    shot = drv.drive.renderer.frame_crop(
                        mesh,
                        a,
                        model,
                        pose,
                        path,
                        drv.drive.occluding_cloud(raw, scan),
                        window,
                        ego,
                    )
                    if not shot:
                        continue
                    x, y = shot["xy"]
                    hh, ww = shot["part"].shape
                    sl = np.s_[y : y + hh, x : x + ww]
                    z = shot["range"]
                    take = (shot["part"] > 0) & (dynamic[sl] == 0) & (z < distance[sl])
                    after[sl][take] = shot["after"][take]
                    which[sl][take] = i
                    part[sl][take] = shot["part"][take]
                    distance[sl][take] = z[take]
                counts = [int(((which == i) & (part > 1)).sum()) for i in [1, 2]]
                if max(counts) < 12:
                    continue
                if (dynamic[which > 0] > 0).any() or np.any(im[which == 0] != after[which == 0]):
                    raise ValueError("Unprotected compositing")
                writer.send(
                    Image.fromarray(after).resize((960, 540), Image.Resampling.LANCZOS).tobytes()
                )
                row = dict(
                    frame=frame,
                    time_s=t,
                    scan=scan,
                    visible_defect_px=counts,
                    clearance_m=clear["lower_bound_m"],
                )
                frames.append(row)
                score = min(counts)
                if score > best:
                    best = score
                    row_best = row
                    Image.fromarray(after).save(dest / "both.jpg", quality=96)
                    Image.fromarray(im).save(dest / "original.jpg", quality=96)
                    y, x = np.where(which > 0)
                    sl = np.s_[y.min() : y.max() + 1, x.min() : x.max() + 1]
                    np.savez_compressed(
                        dest / "both.npz",
                        crop_xy=[x.min(), y.min()],
                        range_m=np.where(which[sl] > 0, distance[sl], np.nan),
                        instance=which[sl],
                        part=part[sl],
                    )
        if best < 12:
            raise ValueError(f"No joint exposure in {side}")
        result["cameras"][side] = dict(
            frames=frames,
            joint_frames=sum(min(r["visible_defect_px"]) >= 12 for r in frames),
            representative=row_best,
        )
    write(out / "report.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("raw", "masks", "models", "drive", "out"):
        p.add_argument(k, type=Path)
    replay(**vars(p.parse_args()))
