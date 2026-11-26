"""Render metric pavement defects in annotated PandaSet exposures, preserving the surrounding road."""

import argparse
from contextlib import closing
from subprocess import check_output

import imageio_ffmpeg
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

drive = import_module("render-pandaset-spill")
from mesh_transfer import RoadMesh, fit_road
from camera_geometry import CAMERAS
from traffic_clearance import world_bounds
from traffic_safe_drive import read, write


def pinhole(camera, intrinsics, shot):
    with Image.open(camera / f"{shot:02}.jpg") as photo:
        width, height = photo.size
    k = [
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ]
    return drive.renderer.view_model(
        camera.name,
        {
            "source_calibration": {
                camera.name: dict(size=[width, height], model="PINHOLE", K=k)
            }
        },
    )


def render(raw, masks, models, out, shot, pixel, keys, camera_name="front_camera", edge_blend_m=0., curb=None, yaw_deg=0., stripe=None, semantic_occlusion=False, pool_layout=None, relief_daylight=None, driveway=None, focus_part=None, allow_hidden_focus=False, focus_group=None):
    survey, times, poses, intrinsics = drive.pandaset(raw)
    front = raw / "camera/front_camera"
    model = pinhole(front, intrinsics, shot)
    if driveway is not None:
        if curb is not None or stripe is not None or pool_layout is not None:
            raise ValueError("A fixed driveway survey supplies its own placement")
        from driveway_installation import anchor_from_driveway
        anchor = anchor_from_driveway(raw, drive, driveway)
    elif stripe is not None:
        if len(keys)!=1 or curb is not None or yaw_deg:
            raise ValueError("A stripe registration uses one marked model and its fitted orientation")
        from stripe_installation import anchor_from_stripe
        (meta_path,) = models.glob(f"*/series/assets/*/{keys[0]}/metadata.json")
        spec=read(meta_path)
        anchor=anchor_from_stripe(raw,shot,model,poses,drive,stripe,spec["local_ground_stripe_centers"],spec.get("local_paint_tip_xy"))
    elif curb is None:
        seed, points, heading = drive.host_patch(raw, shot, model[1], poses, pixel)
        anchor = fit_road(points, seed, heading, up_world=[0, 0, 1], radius_m=1.05)
    else:
        from curb_installation import anchor_from_curb
        anchor = anchor_from_curb(raw, shot, model, poses, drive, curb)
    if relief_daylight is not None:
        if relief_daylight["scene"] != raw.name or not 0 <= relief_daylight["direct_fraction"] < 1:
            raise ValueError("Relief lighting must match the source driving sequence")
        anchor["relief_daylight"] = relief_daylight
    if yaw_deg:
        theta = np.deg2rad(yaw_deg)
        c, s = np.cos(theta), np.sin(theta)
        local = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        anchor["basis_columns"] = (np.array(anchor["basis_columns"]) @ local).tolist()
        anchor["yaw_offset_deg"] = yaw_deg
    camera = raw / "camera" / camera_name
    rear = camera_name == "back_camera"
    if camera_name != "front_camera":
        times, poses, intrinsics = [
            read(camera / name)
            for name in ("timestamps.json", "poses.json", "intrinsics.json")
        ]
        model = pinhole(camera, intrinsics, shot)
    (width, height), project, _ = model
    cutoff = round(height * 0.73) if rear else height
    ego = dict(grid=[width, height], edge=[[0, cutoff], [width, cutoff]])
    frames = (
        range(shot, min(len(times), shot + 45))
        if rear
        else range(max(0, shot - 4), min(len(times), shot + 4))
    )
    if driveway is not None:
        lo,hi = driveway["camera_frame_ranges"][camera_name]
        frames = range(max(0,lo),min(len(times),hi))
    elif camera_name not in ("front_camera","back_camera"):
        frames = range(max(0,shot-5),min(len(times),shot+28))
    camera_index = CAMERAS.index(camera_name)
    window = drive.renderer.RayWindow(model, reflection=pool_layout is not None, metric_reflections=pool_layout is not None)
    scan_times = np.array(read(raw / "lidar/timestamps.json"))
    font = ImageFont.truetype(
        check_output(["fc-match", "-f", "%{file}", "sans:lang=zh"], text=True), 19
    )
    for key in keys:
        (folder,) = models.glob(f"*/series/assets/*/{key}")
        if pool_layout is not None:
            if len(keys) != 1 or curb or stripe or edge_blend_m or yaw_deg:
                raise ValueError("A surveyed waterline supplies its own grade, orientation and blending")
            from waterline_installation import install
            renderer_mesh, mesh, anchor = install(pool_layout, models, key, anchor)
        else:
            mesh = RoadMesh(folder, filter_pavement=True, measured_rim=True, edge_blend_m=edge_blend_m)
            renderer_mesh = mesh
        if relief_daylight is not None and not (mesh.registered_relief or mesh.continuous_lane):
            raise ValueError("Relief daylight requires an albedo scan with registered heights")
        if stripe is not None and stripe.get("match_host_paint_width",False):
            from stripe_installation import match_marking_width
            match_marking_width(mesh, read(folder/"metadata.json"), anchor)
        if curb is not None:
            # These surveyed support surfaces receive the original sidewalk and gutter pixels.
            for gid, (part, tri, uv, texture) in mesh.surfaces.items():
                if part in curb["preserve_host_parts"]:
                    mesh.surfaces[gid] = (1, tri, uv, texture)
        track = survey.evaluate({key: world_bounds(anchor, mesh.bounds)})
        dest = out / key
        dest.mkdir(parents=True, exist_ok=True)
        records, best, video_frames = [], None, []
        label = read(folder / "metadata.json")["label"]
        for frame in frames:
            clear = survey.exposure(track, times[frame])
            if not clear["clear"]:
                continue
            scan = int(np.argmin(abs(scan_times - times[frame])))
            path = camera / f"{frame:02}.jpg"
            if driveway and "receiver_photo_sha256" in driveway:
                from traffic_safe_drive import digest
                if digest(path) != driveway["receiver_photo_sha256"][camera_name][str(frame)]:
                    raise ValueError("Driveway receiver exposure differs from reviewed original")
            pose = drive.pose_matrix(poses[frame])
            optical = (np.asarray(anchor["center_world"]) - pose[:3, 3]) @ pose[:3, :3]
            if camera_name != "front_camera" and (
                optical[2] <= 0
                or not (
                    (project(optical) >= 0) & (project(optical) < [width, cutoff])
                ).all()
            ):
                continue
            replaced_support = None
            if curb and "receiver_extent_m" in curb:
                from curb_installation import registered_curb_cloud
                points,replaced_support = registered_curb_cloud(raw,scan,drive,anchor,semantic_occlusion)
            else:
                points = drive.occluding_cloud(raw, scan) if semantic_occlusion else drive.cloud(raw, scan)
            result = drive.renderer.frame_crop(renderer_mesh, anchor, model, pose, path, points, window, ego)
            if not result or result["clipped_sides"] or result["touches_mask"]:
                continue
            x, y = result["xy"].tolist()
            h, w = result["part"].shape
            with Image.open(masks / f"{frame:03}_{camera_index}.png") as dynamic:
                assert dynamic.size == (width, height)
                traffic = np.array(dynamic.convert("L").crop((x, y, x + w, y + h)))
            if (traffic[result["part"] > 0] != 0).any():
                continue
            count = int((result["part"] > 1).sum())
            changed = np.any(result["before"] != result["after"], axis=2)
            if count < 16 or not changed.any():
                continue
            row = dict(
                frame=frame,
                scan=scan,
                time_s=times[frame],
                visible_px=count,
                changed_px=int(changed.sum()),
                clearance_m=clear["lower_bound_m"],
                crop_xy=[x, y],
            )
            if replaced_support is not None:
                row["replaced_curb_returns"] = replaced_support
            if pool_layout is not None:
                row["reflections"] = result["reflection_stats"]
            if focus_part is not None:
                row["focus_visible_px"] = int((result["part"] == focus_part).sum())
                row["focus_changed_px"] = int(((result["part"] == focus_part) & changed).sum())
            if focus_group:
                row["group_changed_px"] = [int(((result["part"] == part) & changed).sum()) for part in focus_group]
            records.append(row)
            with Image.open(path) as original:
                tile = drive.comparison(original, result, key, label, frame, font)
                video_frames.append(tile.tobytes())
            if focus_part is None:
                score = (count,)
            else:
                score = (row["focus_changed_px"], row["focus_visible_px"], count)
            if focus_group:
                score = (min(row["group_changed_px"]), sum(row["group_changed_px"]), *score)
            if best is None or score > best[3]:
                best = row, result, path, score
        if best is None:
            raise ValueError(f"No complete visible exposure for {key}")
        if focus_part is not None and not allow_hidden_focus and not best[0]["focus_changed_px"]:
            raise ValueError(f"No visible change on requested part {focus_part} of {key}")
        if focus_group and not allow_hidden_focus and min(best[0]["group_changed_px"]) == 0:
            raise ValueError(f"No exposure resolved every requested part of {key}")
        with closing(
            imageio_ffmpeg.write_frames(
                str(dest / "drive.mp4"),
                (960, 752),
                fps=10,
                codec="libx264",
                output_params=["-crf", "17", "-threads", "2"],
            )
        ) as writer:
            writer.send(None)
            for tile in video_frames:
                writer.send(tile)
        row, result, path, _ = best
        after = Image.fromarray(result["after"])
        with Image.open(path) as photo:
            full = photo.convert("RGB")
        full.paste(after, tuple(row["crop_xy"]))
        full.save(dest / "drive.jpg", quality=95)
        for name in ("before", "after", "part"):
            Image.fromarray(result[name]).save(dest / (name + ".png"))
        np.savez_compressed(
            dest / "rays.npz", range_m=result["range"], part=result["part"]
        )
        write(
            dest / "placement.json",
            dict(
                **({"edge_blend_m": edge_blend_m} if edge_blend_m else {}),
                asset=key,
                scene=raw.name,
                camera=camera_name,
                anchor=anchor,
                **({"occlusion_source":"labeled non-reflection LiDAR"} if semantic_occlusion else {}),
                glb_sha256=mesh.digest,
                best_frame=row["frame"],
                **({"focus_part": focus_part} if focus_part is not None else {}),
                **({"focus_group": focus_group} if focus_group else {}),
                exposures=records,
            ),
        )
        print(key, len(records), row["visible_px"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "masks", "models", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--relief-daylight", type=Path, help="Use measured host sunlight on a registered pavement scan")
    p.add_argument("--driveway", "--support-survey", type=Path, help="Fit a labelled support surface from a recorded laser point")
    p.add_argument("--pool-layout", type=Path, help="Seat a registered water interface on the measured host grade")
    p.add_argument("--semantic-occlusion",action="store_true",help="Exclude PandaSet returns labeled Reflection from opaque image occluders")
    p.add_argument("--stripe-survey", type=Path, help="Align the asset stripe to measured paint on the host road")
    p.add_argument("--curb-survey", type=Path, help="Register against marked curb pixels and a two-sided laser profile")
    p.add_argument("--yaw-deg", type=float, default=0., help="Rotate the model about its local road normal")
    p.add_argument("--allow-hidden-focus", action="store_true", help="Retain recorded exposures when the selected part is hidden throughout the pass")
    p.add_argument("--focus-group", type=int, nargs="+", help="Prefer exposures with changed pixels on every selected part")
    p.add_argument("--focus-part", type=int, help="Select the still exposure showing the most changed pixels of this mesh part")
    p.add_argument("--shot", type=int, default=15)
    p.add_argument("--pixel", nargs=2, type=int, default=[1120, 850])
    p.add_argument(
        "--camera", choices=CAMERAS, default="front_camera"
    )
    p.add_argument("--edge-blend-m", type=float, default=0., help="Colour transition inside a surveyed pavement outline, in metres")
    p.add_argument("ids", nargs="+")
    a = p.parse_args()
    render(a.raw, a.masks, a.models, a.out, a.shot, a.pixel, a.ids, a.camera, a.edge_blend_m, read(a.curb_survey) if a.curb_survey else None,a.yaw_deg,read(a.stripe_survey) if a.stripe_survey else None,a.semantic_occlusion,a.pool_layout,read(a.relief_daylight) if a.relief_daylight else None,read(a.driveway) if a.driveway else None,a.focus_part,a.allow_hidden_focus,a.focus_group)
