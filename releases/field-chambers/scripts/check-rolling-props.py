"""Check rolling contact and camera ranges, and export subpixel tracks between consecutive exposures."""

import argparse
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from traffic_safe_drive import digest, pandaset, read, write

drive = import_module("render-pandaset-spill")
from road_geometry import Surface, glb_triangles

OFFSETS = np.array([[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]])


def tracked_surface(mesh, transform, pose, lens, offset, parts, ranges, *, camera_ray_dtype=None):
    row, col = np.nonzero(parts)
    uv = np.stack([col + offset[0], row + offset[1]], axis=1)[:, None, :] + OFFSETS[None, :, :]
    xy = (uv - [lens["cx"], lens["cy"]]) / [lens["fx"], lens["fy"]]
    optical = np.concatenate([xy, np.ones((*xy.shape[:2], 1))], axis=2)
    optical /= np.linalg.norm(optical, axis=2, keepdims=True)
    if camera_ray_dtype is not None:
        # Match the precision at which the camera supplied its optical-frame bearings.
        optical = optical.astype(camera_ray_dtype)
    rays = optical @ (pose[:3, :3].T @ transform[:3, :3])
    origin = (pose[:3, 3] - transform[:3, 3]) @ transform[:3, :3]
    d, labels, _ = mesh.cast(np.broadcast_to(origin, rays.shape), rays)
    # A silhouette pixel can come from any one of the four sampled subpixel rays.
    error = np.where(labels == parts[row, col, None], np.abs(d - ranges[row, col, None]), np.inf)
    best = error.argmin(axis=1)
    ix = np.arange(len(row))
    residual = error[ix, best]
    if not np.isfinite(residual).all() or residual.max() > 0.0015:
        raise ValueError(f"Original mesh does not reproduce exported subpixel depth ({residual.max()} m)")
    xyz = origin + rays[ix, best] * d[ix, best, None]
    return dict(row=row, col=col, uv=uv[ix, best], xyz=xyz, error_m=float(residual.max()))


def flow(source, target, mesh, lens):
    matrix, pose = target["matrix"], target["pose"]
    xyz = source["xyz"]
    world = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    cam = (world - pose[:3, 3]) @ pose[:3, :3]
    uv = cam[:, :2] / cam[:, 2, None] * [lens["fx"], lens["fy"]] + [lens["cx"], lens["cy"]]
    origin = (pose[:3, 3] - matrix[:3, 3]) @ matrix[:3, :3]
    distance = np.linalg.norm(xyz - origin, axis=1)
    first = mesh.cast(np.broadcast_to(origin, xyz.shape), (xyz - origin) / distance[:, None])[0]
    # Continuous-ray self-visibility plus the observed target's foreground mask at the nearest pixel.
    pixel = np.rint(uv - target["offset"]).astype(int)
    part = target["parts"]
    inside = (pixel >= 0).all(axis=1) & (pixel < part.shape[::-1]).all(axis=1) & (cam[:, 2] > 0)
    visible = (abs(first - distance) < 0.0002) & inside
    visible[inside] &= part[pixel[inside, 1], pixel[inside, 0]] > 0
    return dict(
        source_uv=source["uv"],
        target_uv=uv,
        flow_px=np.float32(uv - source["uv"]),
        source_crop_rc=np.int32(np.stack([source["row"], source["col"]], axis=1)),
        asset_xyz_m=np.float32(xyz),
        target_visible=visible,
    )


def inspect(args, dest):
    survey, times, poses, lens = pandaset(args.raw)
    summary = read(args.motion / "summary.json")
    route = read(args.route)
    if set(summary) != set(route["mean_speed_mps"]) or route["scene"] != args.raw.name:
        raise ValueError("Model inventory or route differs")
    result = {}
    for key, s in summary.items():
        folder = args.motion / key
        geom = args.models / key / "asset.glb"
        rec = read(folder / "motion.json")
        movie = folder / "drive.mp4"
        rows = rec["frames"]
        if (
            rec["mesh_sha256"] != digest(geom)
            or digest(movie) != s["movie_sha256"]
            or imageio_ffmpeg.count_frames_and_secs(str(movie))[0] != len(rows)
            or [f["frame"] for f in rows] != list(range(route["first_frame"], route["last_frame"] + 1))
        ):
            raise ValueError(f"{key}: source, exposure sequence or video changed")
        road = np.array(rec["road"]["basis_columns"])
        track = np.array(rec["track_start_world"])
        mesh = Surface(geom)
        verts = np.unique(np.concatenate([xyz for _, xyz, _ in glb_triangles(geom, names=True)]), axis=0)
        section = verts[:, [0, 2]]
        edges = ConvexHull(section)
        section = section[edges.vertices].astype(float)
        if abs(edges.area - rec["perimeter_m"]) > 1e-6:
            raise ValueError("Independent convex perimeter mismatch")
        clear = survey.require_window(
            {key: rec["world_xy_envelope_m"]}, times[rows[0]["frame"]], times[rows[-1]["frame"]], 0.25
        )
        if clear != rec["traffic"]:
            raise ValueError("Traffic survey differs")
        (dest / key).mkdir()
        previous = None
        ground_error = slip_error = range_error = 0.0
        flow_visible = flow_total = 0
        for f in rows:
            i = f["frame"]
            matrix = np.array(f["asset_to_world"])
            angle = f["angle_rad"]
            pose = drive.pose_matrix(poses[i])
            camera = args.raw / "camera/front_camera" / f"{i:02}.jpg"
            if f["time_s"] != times[i] or digest(camera) != f["photo_sha256"]:
                raise ValueError("Source photo or timestamp changed")
            expected = (
                (times[i] - times[rows[0]["frame"]]) * 2 * np.pi * route["mean_speed_mps"][key] / edges.area
            )
            c, sine = np.cos(expected), np.sin(expected)
            if abs(angle - expected) > 1e-5 or not np.allclose(
                matrix[:3, :3], road @ [[c, 0, sine], [0, 1, 0], [-sine, 0, c]], atol=1e-6
            ):
                raise ValueError("Angle or axis mismatch")
            world = verts @ matrix[:3, :3].T + matrix[:3, 3]
            bound = np.array(rec["world_xy_envelope_m"])
            if np.any(world[:, :2] < bound[0]) or np.any(world[:, :2] > bound[1]):
                raise ValueError("Actual mesh outside audited traffic envelope")
            ground_error = max(ground_error, abs(((world - track) @ road[:, 2]).min()))
            origin = (matrix[:3, 3] - track) @ road
            if previous:
                # For zero tangential contact speed, dx/dθ = -min_z(Rθ p).
                a = np.linspace(previous["angle"], angle, 257)
                height = np.cos(a)[:, None] * section[:, 1] - np.sin(a)[:, None] * section[:, 0]
                dx = np.trapz(-height.min(axis=1), a)
                slip_error = max(slip_error, abs(origin[0] - previous["origin"][0] - dx))
            with np.load(folder / f"{i:02}.npz", allow_pickle=False) as packet:
                parts, ranges, shadow = packet["part"], packet["range_m"], packet["shadow"]
            xy = np.array(f["crop_xy"])
            h, w = parts.shape
            box = (*xy, *(xy + [w, h]))
            src = np.asarray(Image.open(camera).convert("RGB").crop(box))
            protected = np.asarray(Image.open(args.masks / f"{i:03}_0.png").convert("L").crop(box)) != 0
            after = np.asarray(Image.open(folder / f"{i:02}.png"))
            changed = (src != after).any(axis=2)
            hit = parts > 0
            if (
                not np.array_equal(hit, np.isfinite(ranges))
                or hit.sum() != f["visible_px"]
                or np.any(changed & (protected | (~hit & (shadow == 0))))
                or not np.any(changed & hit)
            ):
                raise ValueError("Visible mask, raw exposure or shadow differs")
            points = tracked_surface(mesh, matrix, pose, lens, xy, parts, ranges)
            range_error = max(range_error, points["error_m"])
            current = dict(
                parts=parts, offset=xy, pose=pose, matrix=matrix, angle=angle, origin=origin, **points
            )
            if previous:
                tracks = flow(previous, current, mesh, lens)
                flow_total += len(tracks["flow_px"])
                flow_visible += int(tracks["target_visible"].sum())
                np.savez_compressed(dest / key / f"{i - 1:02}-{i:02}.npz", **tracks)
            previous = current
        if ground_error > 0.00005 or slip_error > 0.00005 or not flow_visible:
            raise ValueError(f"{key}: contact error {ground_error} or sliding {slip_error}")
        result[key] = dict(
            frames=len(rows),
            travel_m=rows[-1]["travel_m"],
            rotation_turns=rows[-1]["angle_rad"] / (2 * np.pi),
            max_ground_error_m=ground_error,
            max_increment_slip_m=slip_error,
            max_camera_range_error_m=range_error,
            source_flow_points=flow_total,
            next_visible_points=flow_visible,
            traffic_min_m=clear["minimum_bound_m"],
        )
        print(key, result[key], flush=True)
    write(dest / "report.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "raw", "masks", "route", "motion", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("Choose a fresh output directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as temp:
        stage = Path(temp) / "flow"
        stage.mkdir()
        inspect(args, stage)
        stage.rename(args.out)
