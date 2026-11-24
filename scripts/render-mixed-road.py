"""Replay photo-derived road hazards through the actual KITTI cameras with the CPU mesh renderer."""

import argparse
import json
from pathlib import Path
import sys
from contextlib import ExitStack

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "gen2hole/downstream"),
    str(ROOT / "streetcrafter/joint"),
    str(ROOT / "streetcrafter/backends"),
]
from mesh_transfer import RoadMesh, fit_road, pixel_rays, resolve_samples  # noqa: E402
from prepare import load_rig  # noqa: E402
from prepare_kitti import project_mei, homogeneous  # noqa: E402
from road_geometry import save_json, sha256  # noqa: E402


def view_model(sensor, data):
    calibration = data["source_calibration"][sensor]
    w, h = map(int, calibration["size"])
    if calibration["model"] == "MEI":
        spec = calibration["parameters"]
        p, d = spec["projection_parameters"], spec["distortion_parameters"]
        k = np.array([[p["gamma1"], 0, p["u0"]], [0, p["gamma2"], p["v0"]], [0, 0, 1.0]])
        coeffs = np.array([d[key] for key in ("k1", "k2", "p1", "p2")])
        xi = spec["mirror_parameters"]["xi"]

        def unproject(left, top, width, height, dx=0.0, dy=0.0):
            yy, xx = np.mgrid[top : top + height, left : left + width]
            uv = np.stack([xx + dx, yy + dy], -1)
            xy = cv2.undistortPointsIter(
                uv.reshape(-1, 1, 2),
                k,
                coeffs,
                None,
                None,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10),
            ).reshape(height, width, 2)
            r2 = (xy**2).sum(-1)
            discriminant = 1 + (1 - xi * xi) * r2
            valid = discriminant >= 0
            scale = (xi + np.sqrt(np.maximum(discriminant, 0))) / (r2 + 1)
            rays = np.dstack([xy * scale[..., None], scale - xi])
            valid &= np.linalg.norm(project_mei(rays, spec) - uv, axis=-1) < 0.05
            rays[~valid] = [0, 0, 1]
            return rays.astype("f4"), valid

        return (w, h), lambda xyz: project_mei(xyz, spec), unproject
    k = np.asarray(calibration["K"])
    if calibration["model"] == "Fisheye":
        d = np.ravel(calibration["D"])

        def project(xyz):
            xyz = np.asarray(xyz)
            if not xyz.size:
                return np.empty((*xyz.shape[:-1], 2))
            uv, _ = cv2.fisheye.projectPoints(
                np.ascontiguousarray(xyz.reshape(-1, 1, 3), dtype="f8"), np.zeros(3), np.zeros(3), k, d
            )
            return uv.reshape(*xyz.shape[:-1], 2)

        def unproject(left, top, width, height, dx=0.0, dy=0.0):
            cal = [
                "OPENCV_FISHEYE",
                width,
                height,
                *k.diagonal()[:2],
                k[0, 2] + 0.5 - left - dx,
                k[1, 2] + 0.5 - top - dy,
                *d,
            ]
            return pixel_rays(cal), np.ones((height, width), bool)

        return (w, h), project, unproject

    def project(xyz):
        return xyz[..., :2] / np.maximum(xyz[..., 2:], 1e-6) * k.diagonal()[:2] + k[:2, 2]

    def unproject(left, top, width, height, dx=0.0, dy=0.0):
        cal = ["PINHOLE", width, height, *k.diagonal()[:2], k[0, 2] + 0.5 - left - dx, k[1, 2] + 0.5 - top - dy]
        return pixel_rays(cal), np.ones((height, width), bool)

    return (w, h), project, unproject


def scans(raw, data, cameras):
    lines = (raw / "calibration/perspective.txt").read_text().splitlines()
    rect = homogeneous(
        np.column_stack(
            [
                np.fromstring(
                    next(s for s in lines if s.startswith("R_rect_00:")).split(":")[1], sep=" "
                ).reshape(3, 3),
                np.zeros(3),
            ]
        )
    )
    to_rect = rect @ np.linalg.inv(homogeneous(np.loadtxt(raw / "calibration/calib_cam_to_velo.txt")))
    result = []
    for g, frame in enumerate(data["frames"]):
        lidar = np.fromfile(raw / "velodyne" / f"{frame:010}.bin", "<f4").reshape(-1, 4)[:, :3]
        pose = np.linalg.inv(cameras[f"{g:03}00.png"]) @ to_rect
        result.append(np.float32(lidar @ pose[:3, :3].T + pose[:3, 3]))
    return result


def align_observed_road(anchor, pose, points):
    """Remove only locally measured roll, pitch and vertical drift of the A2D2 visual odometry."""
    center = np.asarray(anchor["center_world"])
    basis = np.asarray(anchor["basis_columns"])
    local = (points - center) @ basis
    # Fit an annulus outside the entire inserted footprint to avoid learning from the asset.
    radial = np.linalg.norm(local[:, :2], axis=1)
    sample = points[(radial > 1.0) & (radial < 3.3)]
    try:
        level = fit_road(sample, center, basis[:, 0], up_world=basis[:, 2], radius_m=3.3)
    except ValueError:
        return None
    a = np.asarray(level["basis_columns"])[:, 2]
    b = basis[:, 2]
    new_center = np.asarray(level["center_world"])
    angle = float(np.arccos(np.clip(a @ b, -1, 1)))
    offset = float(np.linalg.norm(center - new_center))
    if angle > np.deg2rad(6) or offset > 0.20 or level["rms_m"] > 0.035:
        return None
    axis = np.cross(a, b)
    rot = cv2.Rodrigues(axis * (angle / max(np.linalg.norm(axis), 1e-12)))[0]
    fixed = pose.copy()
    fixed[:3, :3] = rot @ pose[:3, :3]
    fixed[:3, 3] = rot @ (pose[:3, 3] - new_center) + center
    return (
        fixed,
        (points - new_center) @ rot.T + center,
        dict(tilt_deg=float(np.rad2deg(angle)), offset_m=offset, rms_m=level["rms_m"], cells=level["cells"]),
    )


def frame_crop(mesh, anchor, sensor_model, pose, raw_image, points, is_fish, ego, packet_path=None):
    (width, height), project, unproject = sensor_model
    center, basis = np.asarray(anchor["center_world"]), np.asarray(anchor["basis_columns"])

    def to_cam(pts):
        return (pts - pose[:3, 3]) @ pose[:3, :3]

    # Enclose all possible intersections: surface.npz spans the full square road patch.
    ex = mesh.extent * 0.5
    box = np.array([[x, y, z] for x in [-ex[0], ex[0]] for y in [-ex[1], ex[1]] for z in [-0.03, 0.16]])
    local = to_cam(box @ basis.T + center)
    if local[:, 2].min() < (0.04 if is_fish else 0.35):
        return None
    bounds = project(local)
    if np.any(bounds.min(0) < 3) or np.any(bounds.max(0) > np.array([width, height]) - 4):
        return None
    lo = np.maximum(0, np.floor(bounds.min(0) - 18)).astype(int)
    hi = np.minimum([width, height], np.ceil(bounds.max(0) + 19)).astype(int)
    if np.any(hi - lo < 8) or not np.all(
        (project(to_cam(center)) >= 0) & (project(to_cam(center)) < [width, height])
    ):
        return None
    x, y = lo
    w, h = hi - lo
    image = np.array(Image.open(raw_image).convert("RGB").crop((*lo, *hi)))
    grid = np.asarray(ego["grid"])
    edge = np.asarray(ego["edge"]) * np.array([width, height]) / grid
    body = (y + np.arange(h)[:, None]) >= np.interp(x + np.arange(w), edge[:, 0], edge[:, 1])
    depth = np.full((h, w), np.inf, "f4")
    # The host's original pavement will be replaced. Only non-ground LiDAR points can occlude it.
    above = (points - center) @ basis[:, 2] > 0.10
    cam = to_cam(points[above])
    cam = cam[(cam[:, 2] > 0) & (np.linalg.norm(cam, axis=1) < 70)]
    uv = project(cam) - lo
    keep = (uv >= 0).all(1) & (uv < [w - 0.5, h - 0.5]).all(1)
    uv = np.rint(uv[keep]).astype(int)
    np.minimum.at(depth, (uv[:, 1], uv[:, 0]), np.linalg.norm(cam[keep], axis=1).astype("f4"))
    depth = cv2.erode(depth, np.ones((7, 7), "u1"))
    samples, packet_rays, packet_masks = [], [], []
    for dx, dy in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]:
        rays, valid = unproject(x, y, w, h, dx, dy)
        protected = body | ~valid
        samples.append(mesh.render(image, pose, rays, anchor, depth, protected))
        if packet_path:
            packet_rays.append(rays)
            packet_masks.append(protected)
    after, part, distance = resolve_samples(samples)
    if packet_path:
        np.savez_compressed(
            packet_path,
            rgb=image,
            camera_to_world=pose,
            host_range=depth,
            rays_camera=packet_rays,
            protected=packet_masks,
        )
    hit = part > 0
    if hit.sum() < 12:
        return None
    mod = np.any(after != image, axis=2)
    if mod[~hit].any() or mod[body].any():
        raise ValueError("A protected or out-of-support pixel changed")
    return dict(
        before=image,
        after=after,
        part=part,
        range=distance,
        xy=lo,
        visible=int(hit.sum()),
        changed=int(mod.sum()),
        protected_ego=int(body.sum()),
    )


def audi_cloud(path, pose):
    with np.load(path) as data:
        forward, left, up = data["pcloud_points"][data["pcloud_attr.valid"]].T
    cam = np.column_stack([-left, -up, forward])
    return np.float32(cam @ pose[:3, :3].T + pose[:3, 3])


def audi_inputs(raw, data, cams, group, offset):
    index = json.loads((raw / "rig_frames.json").read_text())

    def filename(g, sensor):
        parent = raw if sensor == "front_left" else raw / sensor
        return parent / f"{index[sensor][g]:09}.png"

    pose = np.linalg.inv(cams[f"{group:03}00.png"])
    body = pose @ np.linalg.inv(np.asarray(data["source_calibration"]["front_left"]["optical_to_body"]))
    up = body[:3, 2]
    seed = body[:3, 3] - offset * body[:3, 1] - 0.76 * up
    clouds = {}
    views = [(0, range(max(0, group - 27), group + 1)), (5, range(group, min(len(data["frames"]), group + 32)))]
    for cam, groups in views:
        sensor = data["sensors"][cam]
        for g in groups:
            pose = np.linalg.inv(cams[f"{g:03}{cam}0.png"])
            clouds[g, cam] = audi_cloud(filename(g, sensor).with_suffix(".npz"), pose)
    nearby = np.concatenate([c for (g, cam), c in clouds.items() if g % 3 == 0])
    anchor = fit_road(nearby, seed, body[:3, 0], up_world=up, radius_m=2.1)
    return anchor, views, lambda g, cam: clouds[g, cam], filename


def video_segment(writer, candidates, best, mesh_id, label, scene, sensor):
    from PIL import ImageDraw, ImageFont, ImageOps

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    header = ImageFont.truetype(font_path, 26, index=2)
    caption = ImageFont.truetype(font_path, 19, index=2)
    seq = sorted(candidates, key=lambda item: item[2]["group"])
    i = next(i for i, (_, _, r, _) in enumerate(seq) if r is best)
    start = end = i
    while start and seq[start - 1][2]["group"] + 1 == seq[start][2]["group"]:
        start -= 1
    while end + 1 < len(seq) and seq[end][2]["group"] + 1 == seq[end + 1][2]["group"]:
        end += 1
    clip = seq[start : end + 1]
    times = np.asarray([r["time"] for _, _, r, _ in clip])
    for t in np.arange(times[0], times[-1] + 0.08, 0.1):
        _, frame, r, photo = clip[int(np.argmin(abs(times - t)))]
        host = Image.open(photo).convert("RGB")
        host.paste(Image.fromarray(r["after"]), tuple(r["xy"]))
        page = Image.new("RGB", (960, 480), "#20272a")
        d = ImageDraw.Draw(page)
        camera = {
            "image_00": "前视针孔",
            "image_03": "右侧原始鱼眼",
            "front_left": "左前原始鱼眼",
            "rear_center": "后方原始鱼眼",
        }[sensor]
        d.text((16, 7), mesh_id + "  " + label, font=header, fill="white")
        d.text((18, 55), scene + "  ·  " + camera + "  ·  " + str(frame), font=caption, fill="#d4dedb")
        wide = ImageOps.contain(host, (560, 365), Image.Resampling.LANCZOS)
        page.paste(wide, (16 + (560 - wide.width) // 2, 86 + (365 - wide.height) // 2))
        for y, name, key in [(52, "原路面", "before"), (256, "同机位迁入后", "after")]:
            d.text((610, y), name, font=caption, fill="#d4dedb")
            detail = ImageOps.contain(Image.fromarray(r[key]), (330, 167), Image.Resampling.LANCZOS)
            page.paste(detail, (600 + (340 - detail.width) // 2, y + 32 + (167 - detail.height) // 2))
        writer.append_data(np.asarray(page))
    for _ in range(5):
        writer.append_data(np.asarray(page))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mixed-road")
    parser.add_argument("--scene", choices=["kitti360-0000", "a2d2-frontleft"], default="kitti360-0000")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--group", type=int, default=26, help="host trajectory index used to seed the anchor")
    parser.add_argument("--offset", type=float, default=2.25, help="metres toward camera right")
    parser.add_argument(
        "--ids", nargs="+", default=["P220", "P221", "P222", "P223", "A224", "A225", "A226", "A227", "M224"]
    )
    parser.add_argument(
        "--video", action="store_true", help="Encode consecutive, registered physical-camera frames"
    )
    parser.add_argument(
        "--export-packet", help="Also save the best cropped rays, ranges, foreground and RGB for this ID"
    )
    args = parser.parse_args()
    rig = ROOT / "work/streetcrafter-rig" / args.scene
    data, cams = load_rig(rig)
    if args.scene == "a2d2-frontleft":
        raw = ROOT / "work/data" / args.scene
        anchor, views, cloud_at, filename = audi_inputs(raw, data, cams, args.group, args.offset)
    else:
        raw = ROOT / "work/data/kitti360-0000-0200"
        clouds = scans(raw, data, cams)
        rig_pose = np.linalg.inv(cams[f"{args.group:03}00.png"])
        seed = rig_pose[:3, 3] + args.offset * rig_pose[:3, 0] - [0, 0, 1.6]
        nearby = np.concatenate(clouds[max(0, args.group - 7) : args.group + 8 : 2])
        anchor = fit_road(nearby, seed, rig_pose[:3, 2], up_world=[0, 0, 1], radius_m=2.1)
        views = [
            (0, range(max(0, args.group - 18), args.group)),
            (3, range(max(0, args.group - 6), min(len(clouds), args.group + 8))),
        ]

        def cloud_at(g, cam):
            return clouds[g]

        def filename(g, sensor):
            return raw / sensor / f"{data['frames'][g]:010}.png"

    out = args.out or ROOT / "outputs/mixed-road" / args.scene
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "anchor.json", anchor)
    ego_masks = json.loads((ROOT / "gen2hole/examples/mixed-road/ego-masks.json").read_text())[args.scene]
    index = {}
    with ExitStack() as stack:
        writer = None
        if args.video:
            import imageio.v2 as imageio

            writer = stack.enter_context(
                imageio.get_writer(
                    str(out / "driving.tmp.mp4"),
                    fps=10,
                    codec="libx264",
                    quality=7,
                    macro_block_size=2,
                    output_params=["-threads", "3", "-movflags", "+faststart"],
                )
            )
        for key in args.ids:
            (folder,) = args.run.glob(f"*/series/assets/*/{key}")
            mesh = RoadMesh(folder)
            index[key] = {"source_sha256": mesh.digest, "views": {}}
            for cam, groups in views:
                sensor = data["sensors"][cam]
                model = view_model(sensor, data)
                candidates = []
                for g in groups:
                    name = f"{g:03}{cam}0.png"
                    pose = np.linalg.inv(cams[name])
                    frame = data["views"][name]["source_frame"]
                    photo = filename(g, sensor)
                    points = cloud_at(g, cam)
                    correction = None
                    if args.scene == "a2d2-frontleft":
                        adjusted = align_observed_road(anchor, pose, points)
                        if adjusted is None:
                            continue
                        pose, points, correction = adjusted
                    res = frame_crop(
                        mesh,
                        anchor,
                        model,
                        pose,
                        photo,
                        points,
                        data["source_calibration"][sensor]["model"] != "PINHOLE rectified",
                        ego_masks[sensor],
                    )
                    if res:
                        res.update(alignment=correction, group=g, time=data["times"][name])
                        candidates.append((res["visible"], frame, res, photo))
                if not candidates:
                    raise ValueError(f"{key}: no visible support for {sensor}")
                _, f, best, photo = max(candidates, key=lambda c: c[0])
                dest = out / key / sensor
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(best["before"]).save(dest / "before.png")
                Image.fromarray(best["after"]).save(dest / "after.png")
                Image.fromarray(best["part"]).save(dest / "part.png")
                np.save(dest / "range.npy", best["range"])
                image = Image.open(photo).convert("RGB")
                image.paste(Image.fromarray(best["after"]), tuple(best["xy"]))
                image.save(dest / "full.jpg", quality=94)
                if writer:
                    label = json.loads((folder / "metadata.json").read_text())["label"]
                    video_segment(writer, candidates, best, key, label, args.scene, sensor)
                if args.export_packet == key:
                    g = best["group"]
                    pose = np.linalg.inv(cams[f"{g:03}{cam}0.png"])
                    points = cloud_at(g, cam)
                    if args.scene == "a2d2-frontleft":
                        pose, points, _ = align_observed_road(anchor, pose, points)
                    frame_crop(
                        mesh,
                        anchor,
                        model,
                        pose,
                        photo,
                        points,
                        data["source_calibration"][sensor]["model"] != "PINHOLE rectified",
                        ego_masks[sensor],
                        dest / "view.npz",
                    )
                row = dict(
                    frame=f,
                    local_road_alignment=best["alignment"],
                    crop_xy=best["xy"].tolist(),
                    visible=best["visible"],
                    changed=best["changed"],
                    protected_ego_pixels=best["protected_ego"],
                    protected_edits=0,
                    source_rgb_sha256=sha256(photo),
                    captured_frames=[(frame, n) for n, frame, _, _ in candidates],
                    model=data["source_calibration"][sensor]["model"],
                )
                index[key]["views"][sensor] = row
                print(key, sensor, f, best["visible"], flush=True)
            save_json(out / "index.json", index)
    if args.video:
        (out / "driving.tmp.mp4").replace(out / "driving.mp4")


if __name__ == "__main__":
    main()
