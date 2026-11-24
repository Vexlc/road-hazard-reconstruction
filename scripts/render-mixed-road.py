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
from mesh_transfer import RoadMesh, fit_footprint, fit_road, footprint_size, pixel_rays, resolve_samples  # noqa: E402
from prepare import load_rig  # noqa: E402
from prepare_kitti import project_mei, homogeneous  # noqa: E402
from road_geometry import glb_triangles, save_json, sha256  # noqa: E402
from pavement_layout import PavementLayout, resolve_instances
from prop_lighting import fit_receivers, view_bounds
from prop_registration import driving_clearance, registered_geometry, stereo_ranges

SUBPIXELS = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))


class RayWindow:
    """Bound every calibrated subpixel ray in each tile, including fish-eye rays behind the optical plane."""

    def __init__(self, model):
        (self.width, self.height), _, unproject = model
        self.tile = 32
        starts = np.arange(0, self.width, self.tile)
        low, high = [], []
        for y in range(0, self.height, self.tile):
            h = min(self.tile, self.height - y)
            lo = np.full((len(starts), 3), np.inf, 'f4')
            hi = -lo.copy()
            for dx, dy in SUBPIXELS:
                rays, valid = unproject(0, y, self.width, h, dx, dy)
                lo = np.minimum(lo, np.minimum.reduceat(np.where(valid[..., None], rays, np.inf).min(0), starts))
                hi = np.maximum(hi, np.maximum.reduceat(np.where(valid[..., None], rays, -np.inf).max(0), starts))
            low.append(lo)
            high.append(hi)
        self.low, self.high = np.stack(low), np.stack(high)
        self.valid = np.isfinite(self.low).all(-1)
        self.low[~self.valid] = self.high[~self.valid] = 0

    def crop(self, center, radius):
        distance = np.linalg.norm(center)
        if distance <= radius:
            keep = self.valid
        else:
            axis = center / distance
            # A tile is excluded only when even its directional upper bound misses the enclosing sphere.
            upper = (np.where(axis >= 0, self.high, self.low) * axis).sum(-1)
            keep = self.valid & (upper >= np.sqrt(1 - (radius / distance)**2) - 1e-6)
        yy, xx = np.nonzero(keep)
        if not len(xx):
            return None
        lo = np.array([xx.min(), yy.min()]) * self.tile
        hi = np.minimum((np.array([xx.max(), yy.max()]) + 1) * self.tile, [self.width, self.height])
        return lo, hi


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
            radial = np.linalg.norm(xyz[..., :2], axis=-1)
            theta = np.arctan2(radial, xyz[..., 2])
            r = theta * np.polynomial.polynomial.polyval(theta**2, np.r_[1., d])
            scale = np.divide(r, radial, out=np.zeros_like(r), where=radial > 0)
            return xyz[..., :2] * scale[..., None] * k.diagonal()[:2] + k[:2, 2]

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
            return pixel_rays(cal, return_valid=True)

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


def frame_crop(
    mesh, anchor, sensor_model, pose, raw_image, points, window, ego, packet_path=None, sensor_name=None
):
    from pavement_appearance import camera_anchor

    anchor = camera_anchor(anchor, sensor_name)
    (width, height), project, unproject = sensor_model
    center, basis = np.asarray(anchor["center_world"]), np.asarray(anchor["basis_columns"])

    def to_cam(pts):
        return (pts - pose[:3, 3]) @ pose[:3, :3]

    # Buried geometry can only be seen through its road-plane opening. Include every
    # above-ground vertex and the entire aperture/apron, independent of forward projection.
    grouped = isinstance(mesh, PavementLayout)
    bounds = mesh.visible_bounds(anchor) if grouped else view_bounds(mesh, anchor)
    mid = (bounds[0] + bounds[1]) / 2
    crop = window.crop(to_cam(mid @ basis.T + center), np.linalg.norm(bounds[1] - mid))
    if crop is None:
        return None
    lo, hi = crop
    x, y = lo
    w, h = hi - lo
    grid = np.asarray(ego["grid"])
    edge = np.asarray(ego["edge"]) * np.array([width, height]) / grid
    body = (y + np.arange(h)[:, None]) >= np.interp(x + np.arange(w), edge[:, 0], edge[:, 1])
    if body.all():
        return None
    image = np.array(Image.open(raw_image).convert("RGB").crop((*lo, *hi)))
    depth = np.full((h, w), np.inf, "f4")
    # The host's original pavement will be replaced. Only non-ground LiDAR points can occlude it.
    above = mesh.foreground_points(points, anchor, .10) if grouped else (points - center) @ basis[:, 2] > 0.10
    cam = to_cam(points[above])
    radial = np.linalg.norm(cam, axis=1)
    cam = cam[(radial > 0.01) & (radial < 70)]
    uv = project(cam) - lo
    keep = (uv >= 0).all(1) & (uv < [w - 0.5, h - 0.5]).all(1)
    uv = np.rint(uv[keep]).astype(int)
    if len(uv):
        candidates = cam[keep]
        radial = np.linalg.norm(candidates, axis=1)
        rays, valid = unproject(x, y, w, h)
        rows, cols = uv[:, 1], uv[:, 0]
        # A forward model can fold at the lens boundary. Reject returns from the wrong
        # hemisphere by comparing the projected pixel's true bearing to the 3-D return.
        matches = valid[rows, cols] & ((rays[rows, cols] * candidates).sum(-1) > 0.999 * radial)
        np.minimum.at(depth, (rows[matches], cols[matches]), radial[matches].astype("f4"))
    depth = cv2.erode(depth, np.ones((7, 7), "u1"))
    foreground = depth if grouped else None
    receiver_band = None
    if mesh.standing and raw_image.parent.name == "image_00":
        if "stereo_receiver_band_m" in anchor:
            # Keep independently observed non-road LiDAR in front of the replacement,
            # even where an uncertain binocular match appears close to the road.
            lidar = depth
            receiver_band = anchor["stereo_receiver_band_m"]
        depth = stereo_ranges(raw_image)[y:y+h, x:x+w]
        if receiver_band is not None:
            # An eroded LiDAR range alone can extend the side of a pole over unrelated
            # asphalt. Require independent binocular support before treating it as a hard occluder.
            both = np.isfinite(lidar) & np.isfinite(depth)
            residual = np.abs(np.subtract(lidar, depth, out=np.zeros_like(lidar), where=both))
            confirmed = both & (residual < np.maximum(.12, .025 * depth))
            foreground = np.where(confirmed, np.maximum(lidar, depth), np.inf)
    samples, packet_rays, packet_masks = [], [], []
    for dx, dy in SUBPIXELS:
        rays, valid = unproject(x, y, w, h, dx, dy)
        protected = body | ~valid
        if grouped:
            samples.append(mesh.sample(image, pose, rays, anchor, depth, protected, occluder_range=depth,
                                       pixel_width=len(SUBPIXELS)**-0.5))
        else:
            samples.append(mesh.render(image, pose, rays, anchor, depth, protected, with_shadow=mesh.standing,
                                       occluder_range=foreground, receiver_band_m=receiver_band,
                                       pixel_width=len(SUBPIXELS)**-.5))
        if packet_path:
            packet_rays.append(rays)
            packet_masks.append(protected)
    after, part, distance, *receiver = resolve_samples([s[:3] for s in samples] if grouped else samples)
    instance = resolve_instances(samples) if grouped else None
    shadow = receiver[0] if receiver else np.zeros(part.shape,'f4')
    if packet_path:
        np.savez_compressed(
            packet_path,
            **({"sensor_name": np.array(sensor_name)} if "pavement_appearance_by_camera" in anchor else {}),
            rgb=image,
            camera_to_world=pose,
            host_range=depth,
            **({"occluder_range": foreground} if foreground is not None else {}),
            **({"receiver_band_m": np.float64(receiver_band)} if receiver_band is not None else {}),
            rays_camera=packet_rays,
            protected=packet_masks,
        )
    hit = part > 0
    if hit.sum() < 12:
        return None
    mod = np.any(after != image, axis=2)
    if mod[~hit & (shadow == 0)].any() or mod[body].any():
        raise ValueError("A protected or out-of-support pixel changed")
    edges = [side for side, touches in (
        ("left", x == 0 and hit[:, 0].any()), ("right", hi[0] == width and hit[:, -1].any()),
        ("top", y == 0 and hit[0].any()), ("bottom", hi[1] == height and hit[-1].any())
    ) if touches]
    return dict(
        instance=instance,
        before=image,
        after=after,
        part=part,
        range=distance,
        xy=lo,
        visible=int(hit.sum()),
        changed=int(mod.sum()),
        protected_ego=int(body.sum()),
        shadow=shadow,
        clipped_sides=edges,
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


def measure_road_light(anchor, views, cameras, data, filename, ego_masks, models):
    """Fix exposure from the original support patch across usable physical-camera observations."""
    basis = np.asarray(anchor["basis_columns"])
    xx, yy = np.meshgrid(np.linspace(-.5,.5,13), np.linspace(-.5,.5,9))
    size = anchor["footprint_support"]["size_m"]
    points = np.stack([xx*size[0], yy*size[1], np.zeros_like(xx)], -1).reshape(-1,3) @ basis.T + anchor["center_world"]
    colors = []
    for cam, groups in views:
        sensor = data["sensors"][cam]
        (width, height), project, inverse = models[cam]
        mask = ego_masks[sensor]
        edge = np.array(mask['edge']) * [width/mask['grid'][0],height/mask['grid'][1]]
        for g in list(groups)[::3]:
            w2c = cameras[f"{g:03}{cam}0.png"]
            xyz = points @ w2c[:3,:3].T + w2c[:3,3]
            uv = project(xyz)
            usable = ((uv >= 1).all(1) & (uv < [width-2,height-2]).all(1) &
                      (uv[:,1] < np.interp(uv[:,0],edge[:,0],edge[:,1])-2))
            uv = np.rint(uv[usable]).astype(int)
            if len(uv) < 35:
                continue
            lo,hi = uv.min(0),uv.max(0)+1
            rays, valid = inverse(*lo, *(hi-lo))
            coords = uv-lo
            directions = xyz[usable] / np.linalg.norm(xyz[usable],axis=1,keepdims=True)
            good = valid[coords[:,1],coords[:,0]] & ((rays[coords[:,1],coords[:,0]]*directions).sum(1)>.999)
            if good.sum() < 35:
                continue
            with Image.open(filename(g,sensor)) as photo:
                patch = np.asarray(photo.convert("RGB").crop((*lo,*hi)))
            colors.append(np.median(patch[coords[good,1],coords[good,0]],axis=0))
    if not colors:
        raise ValueError("No registered exposures of the support surface for object illumination")
    return dict(road_srgb=np.median(colors,axis=0).tolist(), reference_exposures=len(colors), direct_fraction=0.)


def timed_frames(candidates):
    segments = []
    for item in sorted(candidates, key=lambda v: v[2]["group"]):
        if not segments or item[2]["group"] != segments[-1][-1][2]["group"] + 1:
            segments.append([])
        segments[-1].append(item)
    for i, clip in enumerate(segments):
        times = np.asarray([r["time"] for _, _, r, _ in clip])
        for j, t in enumerate(np.arange(times[0], times[-1] + 0.08, 0.1)):
            yield i > 0 and j == 0, clip[int(np.argmin(abs(times - t)))]


def video_segment(writer, candidates, mesh_id, label, scene, sensor):
    from PIL import ImageDraw, ImageFont, ImageOps

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    header = ImageFont.truetype(font_path, 26, index=2)
    caption = ImageFont.truetype(font_path, 19, index=2)
    for cut, (_, frame, r, photo) in timed_frames(candidates):
        if cut:
            intertitle = Image.new("RGB", (960, 480), "#20272a")
            ImageDraw.Draw(intertitle).text((42, 207), f"下一段可见画面  ·  曝光 {frame}", font=header, fill="white")
            for _ in range(4):
                writer.append_data(np.asarray(intertitle))
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
        yy, xx = np.nonzero(r["part"])
        margin = max(15, round(max(np.ptp(xx), np.ptp(yy)) * 0.18))
        # Keep the same real pixels and framing in both magnified views.
        box = (max(0, xx.min() - margin), max(0, yy.min() - margin),
               min(r["part"].shape[1], xx.max() + margin + 1), min(r["part"].shape[0], yy.max() + margin + 1))
        for y, name, key in [(52, "原路面", "before"), (256, "同机位迁入后", "after")]:
            d.text((610, y), name, font=caption, fill="#d4dedb")
            detail = ImageOps.contain(Image.fromarray(r[key]).crop(box), (330, 167), Image.Resampling.LANCZOS)
            page.paste(detail, (600 + (340 - detail.width) // 2, y + 32 + (167 - detail.height) // 2))
        writer.append_data(np.asarray(page))
    for _ in range(5):
        writer.append_data(np.asarray(page))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mixed-road")
    parser.add_argument("--scene", choices=["kitti360-0000", "kitti360-0003-right", "a2d2-frontleft"], default="kitti360-0000")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--layout", type=Path, help="Fit and render all openings of a pavement recipe together")
    parser.add_argument("--group", type=int, default=26, help="host trajectory index used to seed the anchor")
    parser.add_argument("--offset", type=float, default=2.25, help="metres toward camera right")
    parser.add_argument("--surveyed-depth", action="store_true",
                        help="Compare stereo road returns against measured local normal uncertainty and LiDAR foreground")
    parser.add_argument("--ego-mask",type=Path,help="Optional per-scene calibrated physical camera body outlines")
    parser.add_argument("--support-radius", type=float, default=2.1, help="KITTI candidate terrain search radius in metres")
    parser.add_argument("--seed-lift", type=float, default=0., help="KITTI candidate height above the nominal road, e.g. a sidewalk")
    parser.add_argument("--lighting", type=Path, help="Optional standing-object ground RGB, sun_world and direct_fraction JSON")
    parser.add_argument(
        "--ids", nargs="+", default=["P220", "P221", "P222", "P223", "A224", "A225", "A226", "A227", "M224"]
    )
    parser.add_argument(
        "--video", action="store_true", help="Encode consecutive, registered physical-camera frames"
    )
    parser.add_argument(
        "--export-packet", help="Also save the best cropped rays, ranges, foreground and RGB for this ID"
    )
    parser.add_argument(
        "--paired-preview",
        action="store_true",
        help="Export a frame jointly showing the layout members, ranked by their harmonic mean visible area",
    )
    parser.add_argument("--appearance-views",type=Path,help="Directory of surveyed original camera packets (same layout and metric placement)")
    args = parser.parse_args()
    if args.paired_preview and not args.layout:
        parser.error("--paired-preview requires a pavement layout")
    if args.appearance_views and not args.layout:
        parser.error("--appearance-views requires a surveyed pavement layout")
    rig = ROOT / "work/streetcrafter-rig" / args.scene
    data, cams = load_rig(rig)
    if args.scene == "a2d2-frontleft":
        raw = ROOT / "work/data" / args.scene
        anchor, views, cloud_at, filename = audi_inputs(raw, data, cams, args.group, args.offset)
        nearby = np.concatenate([cloud_at(g, cam) for cam, groups in views for g in groups if g % 3 == 0])
    else:
        raw = ROOT / "work/data" / ("kitti360-0000-0200" if args.scene == "kitti360-0000" else args.scene)
        clouds = scans(raw, data, cams)
        rig_pose = np.linalg.inv(cams[f"{args.group:03}00.png"])
        seed = rig_pose[:3, 3] + args.offset * rig_pose[:3, 0] - [0, 0, 1.6-args.seed_lift]
        nearby = (np.concatenate(clouds[max(0, args.group - 12):args.group + 13]) if args.layout
                  else np.concatenate(clouds[max(0, args.group - 7) : args.group + 8 : 2]))
        anchor = fit_road(nearby, seed, rig_pose[:3, 2], up_world=[0, 0, 1], radius_m=args.support_radius)
        views = [
            (0, range(max(0, args.group - 18), args.group)),
            (3, range(max(0, args.group - 6), min(len(clouds), args.group + 8))),
        ]

        def cloud_at(g, cam):
            return clouds[g]

        def filename(g, sensor):
            return raw / sensor / f"{data['frames'][g]:010}.png"

    out = args.out or ROOT / "outputs/mixed-road" / args.scene
    assets = {}
    for key in ([] if args.layout else args.ids):
        folder = args.run / key
        if not (folder / "asset.glb").is_file():
            (folder,) = args.run.glob(f"*/series/assets/*/{key}")
        assets[key] = folder
    registration_anchor = anchor
    layout = None
    if args.layout:
        layout = PavementLayout(args.layout, args.run)
        anchor = layout.fit(anchor, nearby, np.asarray(anchor["basis_columns"])[:, 2], args.support_radius)
        layout.bind(anchor)
        anchor.update(host_scene=args.scene, placement_group=args.group)
    else:
        anchor = fit_footprint(anchor, nearby, np.max([footprint_size(path) for path in assets.values()], axis=0))
    out.mkdir(parents=True, exist_ok=True)
    ego_masks = json.loads((ROOT / "gen2hole/examples/mixed-road/ego-masks.json").read_text())[args.scene]
    if args.ego_mask:
        ego_masks.update(json.loads(args.ego_mask.read_text())[args.scene])
    models = {cam: view_model(data["sensors"][cam], data) for cam, _ in views}
    standing_bounds = []
    registered = []
    for folder in assets.values():
        with np.load(folder / "surface.npz", allow_pickle=False) as field:
            standing = "sky_blocked" in field
        if standing:
            registered.append(registered_geometry(folder))
            for _, xyz, _ in glb_triangles(folder / "asset.glb"):
                standing_bounds.extend((xyz[:, :2].min(0), xyz[:, :2].max(0)))
    if standing_bounds:
        anchor["registered_geometry"] = registered
        anchor["host_scene"] = args.scene
        anchor["placement_group"] = args.group
        ref = data['source_calibration'][data['sensors'][0]]
        to_body = np.array(ref.get('camera_to_pose', ref.get('optical_to_body')))
        body = np.stack([np.linalg.inv(cams[f'{g:03}00.png']) @ np.linalg.inv(to_body) for g in range(len(data['frames']))])
        centers = body[:,:3,3] + body[:,:3,0]
        bounds = np.stack(standing_bounds)
        anchor['vehicle_clearance'] = driving_clearance(anchor, [bounds.min(0),bounds.max(0)], centers,
                                                       half_width_m=1.03,half_length_m=2.55)
        anchor['vehicle_clearance']['envelope_center_forward_of_rig_m'] = 1.
        anchor['ground_receivers'] = fit_receivers(nearby, anchor)
        if args.surveyed_depth:
            anchor["stereo_receiver_band_m"] = float(np.clip(3 * anchor["rms_m"], .025, .065))
        anchor['lighting'] = measure_road_light(anchor, views, cams, data, filename, ego_masks, models)
        if args.lighting:
            anchor['lighting'].update(json.loads(args.lighting.read_text()))
    if args.appearance_views:
        from pavement_appearance import register_views

        previous = json.loads((args.appearance_views / "anchor.json").read_text())
        if previous != anchor:
            raise ValueError("Appearance packets must use the same host, layout and fitted metric anchors")
        anchor["pavement_appearance_by_camera"] = {
            data["sensors"][cam]: register_views(
                layout, anchor, [args.appearance_views / layout.key / data["sensors"][cam] / "view.npz"]
            )
            for cam, _ in views
        }
    save_json(out / "anchor.json", anchor)
    windows = {cam: RayWindow(model) for cam, model in models.items()}
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
        for key, folder in ([(layout.key, args.layout.parent)] if layout else assets.items()):
            mesh = layout if layout else RoadMesh(folder)
            index[key] = {"source_sha256": mesh.digest, "views": {}}
            for cam, groups in views:
                sensor = data["sensors"][cam]
                model = models[cam]
                candidates = []
                for g in groups:
                    name = f"{g:03}{cam}0.png"
                    pose = np.linalg.inv(cams[name])
                    frame = data["views"][name]["source_frame"]
                    photo = filename(g, sensor)
                    points = cloud_at(g, cam)
                    correction = None
                    if args.scene == "a2d2-frontleft":
                        adjusted = align_observed_road(registration_anchor, pose, points)
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
                        windows[cam],
                        ego_masks[sensor], sensor_name=sensor,
                    )
                    if res:
                        res.update(alignment=correction, group=g, time=data["times"][name])
                        candidates.append((res["visible"], frame, res, photo))
                if not candidates:
                    raise ValueError(f"{key}: no visible support for {sensor}")
                if args.paired_preview:

                    def joint_area(candidate):
                        ids = candidate[2]["instance"]
                        counts = np.bincount(ids.ravel(), minlength=len(mesh.members) + 1)[1:]
                        return 0.0 if np.any(counts < 12) else len(counts) / np.sum(1.0 / counts)

                    chosen = max(candidates, key=joint_area)
                    if not joint_area(chosen):
                        raise ValueError(f"{key}: no {sensor} exposure jointly resolves all members")
                else:
                    chosen = max(candidates, key=lambda c: c[0])
                _, f, best, photo = chosen
                dest = out / key / sensor
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(best["before"]).save(dest / "before.png")
                Image.fromarray(best["after"]).save(dest / "after.png")
                Image.fromarray(best["part"]).save(dest / "part.png")
                np.save(dest / "range.npy", best["range"])
                if layout:
                    Image.fromarray(best["instance"]).save(dest / "instance.png")
                if mesh.standing:
                    np.save(dest / "shadow.npy", best["shadow"])
                image = Image.open(photo).convert("RGB")
                image.paste(Image.fromarray(best["after"]), tuple(best["xy"]))
                image.save(dest / "full.jpg", quality=94)
                if writer:
                    label = layout.label if layout else json.loads((folder / "metadata.json").read_text())["label"]
                    video_segment(writer, candidates, key, label, args.scene, sensor)
                if args.export_packet == key:
                    g = best["group"]
                    pose = np.linalg.inv(cams[f"{g:03}{cam}0.png"])
                    points = cloud_at(g, cam)
                    if args.scene == "a2d2-frontleft":
                        pose, points, _ = align_observed_road(registration_anchor, pose, points)
                    frame_crop(
                        mesh,
                        anchor,
                        model,
                        pose,
                        photo,
                        points,
                        windows[cam],
                        ego_masks[sensor],
                        dest / "view.npz",sensor_name=sensor,
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
                    image_edge_frames=[{"frame": frame, "sides": r["clipped_sides"]}
                                       for _, frame, r, _ in candidates if r["clipped_sides"]],
                    model=data["source_calibration"][sensor]["model"],
                )
                if layout:
                    row["best_frame_instance_pixels"] = {
                        item["id"]: int((best["instance"] == i).sum()) for i, item in enumerate(anchor["instances"], 1)
                    }
                    row["frames_showing_all_instances"] = [
                        frame for _, frame, result, _ in candidates
                        if np.count_nonzero(np.bincount(result["instance"].ravel(), minlength=len(anchor["instances"])+1)[1:]) == len(anchor["instances"])
                    ]
                index[key]["views"][sensor] = row
                print(key, sensor, f, best["visible"], flush=True)
            save_json(out / "index.json", index)
    if args.video:
        (out / "driving.tmp.mp4").replace(out / "driving.mp4")


if __name__ == "__main__":
    main()
