"""Seat the common-20 dry pits in pairs in the trained KITTI-360 joint scene."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from prepare import load_rig

ROOT = Path(__file__).resolve().parents[2]
SCENE = "kitti360-0000"
BASE = ROOT / "runs/streetgs-joint" / SCENE / "30000"
RIG = ROOT / "work/streetcrafter-rig" / SCENE
CATALOG = ROOT / "catalog/common-pit-kit.json"
RESULT = ROOT / "outputs/streetgs-joint" / SCENE / "30000/pit-pairs"
C0 = 0.28209479177387814
PAIR_IDS = [f"P{i}" for i in range(205, 220)] + ["P190", "P198", "P199", "P200", "P201"]
ANCHORS = [(20, 0.48, -12), (33, -0.25, 14)]  # frame group, lateral offset (m), heading offset (degrees)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def unit(v):
    return v / np.linalg.norm(v)


def road_scans(cameras):
    raw = ROOT / "work/data/kitti360-0000-0200"
    calib = raw / "calibration"
    rect = np.eye(4)
    for line in (calib / "perspective.txt").read_text().splitlines():
        key, values = line.split(":", 1) if ":" in line else ("", "")
        if key == "R_rect_00":
            rect[:3, :3] = np.fromstring(values, sep=" ").reshape(3, 3)
    camera_to_velo = np.eye(4)
    camera_to_velo[:3] = np.loadtxt(calib / "calib_cam_to_velo.txt").reshape(3, 4)
    velo_to_rect = rect @ np.linalg.inv(camera_to_velo)
    points = []
    for g in range(7, 40, 3):
        cloud = np.fromfile(raw / "velodyne" / f"{200+g:010}.bin", "<f4").reshape(-1, 4)[:, :3]
        pose = np.linalg.inv(cameras[f"{g:03}00.png"]) @ velo_to_rect
        points.append(cloud @ pose[:3, :3].T + pose[:3, 3])
    return np.concatenate(points)


def anchor(scans, cameras, group, offset):
    pose = np.linalg.inv(cameras[f"{group:03}00.png"])
    xy = pose[:2, 3] + offset * unit(pose[:2, 0])
    select = (np.linalg.norm(scans[:, :2] - xy, axis=1) < 3.6) & (abs(scans[:, 2] - (pose[2, 3] - 1.6)) < .30)
    pts = scans[select]
    x = np.column_stack((pts[:, :2] - xy, np.ones(len(pts))))
    keep = np.ones(len(x), bool)
    for threshold in (.09, .055, .038, .032):
        plane = np.linalg.lstsq(x[keep], pts[keep, 2], rcond=None)[0]
        keep = abs(x @ plane - pts[:, 2]) < threshold
    if keep.sum() < 200 or np.linalg.norm(plane[:2]) > .14:
        raise ValueError("Road plane lacks LiDAR support")
    up = unit(np.r_[-plane[:2], 1.])
    forward = unit(pose[:3, 2] - up * (pose[:3, 2] @ up))
    center = np.r_[xy, plane[2]]
    fit = dict(lidar_inliers=int(keep.sum()), rms_m=float(np.sqrt(np.mean((x[keep] @ plane-pts[keep, 2])**2))))
    return center, np.column_stack((forward, np.cross(up, forward), up)), fit


def projection(world, camera, calibration):
    _, width, height, fx, fy, cx, cy = calibration
    xyz = world @ camera[:3, :3].T + camera[:3, 3]
    z = xyz[:, 2]
    uv = xyz[:, :2] / np.maximum(z[:, None], .01) * [fx, fy] + [cx, cy]
    return uv, z, width, height


def road_plate(center, basis, cameras, split):
    # Register the original joint renderings to the fitted road before inserting either pit.
    axis = np.arange(-6, 6.001, .035)
    x, y = np.meshgrid(axis, axis)
    world = np.stack((x, y, np.zeros_like(x)), -1) @ basis.T + center
    accum = np.zeros((*x.shape, 3))
    total = np.zeros(x.shape)
    for group in range(0, 36, 2):
        for sensor in (0, 1):
            name = f"{group:03}{sensor}0.png"
            k = split["projections"]["eval"][split["views"][name]["projection"]]
            uv, z, w, h = projection(world.reshape(-1, 3), cameras[name], k)
            uv = uv.reshape(*x.shape, 2).astype("f4")
            z = z.reshape(x.shape)
            weight = np.clip((z-3.8)/1.8, 0, 1) * np.clip((23-z)/9, 0, 1) / np.maximum(z, 4)**2
            weight *= np.clip(np.minimum(uv[..., 0], w-1-uv[..., 0])/14, 0, 1)
            weight *= np.clip(np.minimum(uv[..., 1], h-1-uv[..., 1])/5, 0, 1)
            if weight.max() > 0:
                img = np.asarray(Image.open(BASE / "frames" / name).convert("RGB"))
                accum += cv2.remap(img, uv, None, cv2.INTER_LINEAR) * weight[..., None]
                total += weight
    good = total > 1e-8
    plate = (accum / np.maximum(total, 1e-8)[..., None] / 255).astype("f4")
    near = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True)
    plate = plate[tuple(near)]
    return axis, plate


def colors(plate, xy):
    axis, rgb = plate
    coords = ((xy - axis[0])/(axis[1]-axis[0])).T[::-1]
    return np.stack([ndimage.map_coordinates(channel, coords, order=1, mode="nearest")
                     for channel in rgb.transpose(2, 0, 1)], axis=1)


def ply_vertices(path):
    with path.open("rb") as f:
        header = []
        for _ in range(100):
            line = f.readline().strip()
            header.append(line)
            if line == b"end_header":
                break
        if header[-1] != b"end_header" or b"format binary_little_endian 1.0" not in header:
            raise ValueError(f"Unexpected PLY {path}")
        n = int(next(v for v in header if v.startswith(b"element vertex ")).split()[-1])
        props = [v.split()[-1] for v in header if v.startswith(b"property float ")]
        if len(props) != 26 or props[18:] != [b"opacity", b"scale_0", b"scale_1", b"scale_2",
                                             b"rot_0", b"rot_1", b"rot_2", b"rot_3"]:
            raise ValueError("Unexpected Gaussian layout")
        return np.fromfile(f, "<f4").reshape(n, len(props))


class Footprint:
    def __init__(self, path):
        with np.load(path) as f:
            mask = f["defect_mask"] > .5
            self.width, self.length = float(f["width"]), float(f["length"])
        self.h, self.w = mask.shape
        row, col = np.nonzero(mask & ~ndimage.binary_erosion(mask))
        self.edge = np.column_stack(((col/(self.w-1)-.5)*self.width, (.5-row/(self.h-1))*self.length))
        self.tree = cKDTree(self.edge)
        self.mask = mask

    def signed_distance(self, xy):
        row = np.rint((.5 - xy[:, 1]/self.length)*(self.h-1)).astype(int)
        col = np.rint((xy[:, 0]/self.width + .5)*(self.w-1)).astype(int)
        valid = (row >= 0) & (row < self.h) & (col >= 0) & (col < self.w)
        inside = np.zeros(len(xy), bool)
        inside[valid] = self.mask[row[valid], col[valid]]
        distance = self.tree.query(xy, workers=3)[0]
        return np.where(inside, -distance, distance)


def particles(xyz, quat, scale, alpha, rgb, rest=None):
    n = len(xyz)
    sh = np.zeros((n, 15, 3), dtype="f4")
    if rest is not None:
        sh[:, :3] = rest
    arrays = dict(_xyz=xyz, _rotation=quat, _scaling=np.log(scale),
                  _opacity=np.log(alpha/(1-alpha)).reshape(-1, 1),
                  _features_dc=((rgb-.5)/C0)[:, None, :], _features_rest=sh,
                  _semantic=np.empty((n, 0)))
    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype="f4")) for k, v in arrays.items()}


def terrain_asset(key, case, site, plate, heading):
    folder = (ROOT / case["model"]).parent
    path = folder / "asset.ply"
    if digest(path) != case["native_hashes"]["asset.ply"]:
        raise ValueError("Source PLY hash mismatch: "+key)
    vertices = ply_vertices(path)
    with np.load(folder/"splat_material.npz") as f:
        bed = ~f["road"]
        visibility = f["ambient_visibility"].astype("f4") / 255
    foot = Footprint(folder/"surface.npz")
    center, road, fit = site
    eig = np.linalg.eigh(np.cov(vertices[bed, :2].T))[1][:, -1]
    angle = np.deg2rad(heading) - np.arctan2(eig[1], eig[0])
    basis = road @ Rotation.from_rotvec([0, 0, angle]).as_matrix()
    # Keep an actual source splat per spatial cell, including separate vertical lip samples.
    src = np.flatnonzero(bed)
    voxel = .019
    _, picks = np.unique(np.floor(vertices[src, :3] / voxel).astype("i4"), axis=0, return_index=True)
    ids = src[picks]
    v = vertices[ids]
    world = v[:, :3] @ basis.T + center
    point_rot = Rotation.from_quat(v[:, [23, 24, 25, 22]])
    quat = (Rotation.from_matrix(basis) * point_rot).as_quat()[:, [3, 0, 1, 2]]
    scales = np.exp(v[:, 19:22])
    scales[:, :2] = np.maximum(scales[:, :2], voxel*.87)
    scales[:, 2] = np.maximum(scales[:, 2], .0015)
    road_rgb = np.median(vertices[~bed, 6:9] * C0 + .5, axis=0)
    road_xy = ((world-center) @ road)[:, :2]
    axis, image = plate
    # Low-pass target road light carries tree shadows into the cavity. High frequencies come from the source.
    light = colors((axis, cv2.GaussianBlur(image, (0, 0), 6)), road_xy)
    ratio = np.clip((v[:, 6:9]*C0+.5)/np.maximum(road_rgb, .12), .16, 1.85)
    ao = .68 + .32 * visibility[ids]
    rgb = np.clip(light * ratio * ao[:, None] * .94, .006, .985)
    alpha = np.clip(1/(1+np.exp(-v[:, 18])), .70, .986)
    # PLY first-order bands (-y,z,-x) are in source coordinates.
    sh = v[:, 9:18].reshape(-1, 3, 3)
    cart = np.stack((-sh[:, :, 2], -sh[:, :, 0], sh[:, :, 1]), axis=2) @ basis.T
    rest = np.stack((-cart[:, :, 1], cart[:, :, 2], -cart[:, :, 0]), axis=1) * .3
    bed_state = particles(world, quat, scales, alpha, rgb, rest)

    # Replace support for intersecting broad host splats with a narrow feathered strip of this road.
    step = .045
    low, high = foot.edge.min(0)-1.50, foot.edge.max(0)+1.50
    xx, yy = np.meshgrid(np.arange(low[0], high[0], step), np.arange(low[1], high[1], step))
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    dist = foot.signed_distance(xy)
    keep = (dist > -.008) & (dist < 1.45)
    xy, dist = xy[keep], dist[keep]
    floor = np.column_stack((xy, np.full(len(xy), -.018))) @ basis.T + center
    tint = colors(plate, ((floor-center)@road)[:, :2])
    opacity = np.clip((1.45-dist)/.55, .003, .955)
    orientation = Rotation.from_matrix(basis).as_quat()[[3, 0, 1, 2]]
    strip = particles(floor, np.tile(orientation, (len(xy), 1)),
                      np.tile([step*.89, step*.89, .002], (len(xy), 1)), opacity, tint)
    edge = np.column_stack((foot.edge, np.zeros(len(foot.edge)))) @ basis.T + center
    record = dict(id=key, label=case["label"], center_world=center.tolist(), basis_columns=basis.tolist(),
                  source_sha256=case["native_hashes"]["asset.ply"], source_ply=str(path.relative_to(ROOT)),
                  scale=1.0, source_bed_points=int(bed.sum()), inserted_bed_points=len(ids),
                  road_support_points=len(xy), outline_area_m2=case["outline_area_m2"], **fit)
    return (foot, center, basis), [bed_state, strip], edge, record


def carve(layer, world, placements):
    cut = np.zeros(len(world), bool)
    rotations = Rotation.from_quat(layer["_rotation"].numpy()[:, [1, 2, 3, 0]]).as_matrix()
    sigma = layer["_scaling"].exp().numpy()
    for foot, origin, basis in placements:
        xyz = (world-origin) @ basis
        indices = np.flatnonzero((np.linalg.norm(xyz[:, :2], axis=1) < max(foot.width, foot.length)+2) &
                                 (xyz[:, 2] > -1.2) & (xyz[:, 2] < .34))
        support = np.linalg.norm(np.einsum("i,nij->nj", basis[:, 2], rotations[indices])*sigma[indices], axis=1)
        road = (xyz[indices, 2] < .19) | (abs(xyz[indices, 2]) < 1.5*support)
        margin = np.minimum(.99, 2.6 * sigma[indices].max(1)) + .055
        cut[indices] |= road & (foot.signed_distance(xyz[indices, :2]) < margin)
    return ~cut


def frame_boxes(edges, cameras, split):
    info = {}
    for group in range(len(split["frames"])):
        name = f"{group:03}00.png"
        k = split["projections"]["eval"][split["views"][name]["projection"]]
        boxes = []
        for edge in edges:
            uv, z, w, h = projection(edge, cameras[name], k)
            if z.min() < 1 or np.any(uv.max(0) < [0, 0]) or np.any(uv.min(0) >= [w, h]):
                boxes.append(None)
            else:
                boxes.append(np.r_[uv.min(0), uv.max(0)].round(2).tolist())
        info[name] = boxes
    return info


def pair_name(order):
    first, second = PAIR_IDS[(order-1)*2:order*2]
    return f"{order:02}-{first}-{second}", (first, second)


def build_pairs(orders):
    split, cameras = load_rig(RIG)
    scans = road_scans(cameras)
    sites = [anchor(scans, cameras, g, offset) for g, offset, _ in ANCHORS]
    plates = [road_plate(s[0], s[1], cameras, split) for s in sites]
    baseline = torch.load(BASE/"joint.pt", weights_only=False, map_location="cpu")
    if (baseline["scene"], baseline["steps"]) != (SCENE, 30000):
        raise ValueError("Incorrect source checkpoint")
    source_digest = digest(BASE/"joint.pt")
    cases = json.loads(CATALOG.read_text())["assets"]
    for order in orders:
        name, keys = pair_name(order)
        output = BASE/"pit-pairs"/name
        output.mkdir(exist_ok=True, parents=True)
        asset_data = [terrain_asset(k, cases[k], site, plate, a[2])
                      for k, site, plate, a in zip(keys, sites, plates, ANCHORS)]
        placements, states, edges, records = zip(*asset_data)
        separation = cKDTree(edges[0]).query(edges[1])[0].min()
        if separation < .45:
            raise ValueError("Overlapping pit boundaries")
        state = dict(baseline, layers=copy.copy(baseline["layers"]))
        removed = {}
        for layer_name, layer in baseline["layers"].items():
            world = layer["_xyz"].numpy()
            if layer_name != "background":
                actor = next(v for v in baseline["metadata"]["obj_meta"].values()
                             if v["id"] == int(layer_name.split("_")[-1]))
                tracks = baseline["metadata"]["camera_tracklets"][:, :, actor["id"]]
                if actor["class"] != "untracked residual" or not np.allclose(tracks, tracks[0, 0]):
                    continue
                t = tracks[0, 0]
                world = world @ Rotation.from_quat(t[[4, 5, 6, 3]]).as_matrix().T + t[:3]
            keep = carve(layer, world, placements)
            removed[layer_name] = int((~keep).sum())
            layers = [{k: v[keep] for k, v in layer.items()}]
            if layer_name == "background":
                layers.extend(s for pair in states for s in pair)
            state["layers"][layer_name] = {k: torch.cat([s[k] for s in layers], 0) for k in layer}
        report = dict(pair=name, source_checkpoint=str((BASE/"joint.pt").relative_to(ROOT)), source_sha256=source_digest,
                      placements=list(records), boundary_separation_m=float(separation), removed=removed,
                      projected_edges=frame_boxes(edges, cameras, split))
        state["pit_pair"] = {k: report[k] for k in ("pair","source_sha256","placements")}
        tmp = output/"joint.stage.pt"
        torch.save(state, tmp)
        tmp.replace(output/"joint.pt")
        report["checkpoint_sha256"] = digest(output/"joint.pt")
        (output/"placement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(name, removed, f"edge clearance {separation:.2f} m", flush=True)


def hd_camera(batch, pinhole=None):
    intrinsics = dict(batch.intrinsics_OpenCVPinholeCameraModelParameters)
    if pinhole is None:
        for key in ("resolution", "focal_length", "principal_point"):
            intrinsics[key] = np.asarray(intrinsics[key]) * 2
    else:
        _, w, h, fx, fy, cx, cy = pinhole
        for key, values in [("resolution", (w,h)), ("focal_length", (fx,fy)), ("principal_point", (cx,cy))]:
            intrinsics[key] = np.array(values, dtype=np.asarray(intrinsics[key]).dtype)
    width, height = map(int, intrinsics["resolution"])
    fx, fy = intrinsics["focal_length"]
    cx, cy = intrinsics["principal_point"]
    y, x = torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda"),
                          indexing="ij")
    rays = torch.stack(((x+.5-float(cx))/float(fx), (y+.5-float(cy))/float(fy), torch.ones_like(x)), -1)
    direction = torch.nn.functional.normalize(rays, dim=-1)[None].contiguous()
    return replace(batch, rays_ori=torch.zeros_like(direction), rays_dir=direction, rgb_gt=None,
                   mask=None, pixel_coords=None, intrinsics_OpenCVPinholeCameraModelParameters=intrinsics)


def rear_mount():
    # A rigid rear-facing camera on the vehicle at the recorded stereo-rig exposure times.
    tilt = np.deg2rad(36.)
    relative = np.eye(4)
    relative[:3, :3] = [[-1,0,0], [0,np.cos(tilt),np.sin(tilt)], [0,np.sin(tilt),-np.cos(tilt)]]
    relative[:3, 3] = [0, .32, -3.]
    f = 640 / np.tan(np.deg2rad(54.))
    return relative, ["PINHOLE",1280,720,f,f,640.,360.]


def render_pairs(orders, hd=False, rear=False):
    from omegaconf import OmegaConf
    from run import configure
    official = Path(os.getenv("OFFICIAL_ROOT", ROOT.parent.parent/"01_official")).resolve()
    os.chdir(official/"3dgrut")
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    model_type, _ = configure(ROOT, SCENE, BASE, official)
    base = torch.load(BASE/"joint.pt", weights_only=False, map_location="cpu")
    model = model_type(base["metadata"], OmegaConf.create(base["renderer"]))
    split, cameras = load_rig(RIG)
    ds = ColmapDataset(str(RIG/"sensor_eval"), test_split_interval=0)
    idx = {Path(path).name: i for i, path in enumerate(ds.image_paths)}
    low, high = min(split["times"].values()), max(split["times"].values())
    views = []
    relative, rear_k = rear_mount()
    camera_path = {}
    for g in range(20 if rear else 0, len(split["frames"])):
        n = f"{g:03}00.png"
        batch = ds.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([ds[idx[n]]]))
        w2c = cameras[n]
        k = split["projections"]["eval"][split["views"][n]["projection"]]
        pose = batch.T_to_world
        if rear:
            pose = pose @ pose.new_tensor(relative)
            w2c = np.linalg.inv(relative) @ w2c
            if not np.allclose(pose[0].cpu().numpy(), np.linalg.inv(w2c), atol=2e-5):
                raise ValueError("Rear view does not match the calibrated vehicle trajectory")
            k = rear_k
            camera_path[n] = w2c.tolist()
        if hd:
            if not views:
                calibration = hd_camera(batch, k if rear else None)
            batch = replace(calibration, T_to_world=pose, frame_idx=batch.frame_idx)
        t = split["times"][n]
        exposure = model.exposure(g, 0, t, n in split["test"], w2c, k)
        if rear:
            exposure[0].meta["is_novel_view"] = True
        time = torch.tensor((t-low)/(high-low), device="cuda", dtype=torch.float32)
        views.append((n, batch, exposure, time))
    model.eval()
    with torch.no_grad():
        for order in [0] + list(orders):
            if order:
                name, keys = pair_name(order)
                run = BASE/"pit-pairs"/name
                source = run/"joint.pt"
                report = json.loads((run/"placement.json").read_text())
                if digest(source) != report["checkpoint_sha256"]:
                    raise ValueError("Pair checkpoint hash mismatch")
                state = torch.load(source, map_location="cpu", weights_only=False)
                if [p["id"] for p in state["pit_pair"]["placements"]] != list(keys):
                    raise ValueError("Unexpected pair members")
                dest = RESULT/name/("rear-frames" if rear else ("frames-hd" if hd else "frames"))
                dest.mkdir(parents=True, exist_ok=True)
            else:
                state = base
                if hd:
                    dest = RESULT/("original-rear" if rear else "original-hd")
                    dest.mkdir(parents=True, exist_ok=True)
            model.restore(state)
            baseline_error = 0
            for n, batch, exposure, time in views:
                rgb, _ = model(batch, exposure, time)
                if not torch.isfinite(rgb).all():
                    raise ValueError(n)
                arr = rgb[0].clamp(0,1).mul(255).round().byte().cpu().numpy()
                if order or hd:
                    if not cv2.imwrite(str(dest/n), arr[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                        raise OSError(dest/n)
                if not order and not rear:
                    expected = np.asarray(Image.open(BASE/"frames"/n), dtype=int)
                    if hd:
                        resized = cv2.resize(arr, (expected.shape[1], expected.shape[0]), interpolation=cv2.INTER_AREA)
                        err = float(abs(resized.astype(int)-expected).mean())
                    else:
                        err = int(abs(arr.astype(int)-expected).max())
                    baseline_error = max(baseline_error, err)
            if not order and baseline_error > (9 if hd else 2):
                raise ValueError(f"Original checkpoint did not reproduce its frames: {baseline_error}")
            print("rendered", order, len(views), "baseline reload error", baseline_error if not order and not rear else "-", flush=True)
    if rear:
        path = RESULT/"original-rear/camera.json"
        path.write_text(json.dumps(dict(kind="virtual rear camera", parent_sha256=digest(BASE/"joint.pt"),
                                       pinhole=rear_k, rear_to_front=relative.tolist(), w2c=camera_path), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build","render","render-hd","render-rear"))
    parser.add_argument("orders", nargs="*", type=int)
    args = parser.parse_args()
    if any(n not in range(1, 11) for n in args.orders):
        parser.error("Pair numbers run from 1 to 10")
    torch.set_num_threads(3)
    orders = args.orders or range(1, 11)
    if args.action == "build":
        build_pairs(orders)
    else:
        render_pairs(orders, hd=args.action in ("render-hd","render-rear"), rear=args.action=="render-rear")
