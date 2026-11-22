"""Registered RGB, optical depth and labelled pothole pixels for multi-view reconstruction."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat


@dataclass
class View:
    key: str
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    sources: dict

    def xyz(self, uv, z):
        homogeneous = np.column_stack((uv, np.ones(len(uv))))
        return (homogeneous @ np.linalg.inv(self.K).T) * z[:, None]


def checked(view):
    if (
        view.rgb.shape[:2] != view.depth.shape
        or view.mask.shape != view.depth.shape
        or view.K.shape != (3, 3)
        or not np.isfinite(view.K).all()
        or min(view.K[0, 0], view.K[1, 1]) <= 0
    ):
        raise ValueError('Invalid image, depth, mask or camera calibration for ' + view.key)
    view.depth = np.where(
        np.isfinite(view.depth) & (view.depth > 0.15) & (view.depth < 15), view.depth, 0
    ).astype('f4')
    if np.count_nonzero(view.mask & (view.depth > 0)) < 100:
        raise ValueError('Too few valid target samples in ' + view.key)
    return view


def polygons(data, size):
    image = Image.new('L', size)
    draw = ImageDraw.Draw(image)
    for line in data.decode().splitlines():
        xy = np.asarray(line.split()[1:], float).reshape(-1, 2) * (np.array(size) - 1)
        if len(xy) >= 3:
            draw.polygon(list(map(tuple, xy)), fill=1)
    return np.asarray(image) > 0


def pinhole(params):
    fx, fy, cx, cy = params
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def fan_calibration(xyz):
    y, x = np.indices(xyz.shape[:2])
    cloud = xyz[::12, ::12].reshape(-1, 3)
    pixels = np.c_[x[::12, ::12].ravel(), y[::12, ::12].ravel()]
    good = np.isfinite(cloud).all(1) & (cloud[:, 2] > 0.15) & (cloud[:, 2] < 15)
    normalized = cloud[good] / cloud[good, 2:3]
    coeff = np.linalg.lstsq(normalized, pixels[good], rcond=None)[0]
    if np.sqrt(np.mean((normalized @ coeff - pixels[good]) ** 2)) > 0.35:
        raise ValueError('Stereo XYZ does not agree with a pinhole camera')
    return np.vstack([coeff.T, [0, 0, 1]])


def read_views(spec, source, catalog, base):
    keys = list(dict.fromkeys(spec['fit'] + spec['holdout']))
    if any(not isinstance(k, str) or not k or k in ('.', '..') or Path(k).name != k for k in keys):
        raise ValueError('Use frame names that can be stored as individual files')
    if (
        len(keys) != len(spec['fit']) + len(spec['holdout'])
        or not spec['fit']
        or (not spec['holdout'] and spec.get('mode') != 'single_rgbd')
        or (spec.get('mode') == 'single_rgbd' and (len(spec['fit']) != 1 or spec['holdout']))
        or spec['anchor'] not in keys
    ):
        raise ValueError('Use distinct fit and held-out frames and an anchor in those frames')
    views = {}
    if spec['source'] == 'PothRGBD':
        index = json.loads(catalog.read_text())
        folder = source / 'rgbd'
        with (
            ZipFile(folder / 'images.zip') as rgb,
            ZipFile(folder / '-depths.zip') as depth,
            ZipFile(folder / 'labels.zip') as labels,
        ):
            for key in keys:
                record = index['rgbd_pothole_' + key]
                data = {}
                for role, archive in [('rgb', rgb), ('depth', depth), ('mask', labels)]:
                    data[role] = archive.read(record[role]['filename'])
                    if sha256(data[role]).hexdigest() != record[role]['sha256']:
                        raise ValueError('Source member changed: ' + key + ' ' + role)
                image = Image.open(BytesIO(data['rgb'])).convert('RGB')
                w, h = image.size
                raw = np.load(BytesIO(data['depth']), allow_pickle=False)
                K = pinhole([462.0, 462.0, (w - 1) / 2, (h - 1) / 2])
                views[key] = checked(
                    View(key, np.array(image), raw * 0.001, polygons(data['mask'], image.size), K, record)
                )
    elif spec['source'] == 'FanStereo':
        folder = source / 'stereo' / spec['dataset']
        for key in keys:
            paths = {
                'rgb': folder / 'rgb' / (key + '.png'),
                'xyz': folder / 'ptcloud' / (key + '.mat'),
                'mask': folder / 'label' / (key + '.png'),
            }
            xyz = loadmat(paths['xyz'])['xyzPoints'].astype('f4') * 0.001
            record = {
                k: dict(filename=str(p.relative_to(source)), sha256=sha256(p.read_bytes()).hexdigest())
                for k, p in paths.items()
            }
            views[key] = checked(
                View(
                    key,
                    np.array(Image.open(paths['rgb']).convert('RGB')),
                    xyz[:, :, 2],
                    np.array(Image.open(paths['mask']).convert('L')) > 0,
                    fan_calibration(xyz),
                    record,
                )
            )
    elif spec['source'] == 'files':
        for key in keys:
            frame = spec['frames'][key]
            paths = {k: base / frame[k] for k in ('rgb', 'depth', 'mask')}
            raw = (
                np.load(paths['depth'], allow_pickle=False)
                if paths['depth'].suffix == '.npy'
                else np.array(Image.open(paths['depth']))
            )
            if not np.isfinite(frame['depth_scale']) or frame['depth_scale'] <= 0:
                raise ValueError('Expected a positive optical depth scale')
            record = {
                k: dict(filename=frame[k], sha256=sha256(p.read_bytes()).hexdigest())
                for k, p in paths.items()
            }
            views[key] = checked(
                View(
                    key,
                    np.array(Image.open(paths['rgb']).convert('RGB')),
                    raw * float(frame['depth_scale']),
                    np.array(Image.open(paths['mask']).convert('L')) > 0,
                    pinhole(frame['fx_fy_cx_cy']),
                    record,
                )
            )
    else:
        raise ValueError('Unknown capture source ' + spec['source'])
    return views


def features(view):
    # Never use pit texture or interior depth to locate a camera. The guard band
    # also keeps depth discontinuities at the annotated rim out of the pose solve.
    guard = cv2.dilate(view.mask.astype('u1'), np.ones((17, 17), 'u1'))
    mask = np.uint8((guard == 0) & (view.depth > 0)) * 255
    grey = cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY)
    # SIFT samples a scale-dependent neighbourhood around each feature. A mask
    # passed to OpenCV constrains its centre only. Erase the labelled interior
    # before building the scale space and exclude descriptors touching the hole.
    grey[view.mask] = np.median(grey[guard == 0]) if (guard == 0).any() else 0
    clearance = cv2.distanceTransform((~view.mask).astype('u1'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    extractor = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.018)
    keypoints, des = extractor.detectAndCompute(grey, mask)
    indices = [
        i for i, k in enumerate(keypoints) if clearance[round(k.pt[1]), round(k.pt[0])] > 6 * k.size + 2
    ]
    if des is None or len(indices) < 10:
        raise ValueError('Insufficient features outside pit in ' + view.key)
    uv = np.array([keypoints[i].pt for i in indices])
    u, v = np.rint(uv).astype(int).T
    z = view.depth[v, u]
    return uv, view.xyz(uv, z), des[indices]


def rigid(source, target):
    c, d = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - c).T @ (target - d))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = d - rotation @ c
    return result


def project(xyz, k):
    p = xyz @ k.T
    return p[..., :2] / p[..., 2:]


def transform(xyz, matrix):
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def align(view, ref, features_view, features_ref):
    uv, xyz, desc = features_view
    rp, rx, rd = features_ref
    matcher = cv2.BFMatcher()

    def neighbors(a, b):
        return {
            m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(a, b, k=2) if m.distance < 0.79 * n.distance
        }

    forward = neighbors(desc, rd)
    back = neighbors(rd, desc)
    pairs = [(i, j) for i, j in forward.items() if back.get(j) == i]
    if len(pairs) < 12:
        raise ValueError(f'{view.key} to {ref.key}: only {len(pairs)} mutual road features')
    i, j = np.transpose(pairs)
    src, dst = xyz[i], rx[j]
    # PnP rejects chance asphalt matches before the two measured point sets fix metric scale.
    cv2.setRNGSeed(31415)
    ok, rv, tv, idx = cv2.solvePnPRansac(
        src.astype('f8'),
        rp[j],
        ref.K,
        None,
        iterationsCount=2400,
        reprojectionError=3.5,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or idx is None or len(idx) < 10:
        raise ValueError(f'{view.key} to {ref.key}: no consistent camera pose')
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(rv)[0]
    m[:3, 3] = tv.ravel()
    inliers = idx.ravel()
    for metric_limit, pixel_limit in [(0.065, 5.0), (0.035, 3.5), (0.025, 3.0)]:
        metric = np.linalg.norm(transform(src, m) - dst, axis=1)
        px = np.linalg.norm(project(transform(src, m), ref.K) - rp[j], axis=1)
        selected = inliers[(metric[inliers] < metric_limit) & (px[inliers] < pixel_limit)]
        if len(selected) < 10:
            raise ValueError(f'{view.key} to {ref.key}: inconsistent 3D observations')
        m = rigid(src[selected], dst[selected])
        inliers = selected
    px = np.linalg.norm(project(transform(src[inliers], m), ref.K) - rp[j[inliers]], axis=1)
    error = np.linalg.norm(transform(src[inliers], m) - dst[inliers], axis=1)
    hull = cv2.contourArea(cv2.convexHull(rp[j[inliers]].astype('f4')))
    if hull < 450 or np.median(px) > 3.0:
        raise ValueError('Insufficient spatial or pixel support for ' + view.key)
    return m, dict(
        matches=len(pairs),
        inliers=len(inliers),
        road_hull_px2=hull,
        median_3d_m=float(np.median(error)),
        median_px=float(np.median(px)),
    )


def registration(spec, views):
    ref = views[spec['anchor']]
    parents = spec.get('pose_parents', {})
    if any(k == ref.key or k not in views or p not in spec['fit'] for k, p in parents.items()):
        raise ValueError('Registration parents must be fitting views and cannot override the anchor')
    parent_features = {ref.key: features(ref)}
    required = set(parents.values())
    poses = {ref.key: np.eye(4)}
    report = {}
    while len(poses) < len(views):
        ready = [k for k in views if k not in poses and parents.get(k, ref.key) in poses]
        if not ready:
            raise ValueError('Cycle in the road-camera registration paths')
        for key in ready:
            parent = parents.get(key, ref.key)
            current = features(views[key])
            step, report[key] = align(views[key], views[parent], current, parent_features[parent])
            if key in required:
                parent_features[key] = current
            poses[key] = poses[parent] @ step
            if parents:
                report[key]['parent'] = parent
    return poses, report
