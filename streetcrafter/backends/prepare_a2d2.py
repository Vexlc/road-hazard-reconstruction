"""Prepare the native A2D2 front-left lens and a registered SC-GS perspective."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from download_a2d2 import DRIVE, FRAMES
from fisheye_bundle import FisheyeBundle, held_out


def read_scan(path):
    with np.load(path) as scan:
        valid = scan['pcloud_attr.valid']
        forward, left, up = scan['pcloud_points'][valid].T
    cam = np.column_stack((-left, -up, forward))
    return cam[(forward > 2) & (np.linalg.norm(cam, axis=1) < 100)]


def motion(prev, curr, scan, k):
    """Track image features and lift the previous observations with metric LiDAR."""
    corners = cv2.goodFeaturesToTrack(prev, 2500, .008, 4).reshape(-1, 2)
    corners = corners[corners[:, 1] < .80*prev.shape[0]]  # exclude the vehicle hood
    uv = scan[:, :2] / scan[:, 2:] * [k[0, 0], k[1, 1]] + k[:2, 2]
    dist, idx = cKDTree(uv).query(corners)
    keep = dist <= 3.0
    a, depths = corners[keep], scan[idx[keep], 2]
    lk = dict(winSize=(25, 25), maxLevel=4)
    b, ok, _ = cv2.calcOpticalFlowPyrLK(prev, curr, a, None, **lk)
    back, rev, _ = cv2.calcOpticalFlowPyrLK(curr, prev, b, None, **lk)
    good = ok.ravel().astype(bool) & rev.ravel().astype(bool) & (np.linalg.norm(back-a, axis=1) < .8)
    pts = (np.column_stack((a, np.ones(len(a)))) @ np.linalg.inv(k).T) * depths[:, None]
    pts, b = pts[good], b[good]
    if len(pts) < 30:
        raise ValueError('Too few LiDAR-supported tracks to estimate camera motion')
    ok, rvec, tvec, ids = cv2.solvePnPRansac(pts, b, k, None, iterationsCount=350,
                      reprojectionError=1.7, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or len(ids) < 30:
        raise ValueError('Metric camera pose could not be verified')
    ids = ids.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(pts[ids], b[ids], k, None, rvec, tvec)
    pred = cv2.projectPoints(pts[ids], rvec, tvec, k, None)[0].reshape(-1, 2)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = cv2.Rodrigues(rvec)[0], tvec[:, 0]
    return transform, dict(inliers=len(ids), median_error_px=float(np.median(np.linalg.norm(pred-b[ids], axis=1))))


def prepare(raw, dest):
    calib = json.loads((raw/'cams_lidars.json').read_text())['cameras']['front_left']
    if calib['Lens'] != 'Fisheye':
        raise ValueError('Expected the original fisheye model')
    k, d = np.array(calib['CamMatrixOriginal']), np.array(calib['Distortion']).ravel()
    w, h = calib['Resolution']
    fw, fh = 768, round(h*768/w)
    pw, ph = 640, 400
    f = pw/(2*np.tan(np.deg2rad(46.)))
    pin_k = np.array([[f, 0, (pw-1)/2], [0, f, (ph-1)/2], [0, 0, 1.]])
    maps = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), pin_k, (pw, ph), cv2.CV_32FC1)
    if any(m.min() < 0 or m.max() > limit-1 for m, limit in zip(maps, (w, h))):
        raise ValueError('Rectified rays outside the sensor')
    # OpenCV calibration uses integer pixel centres. COLMAP uses half-integer centres.
    params = [k[0, 0]*fw/w, k[1, 1]*fh/h, (k[0, 2]+.5)*fw/w, (k[1, 2]+.5)*fh/h, *d]
    cameras = {'all': f'1 PINHOLE {pw} {ph} {f} {f} {pw/2} {ph/2}',
               'fish_all': ' '.join(map(str, [1, 'OPENCV_FISHEYE', fw, fh, *params]))}
    meta = dict(scene='a2d2-frontleft', label='Audi A2D2 · 左前鱼眼', raw_size=[w, h],
                raw_pattern=raw.name+'/{frame:09d}.png', camera_model='PINHOLE', resolution=[pw, ph],
                intrinsics=[f, f, pw/2, ph/2], fisheye_resolution=[fw, fh],
                fisheye_label='原镜头完整视场', perspective_horizontal_fov=92,
                source_camera='front_left', sequence=DRIVE, odometry={})
    bundle = FisheyeBundle(dest, cameras, meta)
    times = []
    poses, train_times = [], []
    last_scan = last_gray = None
    pose = np.eye(4)
    for i, frame in enumerate(FRAMES):
        stamp = json.loads((raw/f'{frame:09d}.json').read_text())
        if stamp['cam_name'] != 'front_left' or stamp['image_zoom'] != 1.0:
            raise ValueError(f'Unexpected camera or zoom at {frame}')
        times.append(stamp['cam_tstamp'] / 1e6)
        if held_out(i):
            continue
        gray = cv2.imread(str(raw/f'{frame:09d}.png'), cv2.IMREAD_GRAYSCALE)
        gray = cv2.remap(gray, *maps, cv2.INTER_LINEAR)
        if last_gray is not None:
            delta, stats = motion(last_gray, gray, last_scan, pin_k)
            pose = pose @ np.linalg.inv(delta)
            meta['odometry'][str(frame)] = stats
        poses.append(pose.copy())
        train_times.append(times[-1])
        last_scan, last_gray = read_scan(raw/f'{frame:09d}.npz'), gray
    poses = np.stack(poses)
    rotations = Slerp(train_times, Rotation.from_matrix(poses[:, :3, :3]))(times).as_matrix()
    centers = np.column_stack([np.interp(times, train_times, poses[:, a, 3]) for a in range(3)])
    meta['camera_path_m'] = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    cloud, rng = [], np.random.default_rng(42)
    for i, frame in enumerate(FRAMES):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotations[i], centers[i]
        img = cv2.imread(str(raw/f'{frame:09d}.png'))
        if img is None or img.shape[1::-1] != (w, h):
            raise ValueError(f'Unexpected image at {frame}')
        views = {'fish_all': cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA),
                 'all': cv2.remap(img, *maps, cv2.INTER_LINEAR)}
        bundle.frame(i, frame, times[i]-times[0], pose, views)
        if held_out(i):
            continue
        cam = read_scan(raw/f'{frame:09d}.npz')
        uv = cv2.fisheye.projectPoints(cam[:, None], np.zeros(3), np.zeros(3), k, d)[0][:, 0]
        uv = np.rint(uv).astype(int)
        good = np.flatnonzero(((uv >= 0) & (uv < [w, h])).all(axis=1))
        idx = rng.choice(good, min(4500, len(good)), replace=False)
        u, v = uv[idx].T
        world = cam[idx] @ rotations[i].T + centers[i]
        cloud.append(np.column_stack((world, img[v, u, ::-1])))
    bundle.finish(cloud)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.destination.parent) as tmp:
        stage = Path(tmp) / 'bundle'
        prepare(args.raw, stage)
        stage.rename(args.destination)
