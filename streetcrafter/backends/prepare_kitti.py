"""Build calibrated equidistant and perspective views of KITTI-360 image_02."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from fisheye_bundle import FisheyeBundle, held_out


def homogeneous(row):
    return np.vstack((np.reshape(row, (3, 4)), [0., 0., 0., 1.]))


def project_mei(xyz, calib):
    xi = calib['mirror_parameters']['xi']
    dist = calib['distortion_parameters']
    proj = calib['projection_parameters']
    denom = xyz[..., 2] + xi * np.linalg.norm(xyz, axis=-1)
    x, y = np.moveaxis(xyz[..., :2] / np.maximum(denom[..., None], 1e-10), -1, 0)
    r2 = x*x + y*y
    radial = 1 + dist['k1']*r2 + dist['k2']*r2*r2
    u = x*radial + 2*dist['p1']*x*y + dist['p2']*(r2 + 2*x*x)
    v = y*radial + 2*dist['p2']*x*y + dist['p1']*(r2 + 2*y*y)
    return np.stack((proj['gamma1']*u + proj['u0'], proj['gamma2']*v + proj['v0']), axis=-1)


def remap_grid(size, f, fish, calib):
    w, h = size
    y, x = np.mgrid[:h, :w]
    x, y = (x + .5 - w/2)/f, (y + .5 - h/2)/f
    if fish:
        theta = np.hypot(x, y)
        scale = np.sinc(theta / np.pi)
        rays = np.stack((x*scale, y*scale, np.cos(theta)), axis=-1)
    else:
        rays = np.stack((x, y, np.ones_like(x)), axis=-1)
    grid = project_mei(rays, calib)
    if np.any(grid < 0) or np.any(grid >= [calib['image_width']-1, calib['image_height']-1]):
        raise ValueError('Requested view extends beyond the calibrated image')
    return grid.astype('float32')


def prepare(raw, dest, camera='image_02', start=200, count=48, drive='0000'):
    calib = yaml.safe_load((raw / 'calibration' / (camera + '.yaml')).read_text().split('\n', 1)[1])
    rig = {}
    for line in (raw / 'calibration/calib_cam_to_pose.txt').read_text().splitlines():
        name, numbers = line.split(':')
        rig[name] = homogeneous(np.fromstring(numbers, sep=' '))
    velo_to_cam = np.linalg.inv(rig[camera]) @ rig['image_00'] @ np.linalg.inv(
        homogeneous(np.loadtxt(raw / 'calibration/calib_cam_to_velo.txt')))
    poses = np.loadtxt(raw / 'poses.txt')
    frame_ids = np.arange(start, start + count)
    timestamps = np.loadtxt(raw / camera / 'timestamps.txt', delimiter='\t', dtype=str)
    clock = np.char.replace(timestamps[frame_ids], ' ', 'T').astype('datetime64[ns]')
    seconds = (clock - clock[0]) / np.timedelta64(1, 's')
    if np.any(np.diff(seconds) <= 0):
        raise ValueError('Camera timestamps must increase strictly')
    xyz = np.column_stack([np.interp(frame_ids, poses[:, 0], poses[:, i]) for i in (4, 8, 12)])
    rot = Slerp(poses[:, 0], Rotation.from_matrix(poses[:, 1:].reshape(-1, 3, 4)[:, :, :3]))(frame_ids)
    xyz -= xyz[0].copy()
    pin_size, fish_size = (640, 400), (512, 512)
    pin_f = pin_size[0] / (2*np.tan(np.deg2rad(46)))
    fish_f = fish_size[0] / np.deg2rad(125)
    variants = [('all', pin_size, pin_f, False), ('fish_all', fish_size, fish_f, True)]
    grids = {}
    cameras = {}
    for part, size, f, fish in variants:
        grids[part] = remap_grid(size, f, fish, calib)
        width, height = size
        camera_row = (f'1 OPENCV_FISHEYE {width} {height} {f} {f} {width/2} {height/2} 0 0 0 0'
                  if fish else f'1 PINHOLE {width} {height} {f} {f} {width/2} {height/2}')
        cameras[part] = camera_row
    meta = dict(scene=f'kitti360-{drive}', label=f'KITTI-360 {drive} · ' + ('左侧鱼眼' if camera == 'image_02' else '右侧鱼眼'),
                raw_pattern=raw.name + '/' + camera + '/{frame:010d}.png', raw_size=[1400,1400], camera_model='PINHOLE', resolution=pin_size,
                intrinsics=[pin_f, pin_f, pin_size[0]/2, pin_size[1]/2], train=[], test=[], times={},
                source_frames={}, source_camera=camera+' MEI', fisheye_horizontal_fov=125,
                perspective_horizontal_fov=92, pose_interpolated=[])
    bundle = FisheyeBundle(dest, cameras, meta)
    cloud, rng = [], np.random.default_rng(42)
    for i, frame in enumerate(frame_ids):
        if frame not in poses[:, 0]:
            meta['pose_interpolated'].append(int(frame))
        pose = homogeneous(np.column_stack((rot[i].as_matrix(), xyz[i]))) @ rig[camera]
        img = cv2.imread(str(raw / camera / f'{frame:010d}.png'))
        if img is None or img.shape[:2] != (calib['image_height'], calib['image_width']):
            raise ValueError(f'Missing or incompatible raw image {frame}')
        views = {part: cv2.remap(img, grid, None, cv2.INTER_LINEAR) for part, grid in grids.items()}
        bundle.frame(i, frame, seconds[i], pose, views)
        if held_out(i):
            continue
        scan = np.fromfile(raw / f'velodyne/{frame:010d}.bin', '<f4').reshape(-1, 4)[:, :3]
        cam = scan @ velo_to_cam[:3, :3].T + velo_to_cam[:3, 3]
        cam = cam[(cam[:, 2] > .7) & (np.linalg.norm(cam, axis=1) < 90)]
        uv = np.rint(project_mei(cam, calib)).astype(int)
        good = np.flatnonzero(((uv >= 0) & (uv < list(img.shape[1::-1]))).all(axis=1))
        idx = rng.choice(good, min(4000, len(good)), replace=False)
        u, v = uv[idx].T
        world = cam[idx] @ pose[:3, :3].T + pose[:3, 3]
        cloud.append(np.column_stack((world, img[v, u, ::-1])))
    bundle.finish(cloud)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--drive', default='0000')
    parser.add_argument('--camera', choices=['image_02', 'image_03'], default='image_02')
    parser.add_argument('--start', type=int, default=200)
    parser.add_argument('--count', type=int, default=48)
    args = parser.parse_args()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.destination.parent) as tmp:
        stage = Path(tmp) / 'scene'
        prepare(args.raw, stage, args.camera, args.start, args.count, args.drive)
        stage.rename(args.destination)
