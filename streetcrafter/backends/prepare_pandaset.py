"""Export the cameras selected by StreetCrafter and a training-only LiDAR seed."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from street_gaussian.config import cfg
from street_gaussian.datasets.pandaset_readers import readPandasetInfo


def seed_points(camera, source, rng):
    with np.load(source / 'lidar_forward/depth' / (camera.image_name + '.npz')) as depth:
        v, u = np.nonzero(depth['mask'])
        z = depth['value'].reshape(-1)
    valid = np.flatnonzero(np.isfinite(z) & (z > 1) & (z < 95))
    idx = rng.choice(valid, min(2200, len(valid)), replace=False)
    v, u, z = v[idx], u[idx], z[idx]
    # Sparse forward LiDAR leaves the sky unobserved. Add distant sky samples
    # from the training masks and RGB only, outside the road geometry.
    sv, su = np.nonzero(np.asarray(camera.guidance['sky_mask']))
    idx = rng.choice(len(su), min(120, len(su)), replace=False)
    u, v = np.r_[u, su[idx]], np.r_[v, sv[idx]]
    z = np.r_[z, np.full(len(idx), 85.)]
    rays = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(camera.K).T
    xyz = (rays * z[:, None] - camera.T) @ camera.R.T
    return np.column_stack((xyz, np.asarray(camera.image.convert('RGB'))[v, u]))


def export(destination):
    source = Path(cfg.source_path)
    scene = readPandasetInfo(str(source), split_train=cfg.data.split_train,
                             split_test=cfg.data.split_test)
    cams = sorted(scene.train_cameras + scene.test_cameras, key=lambda c: c.metadata['frame'])
    if len({c.metadata['cam'] for c in cams}) != 1:
        raise ValueError('The front-camera comparison requires a single camera')
    width = int(cfg.resolution)
    ref = cams[0]
    height = round(ref.height * width / ref.width)
    k = np.diag([width / ref.width, height / ref.height, 1.]) @ ref.K
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    meta = dict(train=[], test=[], times={}, resolution=[width, height],
                intrinsics=[fx, fy, cx, cy], camera_model='PINHOLE', scene=source.name)
    rng = np.random.default_rng(42)
    points, poses = [], {'all': [], 'train': []}
    for part in poses:
        (destination / part / 'images').mkdir(parents=True)
        sparse = destination / part / 'sparse/0'
        sparse.mkdir(parents=True)
        (sparse / 'cameras.txt').write_text(f'1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n')
    for i, cam in enumerate(cams):
        if not np.allclose(cam.K, ref.K) or cam.image.size != ref.image.size:
            raise ValueError('Camera calibration changed within the clip')
        name = f"{cam.metadata['frame']:04d}.png"
        split = 'test' if cam.metadata['is_val'] else 'train'
        meta[split].append(name)
        meta['times'][name] = cam.metadata['timestamp']
        q = Rotation.from_matrix(cam.R.T).as_quat()[[3, 0, 1, 2]]
        pose = ' '.join(map(str, [i + 1, *q, *cam.T, 1, name])) + '\n\n'
        poses['all'].append(pose)
        image = destination / 'all/images' / name
        cam.image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR).save(image)
        if split == 'train':
            os.link(image, destination / 'train/images' / name)
            poses['train'].append(pose)
            points.append(seed_points(cam, source, rng))
    points = np.concatenate(points)
    _, unique = np.unique(np.floor(points[:, :3] / .12), axis=0, return_index=True)
    points = points[unique]
    cloud = destination / 'all/sparse/0/points3D.txt'
    np.savetxt(cloud, np.column_stack((np.arange(1, len(points)+1), points)),
               fmt='%d %.7f %.7f %.7f %d %d %d 0')
    os.link(cloud, destination / 'train/sparse/0/points3D.txt')
    for part, lines in poses.items():
        (destination / part / 'sparse/0/images.txt').write_text(''.join(lines))
    meta['seed_points'] = len(points)
    (destination / 'all/split.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    target = Path(os.environ['SCENE_BUNDLE'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        staged = Path(tmp) / 'bundle'
        export(staged)
        staged.rename(target)
