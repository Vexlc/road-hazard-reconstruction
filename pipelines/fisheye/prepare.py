"""Prepare matching fisheye/perspective COLMAP views and train-only seed colors."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pycolmap

from download import REVISION


def make_models(raw, destination, views, width):
    source = pycolmap.Reconstruction(raw / 'colmap/sparse/0')
    if source.num_cameras() != 1:
        raise ValueError('Expected one camera per Nerfstudio scene')
    camera = next(iter(source.cameras.values()))
    if camera.model.name != 'OPENCV_FISHEYE':
        raise ValueError(f'Not a fisheye scene: {camera.model.name}')
    files = sorted(path for path in (raw / 'images_4').glob('*.*') if source.find_image_with_name(path.name))
    if not 16 <= views <= len(files):
        raise ValueError(f'Need 16..{len(files)} registered views')
    files = [files[idx] for idx in np.linspace(0, len(files) - 1, views, dtype=int)]
    original_size = np.array([camera.width, camera.height])
    size = (width, round(width * camera.height / camera.width))
    scale = np.array(size) / original_size
    camera.rescale(*size)
    intrinsics = camera.calibration_matrix().copy()
    intrinsics[:2, 2] -= 0.5
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(intrinsics, camera.params[4:], np.eye(3), intrinsics, size, cv2.CV_32FC1)
    tracks = defaultdict(list)
    colors = defaultdict(list)
    poses = []
    names = {}
    for index, path in enumerate(files):
        image_id = index + 1
        name = f'{index:04d}.png'
        names[name] = path.name
        image = source.find_image_with_name(path.name)
        pose = image.cam_from_world()
        poses.append(' '.join(map(str, [image_id, *np.roll(pose.rotation.quat, 1), *pose.translation, 1, name])))
        bgr = cv2.imread(str(path))
        if bgr is None or tuple(original_size // 4) != bgr.shape[1::-1]:
            raise ValueError(f'Unexpected image dimensions: {path}')
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        perspective = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR)
        for folder, pixels in [('fish', bgr), ('.', perspective)]:
            target = destination / folder / 'images' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), pixels):
                raise OSError(f'Cannot write {target}')
        if index % 8 == 0:
            continue
        features = image.get_observation_points2D()
        coords = np.array([feature.xy for feature in features]) * scale
        pinhole = cv2.fisheye.undistortPoints((coords - 0.5).reshape(-1, 1, 2), intrinsics, camera.params[4:], P=intrinsics).reshape(-1, 2) + 0.5
        good = np.all((pinhole >= 0) & (pinhole < size) & (coords >= 0) & (coords < size), axis=1)
        for feature_idx in np.flatnonzero(good):
            point_id = features[feature_idx].point3D_id
            fish_xy = coords[feature_idx]
            pixel_x, pixel_y = fish_xy.astype(int)
            tracks[point_id].append((image_id, fish_xy, pinhole[feature_idx]))
            colors[point_id].append(bgr[pixel_y, pixel_x, ::-1])

    indices = defaultdict(list)
    points = []
    for point_id, observations in tracks.items():
        if len(observations) < 2:
            continue
        point = source.point3D(point_id)
        rgb = np.rint(np.mean(colors[point_id], axis=0)).astype(int)
        track = []
        for image_id, fish_xy, pin_xy in observations:
            track.extend((image_id, len(indices[image_id])))
            indices[image_id].append((point_id, fish_xy, pin_xy))
        points.append(' '.join(map(str, [point_id, *point.xyz, *rgb, 0, *track])))
    if not points:
        raise ValueError('No multi-view training tracks')
    for fish in [True, False]:
        model = destination / ('fish' if fish else '.') / 'sparse/0'
        model.mkdir(parents=True)
        params = camera.params if fish else camera.params[:4]
        model_name = camera.model.name if fish else 'PINHOLE'
        with tempfile.TemporaryDirectory() as tmp:
            text = Path(tmp)
            (text / 'cameras.txt').write_text(' '.join(map(str, [1, model_name, *size, *params])) + '\n')
            (text / 'points3D.txt').write_text('\n'.join(points) + '\n')
            with (text / 'images.txt').open('w') as stream:
                for image_id, pose_line in enumerate(poses, 1):
                    stream.write(pose_line + '\n')
                    entries = []
                    for point_id, fish_xy, pin_xy in indices[image_id]:
                        coord = fish_xy if fish else pin_xy
                        entries.append(f'{coord[0]:.5f} {coord[1]:.5f} {point_id}')
                    stream.write(' '.join(entries) + '\n')
            recon = pycolmap.Reconstruction(text)
            recon.update_point_3d_errors()
            recon.write_binary(model)
    split = {'revision': REVISION, 'scene': raw.name, 'resolution': size, 'params_fisheye': camera.params.tolist(),
             'sources': names, 'test': list(names)[::8], 'train': [name for idx, name in enumerate(names) if idx % 8], 'seed_points': len(points)}
    (destination / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
    print(f'{raw.name}: {len(split["train"])} train, {len(split["test"])} test, {len(points)} seeds, {size}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--views', type=int, default=64)
    parser.add_argument('--width', type=int, default=480)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error('Destination already exists; choose an unused output directory')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.destination.parent) as staging:
        output = Path(staging) / 'data'
        make_models(args.raw, output, args.views, args.width)
        output.rename(args.destination)
