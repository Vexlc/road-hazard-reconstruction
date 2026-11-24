"""Score saved PNGs on identical test views; distinguish full lens and shared perspective."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData


def rgb(path, size):
    pixels = cv2.imread(str(path))
    if pixels is None or pixels.shape[1::-1] != tuple(size):
        raise ValueError(f'Missing render or unexpected resolution: {path}')
    return pixels


def psnr(reference, prediction):
    mse = np.mean((reference.astype(np.float64) - prediction) ** 2)
    return -10 * np.log10(max(mse / 255**2, 1e-15))


def panel(images, name):
    tiles = []
    for title, image in images.items():
        tile = cv2.copyMakeBorder(image, 33, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
        cv2.putText(tile, f'{name}   {title}', (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 245, 245), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def evaluate(scene, runs, steps):
    split = json.loads((scene / 'split.json').read_text())
    size = split['resolution']
    params = split['params_fisheye']
    matrix = np.array([[params[0], 0, params[2] - 0.5], [0, params[1], params[3] - 0.5], [0, 0, 1]])
    maps = cv2.fisheye.initUndistortRectifyMap(matrix, np.array(params[4:]), np.eye(3), matrix, tuple(size), cv2.CV_32FC1)
    shared_mask = np.logical_and.reduce([(coord >= 0) & (coord <= dim - 1) for coord, dim in zip(maps, size)])
    if not shared_mask.any():
        raise ValueError('Perspective and fisheye images have no shared pixels')
    stages = {'3dgut': steps}
    if (runs / '3dgut-7000').is_dir():
        stages['3dgut-7000'] = 7000
    folders = {}
    for method, step in stages.items():
        candidates = list((runs / method).glob(f'fit/*/ours_{step}/renders'))
        if len(candidates) != 1:
            raise ValueError(f'Expected one completed {method} render set: {candidates}')
        folders[method] = candidates[0]
    scgs = runs / 'sc-gs/fit_node'
    self_cali = runs / 'self-cali-gs'
    models = {method: runs / method / 'scene.ply' for method in stages}
    for method, root in [('sc-gs', scgs), ('self-cali-gs', self_cali)]:
        models[method] = root / f'point_cloud/iteration_{steps}/point_cloud.ply'
    counts = {}
    for method, model in models.items():
        vertices = PlyData.read(model)['vertex']
        if not len(vertices) or not all(np.isfinite(vertices[key]).all() for key in vertices.data.dtype.names):
            raise ValueError(f'Invalid Gaussian parameters: {model}')
        counts[method] = len(vertices)

    results = []
    previews = {'full-fisheye': [], 'shared-perspective': []}
    for idx, name in enumerate(split['test']):
        source = split['sources'][name]
        gt = rgb(scene / 'fish/images' / name, size)
        pin_gt = rgb(scene / 'images' / name, size)
        fisheyes = {method: rgb(folder / f'{idx:05d}.png', size) for method, folder in folders.items()}
        validation = self_cali / f'training_val_{steps}'
        suffix = f'{Path(name).stem}_test.png'
        if not np.array_equal(rgb(validation / 'gt' / suffix, size), gt):
            raise ValueError(f'Self-Cali-GS target differs from raw test view: {name}')
        fisheyes['self-cali-gs'] = rgb(validation / 'renderred' / ('distorted_' + suffix), size)
        shared = {method: cv2.remap(image, *maps, cv2.INTER_LINEAR) for method, image in fisheyes.items()}
        shared['sc-gs'] = rgb(scgs / f'heldout_{steps}' / name, size)
        row = {'source': source}
        for domain, target, images in [('full-fisheye', gt, fisheyes), ('shared-perspective', pin_gt, shared)]:
            valid = shared_mask if domain == 'shared-perspective' else slice(None)
            row[domain] = {method: psnr(target[valid], image[valid]) for method, image in images.items()}
            if idx in (0, len(split['test']) // 2, len(split['test']) - 1):
                previews[domain].append(panel({'GT': target, **images}, Path(source).stem))
        results.append(row)
    means = {}
    for domain, rows in previews.items():
        if not cv2.imwrite(str(runs / f'{domain}.jpg'), np.vstack(rows)):
            raise OSError(f'Cannot write {domain} preview')
        means[domain] = {key: float(np.mean([row[domain][key] for row in results])) for key in results[0][domain]}
    report = {'scene': scene.name, 'resolution': size, 'train': len(split['train']), 'test': len(results),
              'seed_points': split['seed_points'], 'gaussians': counts, 'steps': {**stages, 'sc-gs': steps, 'self-cali-gs': steps},
              'shared_valid_fraction': float(shared_mask.mean()), 'mean_psnr_db': means, 'frames': results}
    (runs / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--steps', type=int, default=1500)
    args = parser.parse_args()
    evaluate(args.scene, args.runs, args.steps)
