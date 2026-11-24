"""Render an identical camera path with the two local, ABI-incompatible backends."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch


def trajectory(anchors, sources, count):
    times = np.array([int(Path(name).stem.split('_')[-1]) for name in sources], dtype=float)
    query = np.linspace(times[0], times[-1], count)
    path = np.tile(np.eye(4), (count, 1, 1))
    path[:, :3, :3] = Slerp(times, Rotation.from_matrix(anchors[:, :3, :3]))(query).as_matrix()
    for axis in range(3):
        path[:, axis, 3] = np.interp(query, times, anchors[:, axis, 3])
    return path.astype(np.float32)


def render_gut(scene, runs, work, split, count):
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.utils.render import apply_background

    checkpoint_path, = (runs / '3dgut-7000').glob('fit/*/ckpt_last.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    config = checkpoint['config']
    if config.dataset.normalize_world_space or config.post_processing.method is not None:
        raise ValueError('Expected the unnormalized, non-postprocessed fisheye baseline')
    model = MixtureOfGaussians(config)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.eval()
    model.build_acc()
    dataset = ColmapDataset(str(scene / 'fish'), test_split_interval=0)
    names = [Path(path).name for path in dataset.image_paths]
    if names != sorted(split['sources']) or len(dataset.cam_intrinsics) != 1:
        raise ValueError('Unexpected pose order or camera count')
    poses = trajectory(dataset.get_poses(), [split['sources'][name] for name in names], count)
    np.save(work / 'camera-path.npy', poses)
    batch = dataset.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([dataset[0]]))
    reference = checkpoint_path.parent / 'ours_7000/renders/00000.png'
    for index, pose in enumerate(poses):
        batch.T_to_world = torch.from_numpy(pose[None]).cuda()
        output = apply_background(model.background, model(batch), batch)
        save_frame(output['pred_features'][0], work / 'gut', index, reference)


def render_sc(runs, work, split, count):
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel

    root = runs / 'sc-gs/fit_node'
    if 'is_scene_static=True' not in (root / 'cfg_args').read_text():
        raise ValueError('Zero deformation is valid only for the static SC-GS branch')
    model = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    model.load_ply(str(root / 'point_cloud/iteration_1500/point_cloud.ply'))
    model.active_sh_degree = 1
    width, height = split['resolution']
    focal_x, focal_y = split['params_fisheye'][:2]
    camera = Camera(1, np.eye(3), np.zeros(3), 2 * np.arctan(width / (2 * focal_x)),
                    2 * np.arctan(height / (2 * focal_y)), torch.zeros(3, height, width),
                    None, 'path', 0, fid=0.)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, device='cuda')
    poses = np.load(work / 'camera-path.npy')
    if len(poses) != count:
        raise ValueError('Path length mismatch')
    for index, pose in enumerate(poses):
        extrinsic = np.linalg.inv(pose)
        camera.reset_extrinsic(extrinsic[:3, :3].T, extrinsic[:3, 3])
        result = render(camera, model, pipe, background, 0., None, 0.)
        save_frame(result['render'].permute(1, 2, 0), work / 'sc', index, root / 'heldout_1500/0000.png')


def save_frame(tensor, folder, index, reference):
    if not torch.isfinite(tensor).all():
        raise ValueError(f'Invalid rendering at frame {index}')
    pixels = tensor.clamp(0, 1).mul(255).add_(0.5).byte().cpu().numpy()[:, :, ::-1]
    if index == 0:
        folder.mkdir(parents=True, exist_ok=True)
        original = cv2.imread(str(reference))
        if original is None or original.shape != pixels.shape:
            raise ValueError(f'Invalid reference: {reference}')
        error = np.abs(original.astype(float) - pixels).mean()
        print(f'Anchor vs. saved evaluation render: MAE {error:.3f}/255', flush=True)
        if error > 1:
            raise ValueError('Camera or checkpoint disagrees with the held-out render')
    if not cv2.imwrite(str(folder / f'{index:04d}.png'), pixels):
        raise OSError(f'Cannot write frame {index}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=['3dgut', 'sc-gs'])
    parser.add_argument('scene', type=Path)
    parser.add_argument('runs', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--frames', type=int, default=288)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    split = json.loads((args.scene / 'split.json').read_text())
    with torch.no_grad():
        if args.backend == '3dgut':
            render_gut(args.scene, args.runs, args.work, split, args.frames)
        else:
            render_sc(args.runs, args.work, split, args.frames)
