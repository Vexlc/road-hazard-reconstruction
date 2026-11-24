"""Train and replay official dynamic SC-GS with the driving camera calibration."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from PIL import Image
from torchvision.utils import save_image

sys.path.insert(0, str(Path.cwd()))
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene, DeformModel
from scene.dataset_readers import getNerfppNorm, readColmapSceneInfo, sceneLoadTypeCallbacks
from utils import camera_utils


def install_reader(fisheye_training=False):
    calibration = {}
    original_load = camera_utils.loadCam

    def read(path, images, evaluate):
        if not evaluate:
            raise ValueError('Driving scenes require the explicit evaluation split')
        info = readColmapSceneInfo(path, images, False)
        meta = json.loads((Path(path) / 'split.json').read_text())
        calibration.update(meta)
        times = meta['times']
        first, last = min(times.values()), max(times.values())
        cameras = {c.image_name + '.png': c for c in info.train_cameras}
        if set(meta['train']) & set(meta['test']) or set(cameras) != set(meta['train'] + meta['test']):
            raise ValueError('Camera names disagree with the driving split')
        def timed(names):
            return [cameras[n]._replace(fid=(times[n] - first) / (last - first)) for n in names]
        train = timed(meta['train'])
        training_views = train
        if fisheye_training:
            from scgs_fisheye import tangent_views
            training_views = train + tangent_views(train, Path(path).parent / 'fish_all')
        return info._replace(train_cameras=training_views, test_cameras=timed(meta['test']),
                             nerf_normalization=getNerfppNorm(train))

    def calibrated_camera(*args, **kwargs):
        cam = original_load(*args, **kwargs)
        if 'views' in calibration:
            view = calibration['views'][cam.image_name + '.png']
            _, width, height, _, _, cx, cy = calibration['projections']['all'][view['projection']]
        else:
            width, height = calibration['resolution']
            cx, cy = calibration['intrinsics'][2:4]
        # The rasterizer maps NDC zero to the midpoint between the outer pixel centres.
        cam.projection_matrix[2, 0] = 2 * cx / width - 1
        cam.projection_matrix[2, 1] = 2 * cy / height - 1
        cam.full_proj_transform = cam.world_view_transform @ cam.projection_matrix
        return cam

    sceneLoadTypeCallbacks['Colmap'] = read
    camera_utils.loadCam = calibrated_camera


@torch.no_grad()
def replay():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    if dataset.is_scene_static or dataset.deform_type != 'node':
        raise ValueError('Replay requires a dynamic node checkpoint')
    deform = DeformModel(K=dataset.K, deform_type='node', is_blender=dataset.is_blender,
                         skinning=dataset.skinning, hyper_dim=dataset.hyper_dim, node_num=dataset.node_num,
                         pred_opacity=dataset.pred_opacity, pred_color=dataset.pred_color,
                         use_hash=dataset.use_hash, hash_time=dataset.hash_time,
                         d_rot_as_res=dataset.d_rot_as_res and not dataset.d_rot_as_rotmat,
                         local_frame=dataset.local_frame, progressive_brand_time=dataset.progressive_brand_time,
                         max_d_scale=dataset.max_d_scale)
    if not deform.load_weights(dataset.model_path, iteration=args.iteration):
        raise FileNotFoundError(f'Missing node checkpoint at iteration {args.iteration}')
    gaussians = GaussianModel(dataset.sh_degree, fea_dim=dataset.hyper_dim,
                              with_motion_mask=dataset.gs_with_motion_mask)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.active_sh_degree = min(dataset.sh_degree, args.iteration // 1000)
    background = torch.full((3,), float(dataset.white_background), device='cuda')
    args.frames.mkdir(parents=True, exist_ok=True)
    pipe = pipeline.extract(args)
    fish_file = Path(dataset.source_path).parent / 'fish_all/sparse/0/cameras.txt'
    fish = None
    if fish_file.exists():
        from scgs_fisheye import FisheyeReplay
        fish = FisheyeReplay(fish_file)
        fish_dir = args.frames.parent / 'fisheye_frames'
        fish_dir.mkdir(exist_ok=True)
    rig_fish = {}
    source = Path(dataset.source_path)
    rig = json.loads((source/'split.json').read_text())
    if 'views' in rig and any(c[0] == 'OPENCV_FISHEYE' for c in rig['projections']['gut_train'].values()):
        from scgs_fisheye import FisheyeReplay
        fish_dir = args.frames.parent/'sensor_frames'
        fish_dir.mkdir(exist_ok=True)
        for cid, (model,w,h,*params) in rig['projections']['gut_train'].items():
            if model == 'OPENCV_FISHEYE':
                _, pw, ph, fx, fy, *_ = rig['projections']['eval'][cid]
                rig_fish[cid] = FisheyeReplay(sensor=((w,h), params), central=(pw,ph,fx,fy))
    for cam in scene.getTrainCameras() + scene.getTestCameras():
        name = cam.image_name + '.png'
        view = rig.get('views', {}).get(name)
        if view and view['tile']:
            continue
        cam.load2device()
        d = deform.step(gaussians.get_xyz.detach(), deform.deform.expand_time(cam.fid),
                        feature=gaussians.feature, is_training=False, motion_mask=gaussians.motion_mask,
                        camera_center=cam.camera_center)
        def render_view(view):
            return render(view, gaussians, pipe, background, d['d_xyz'], d['d_rotation'],
                          d['d_scaling'], d_opacity=d['d_opacity'], d_color=d['d_color'],
                          d_rot_as_res=deform.d_rot_as_res)['render']
        rgb = render_view(cam)
        if not torch.isfinite(rgb).all() or (rgb.std() < 1/255 and cam.original_image.std() > 5/255):
            raise ValueError(f'Invalid render for {cam.image_name}')
        save_image(rgb.clamp(0, 1), args.frames / (cam.image_name + '.png'))
        if fish is not None:
            Image.fromarray(fish.frame(cam, render_view)).save(fish_dir / name)
        if rig_fish:
            sensor = rig_fish.get(view['projection'])
            if sensor:
                Image.fromarray(sensor.frame(cam, render_view)).save(fish_dir/name)
            elif not (fish_dir/name).exists():
                os.link(args.frames/name, fish_dir/name)
        if dataset.load2gpu_on_the_fly:
            cam.load2device('cpu')


if __name__ == '__main__':
    action = sys.argv.pop(1)
    install_reader(fisheye_training=action == 'train-fisheye')
    if action in ('train', 'train-fisheye'):
        runpy.run_path('train_gui.py', run_name='__main__')
    else:
        replay()
