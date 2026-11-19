"""Export calibrated reconstruction videos, held-out metrics and a world-space splat."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from plyfile import PlyData, PlyElement
from create_scene import create_scene
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from street_gaussian.utils.loss_utils import ssim
from street_gaussian.utils.lpipsPyTorch import lpips


def export_splat(pc, camera, dest):
    pc.set_visibility(list(pc.model_name_id.keys()))
    pc.parse_camera(camera)
    xyz = pc.get_xyz.detach().cpu().numpy()
    features = pc.get_features.detach().cpu().numpy()
    opacity = torch.logit(pc.get_opacity.clamp(1e-6, 1-1e-6)).detach().cpu().numpy()
    scaling = pc.get_scaling.log().detach().cpu().numpy()
    rotation = pc.get_rotation.detach().cpu().numpy()
    dc = features[:, 0, :]
    rest = features[:, 1:, :].transpose(0, 2, 1).reshape(len(xyz), -1)
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    names += [f'f_dc_{i}' for i in range(3)]
    names += [f'f_rest_{i}' for i in range(rest.shape[1])]
    names += ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    values = np.concatenate([xyz, np.zeros_like(xyz), dc, rest, opacity, scaling, rotation], axis=1).astype(np.float32)
    vertices = np.empty(len(values), dtype=[(n, 'f4') for n in names])
    for i, name in enumerate(names):
        vertices[name] = values[:, i]
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(dest)
    return len(xyz)


@torch.no_grad()
def main():
    torch.cuda.reset_peak_memory_stats()
    scene = create_scene()
    renderer = StreetGaussianRenderer()
    scene_id = Path(cfg.source_path).name
    label = 'A' if scene_id == '056' else 'B'
    dest = Path(__file__).resolve().parents[2] / 'outputs' / f'scene_{label}_{scene_id}'
    dest.mkdir(parents=True, exist_ok=True)
    cameras = sorted(scene.getTrainCameras()+scene.getTestCameras(), key=lambda c:c.meta['frame'])
    metrics = []
    frame_dir = dest/'frames'
    frame_dir.mkdir(exist_ok=True)
    started = time.time()
    writer = imageio.get_writer(str(dest/'reconstruction.mp4'), fps=10, codec='libx264', quality=8, macro_block_size=2)
    compare = imageio.get_writer(str(dest/'input_vs_reconstruction.mp4'), fps=10, codec='libx264', quality=8, macro_block_size=2)
    for i, camera in enumerate(cameras):
        result = renderer.render(camera, scene.gaussians)
        rgb = result['rgb'].clamp(0, 1)
        gt = camera.original_image.to(rgb.device)
        rendered = (rgb.permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8)
        source = (gt.permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8)
        writer.append_data(rendered)
        compare.append_data(np.concatenate([source,rendered],axis=1))
        frame = camera.meta['frame']
        Image.fromarray(rendered).save(frame_dir/f'{frame:03d}.jpg',quality=95)
        if camera.meta['is_val']:
            mse=(rgb-gt).square().mean()
            metrics.append(dict(frame=frame,psnr=float(-10*torch.log10(mse)),
                ssim=float(ssim(rgb,gt)),lpips=float(lpips(rgb*2-1,gt*2-1))))
        if i in [0,25,40,50,79]:
            Image.fromarray(np.concatenate([source,rendered],axis=1)).save(dest/f'comparison_{frame:03d}.jpg',quality=95)
        if i%20==0:
            print('render',scene_id,i,flush=True)
    writer.close()
    compare.close()
    snapshot = cameras[40]
    count=export_splat(scene.gaussians,snapshot,dest/'scene_frame040.ply')
    novel_writer=imageio.get_writer(str(dest/'novel_view_0.5m.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2)
    for camera in sorted([c for c in scene.getNovelViewCameras() if c.meta['novel_view_id']==.5],key=lambda c:c.meta['frame']):
        rgb=renderer.render_novel_view(camera,scene.gaussians)['rgb'].clamp(0,1)
        novel_writer.append_data((rgb.permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8))
    novel_writer.close()
    summary=dict(scene=scene_id,stage='initial 3DGS reconstruction; diffusion not applied',
        iterations=7000,train_views=len(scene.getTrainCameras()),test_views=len(metrics),
        moving_objects=len(scene.dataset.metadata['obj_meta']),resolution=[cameras[0].image_width,cameras[0].image_height],
        lpips_input_range='[-1, 1]',
        gaussians_at_frame040=count,seconds=time.time()-started,
        peak_render_memory_MiB=torch.cuda.max_memory_allocated()/2**20,
        mean={key:float(np.mean([m[key] for m in metrics])) for key in ['psnr','ssim','lpips']},per_frame=metrics)
    (dest/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
