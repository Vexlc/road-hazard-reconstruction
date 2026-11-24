"""Replay a trained 3DGUT model on the complete calibrated driving trajectory."""
import argparse
from pathlib import Path

from PIL import Image
import torch
from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.render import apply_background


@torch.no_grad()
def replay(checkpoint, source, output, offset_right=0.):
    state = torch.load(checkpoint, map_location='cuda', weights_only=False)
    config = state['config']
    if config.dataset.normalize_world_space or config.post_processing.method is not None:
        raise ValueError('Driving replay expects the original world space and no postprocessing')
    model = MixtureOfGaussians(config)
    model.init_from_checkpoint(state, setup_optimizer=False)
    model.eval()
    model.build_acc()
    ds = ColmapDataset(str(source), test_split_interval=0)
    output.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(ds.image_paths):
        batch = ds.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([ds[i]]))
        if offset_right:
            shifted = batch.T_to_world.clone()
            shifted[:, :3, 3] += offset_right * shifted[:, :3, 0]
            batch.T_to_world = shifted
        rgb = apply_background(model.background, model(batch), batch)['pred_features'][0]
        if not torch.isfinite(rgb).all():
            raise ValueError(f'Invalid render for {path}')
        Image.fromarray(rgb.clamp(0, 1).mul(255).round().byte().cpu().numpy()).save(output / Path(path).name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint', 'source', 'output']:
        parser.add_argument(key, type=Path)
    parser.add_argument('--offset-right', type=float, default=0., help='Offset along the camera x axis in metres')
    replay(**vars(parser.parse_args()))
