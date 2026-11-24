"""Run the upstream static SC-GS trainer with disjoint views and saved test renders."""

import json
from pathlib import Path
import runpy
import sys

from torchvision.utils import save_image

sys.path.insert(0, str(Path.cwd()))

import train
from scene.dataset_readers import getNerfppNorm, readColmapSceneInfo, sceneLoadTypeCallbacks


def read_split(path, images, evaluate):
    if not evaluate:
        raise ValueError('Pass --eval to use the shared split')
    info = readColmapSceneInfo(path, images, False)
    split = json.loads((Path(path) / 'split.json').read_text())
    by_name = {cam.image_name + '.png': cam for cam in info.train_cameras}
    if set(split['train']) & set(split['test']) or set(by_name) != set(split['train'] + split['test']):
        raise ValueError('Invalid train/test split')
    training = [by_name[name] for name in split['train']]
    return info._replace(train_cameras=training, test_cameras=[by_name[name] for name in split['test']],
                         nerf_normalization=getNerfppNorm(training))


original_report = train.training_report


def training_report(writer, step, absolute_loss, loss, loss_fn, elapsed, test_steps, scene, render, *args):
    if step in test_steps:
        output = Path(scene.model_path) / f'heldout_{step}'
        output.mkdir(exist_ok=True)
        heldout = {camera.image_name for camera in scene.getTestCameras()}

        def capture(camera, *values, **options):
            result = render(camera, *values, **options)
            if camera.image_name in heldout:
                save_image(result['render'].clamp(0, 1), output / (camera.image_name + '.png'))
            return result
    else:
        capture = render
    return original_report(writer, step, absolute_loss, loss, loss_fn, elapsed, test_steps, scene, capture, *args)


if __name__ == '__main__':
    sceneLoadTypeCallbacks['Colmap'] = read_split
    train.training_report = training_report
    runpy.run_path('train_gui.py', run_name='__main__')
