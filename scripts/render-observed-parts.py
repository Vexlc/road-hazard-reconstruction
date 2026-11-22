"""Render the textured mesh, labelled mesh and selected surfaces with one inspection camera."""

import argparse
from pathlib import Path
import os
import sys

import bpy
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gen2hole/recipes' / n) for n in ('common', 'series')]
from render_mesh import prepare_view


def render(run, key):
    b = run / 'series'
    asset = b / 'assets/pothole' / key
    scene, _ = prepare_view(asset, b / 'mesh_cache', 'inspection', 800, 550)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    folder = run / 'evaluation' / key
    for name, path in [
        ('textured', asset / 'asset.glb'),
        ('labels', b / 'comparison' / key / 'labels.glb'),
        ('isolated', b / 'comparison' / key / 'isolated.glb'),
    ]:
        for obj in list(scene.objects):
            if obj.type == 'MESH':
                bpy.data.objects.remove(obj, do_unlink=True)
        bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
        file = folder / (name + '-render.png')
        scene.render.filepath = str(file.resolve())
        bpy.ops.render.render(write_still=True)
        with Image.open(file) as image:
            background = Image.new('RGB', image.size, (225, 230, 221))
            background.paste(image, mask=image.getchannel('A'))
            background.save(folder / (name + '-render.jpg'), quality=96)
        file.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('keys', nargs='+')
    args = p.parse_args()
    for key in args.keys:
        render(args.run, key)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
