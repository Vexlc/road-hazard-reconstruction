"""Render the exported crack GLBs under raking light with all texture and bump slots removed."""

import argparse
import math
import os
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gen2hole/recipes' / n) for n in ('common', 'series')]
from render_mesh import prepare_view


def render(run, key):
    bundle = run / 'series'
    scene, _ = prepare_view(bundle / 'assets/pothole' / key, bundle / 'mesh_cache', 'inspection', 840, 568)
    clay = bpy.data.materials.new('Neutral pavement geometry')
    clay.use_nodes = True
    shader = clay.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (0.41, 0.40, 0.37, 1)
    shader.inputs['Roughness'].default_value = 0.94
    for obj in scene.objects:
        if obj.type == 'MESH':
            for slot in obj.material_slots:
                slot.material = clay
        if obj.type == 'LIGHT':
            obj.hide_render = True
    sun = bpy.data.lights.new('Low sun', 'SUN')
    sun.energy = 3.4
    sun.angle = math.radians(3)
    lamp = bpy.data.objects.new('Low sun', sun)
    scene.collection.objects.link(lamp)
    lamp.rotation_euler = Vector((0.7, 0.57, -0.18)).to_track_quat('-Z', 'Y').to_euler()
    scene.world.node_tree.nodes.get('Background').inputs[1].default_value = 0.27
    dest = run / 'evaluation' / key / 'clay.jpg'
    dest.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(dest.resolve())
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('ids', nargs='+')
    a = p.parse_args()
    for key in a.ids:
        render(a.run, key)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
