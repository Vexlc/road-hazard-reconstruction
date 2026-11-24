"""Render independent fisheye RGB-D observations and held-out Cycles dielectric reflections."""

import argparse
from importlib import import_module
import os
from pathlib import Path
import sys

import bpy
import numpy as np

rig = import_module("render-cone-cycles")


def material(obj, name, kind, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.new(kind)
    node.inputs["Color"].default_value = (*color, 1)
    if kind == "ShaderNodeBsdfGlass":
        node.inputs["IOR"].default_value = 1.333
        node.inputs["Roughness"].default_value = 0.0
    mat.node_tree.links.new(node.outputs[0], mat.node_tree.nodes["Material Output"].inputs[0])
    obj.data.materials.append(mat)


def main(out):
    scene, camera = rig.setup_scene(64)
    scene.cycles.use_adaptive_sampling = False
    scene.render.dither_intensity = 0.0
    scene.world = bpy.data.worlds.new("Dark environment")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -0.08))
    bed = bpy.context.object
    bed.pass_index = 1
    material(bed, "Absorbing bed", "ShaderNodeEmission", (0, 0, 0))
    bpy.ops.mesh.primitive_plane_add(size=40)
    water = bpy.context.object
    water.pass_index = 1
    material(water, "Smooth air to water", "ShaderNodeBsdfGlass", (1, 1, 1))
    for idx, name, center, scale, color in [
        (2, "Near red screen", (-0.75, 2.15, 0.88), (0.98, 0.12, 1.65), (0.82, 0.035, 0.01)),
        (3, "Far green screen", (0.05, 3.6, 0.73), (3.7, 0.12, 1.36), (0.013, 0.72, 0.04)),
    ]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        obj = bpy.context.object
        obj.dimensions = scale
        obj.pass_index = idx
        material(obj, name, "ShaderNodeEmission", color)
    out = out.resolve()
    rig.SIZE = 1024
    scene.render.resolution_x = scene.render.resolution_y = rig.SIZE
    world = np.eye(4)
    for name, yaw, eye in [
        ("survey-left", 18, [-2, -0.8, 1.55]),
        ("survey-right", -22, [1.85, -0.6, 1.5]),
        ("survey-rear", 175, [0.8, 6.4, 1.5]),
        ("survey-between", 180, [-0.8, 3.03, 0.75]),
        ("survey-far", 0, [0.08, 2.70, 0.8]),
        ("survey-side", 90, [-3.5, 2.95, 0.70]),
        ("survey-low", 125, [-2.25, 4.65, 0.21]),
    ]:
        rig.render(scene, camera, world, [water], out, name, yaw, False, eye=eye, pitch_deg=8)
    scene.cycles.samples = 512
    scene.cycles.filter_width = 0.35
    rig.SIZE = 512
    scene.render.resolution_x = scene.render.resolution_y = rig.SIZE
    for name, yaw, eye in [
        ("front", 3, [0.2, -2.4, 1.24]),
        ("edge", 85, [0.8, -1.8, 1.15]),
        ("rear", 124, [-3.8, 4.85, 1.25]),
    ]:
        for show in (False, True):
            rig.render(scene, camera, world, [water], out, name, yaw, show, eye=eye, pitch_deg=21)
    bpy.ops.wm.read_factory_settings(use_empty=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    main(parser.parse_args().out)
    sys.stdout.flush()
    os._exit(0)
