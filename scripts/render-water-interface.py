"""Measure single-interface reflectance with native Cycles glass, a black bed and a split-color sky."""

import argparse
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


def main(out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 256
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = 1843
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.render.resolution_x = scene.render.resolution_y = 20
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.world = bpy.data.worlds.new("Two directional reference colors")
    scene.world.use_nodes = True
    tree = scene.world.node_tree
    coords = tree.nodes.new("ShaderNodeTexCoord")
    axis = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coords.outputs["Generated"], axis.inputs[0])
    choose = tree.nodes.new("ShaderNodeMath")
    choose.operation = "GREATER_THAN"
    choose.inputs[1].default_value = 0.12
    tree.links.new(axis.outputs["X"], choose.inputs[0])
    tint = tree.nodes.new("ShaderNodeMixRGB")
    tree.links.new(choose.outputs[0], tint.inputs[0])
    tint.inputs[1].default_value = (0.12, 0.42, 0.63, 1)
    tint.inputs[2].default_value = (0.70, 0.26, 0.11, 1)
    tree.links.new(tint.outputs[0], tree.nodes["Background"].inputs["Color"])
    bpy.ops.mesh.primitive_plane_add(size=200)
    water = bpy.context.object
    material = bpy.data.materials.new("Pure water, smooth boundary")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = 1.333
    glass.inputs["Roughness"].default_value = 0.0
    material.node_tree.links.new(glass.outputs[0], nodes.get("Material Output").inputs[0])
    water.data.materials.append(material)
    bpy.ops.mesh.primitive_plane_add(size=300, location=(0, 0, -0.2))
    bed = bpy.context.object
    absorb = bpy.data.materials.new("Black absorbing substrate")
    absorb.use_nodes = True
    absorb.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0, 0, 0, 1)
    absorb.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0
    bed.data.materials.append(absorb)
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    scene.camera = cam
    cam.data.type, cam.data.ortho_scale = "ORTHO", 0.12
    out.mkdir(parents=True, exist_ok=True)

    def render(name, direction, position):
        cam.location = position.tolist()
        cam.rotation_euler = Vector(direction.tolist()).to_track_quat("-Z", "Y").to_euler()
        target = out / (name + ".exr")
        scene.render.filepath = str(target.resolve())
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(target.resolve()), check_existing=False)
        values = np.empty(len(img.pixels), "f4")
        img.pixels.foreach_get(values)
        bpy.data.images.remove(img)
        return values.reshape(20, 20, 4)[2:-2, 2:-2, :3].mean(axis=(0, 1)).tolist()

    rows = []
    for angle in [0, -45, 55, -72, 82]:
        a = np.deg2rad(angle)
        direction = np.array([np.sin(a), 0.0, -np.cos(a)])
        water_color = render(f"water-{angle}", direction, -direction * 3.0)
        mirror = direction * [1.0, 1.0, -1.0]
        water.hide_render = bed.hide_render = True
        sky_color = render(f"sky-{angle}", mirror, np.array([0.0, 0.0, 1.0]))
        water.hide_render = bed.hide_render = False
        rows.append(
            dict(
                incident=direction.tolist(),
                sky_direction=mirror.tolist(),
                angle_deg=angle,
                water_linear=water_color,
                sky_linear=sky_color,
            )
        )
    (out / "native.json").write_text(json.dumps(rows, indent=2) + "\n")
    bpy.ops.wm.read_factory_settings(use_empty=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    main(p.parse_args().out)
    sys.stdout.flush()
    os._exit(0)
