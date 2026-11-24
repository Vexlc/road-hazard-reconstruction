"""Render the road debris in independent 220-degree Cycles cameras, including material-normal AOVs."""

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from scipy.spatial.transform import Rotation

rig = import_module("render-cone-cycles")
ROOT = rig.ROOT
PAIRS = ("R242-R243", "R244-R245")
VIEWS = (
    ("front", 0, 30, [0, -1.55, 0.86]),
    ("edge", 85, 15, [0, -1.5, 0.76]),
    ("rear", 180, 58, [0, 0.70, 1.18]),
)


def normal_pass(objects, scene):
    view_layer = bpy.context.view_layer
    aov = view_layer.aovs.add()
    aov.name = "MaterialNormal"
    aov.type = "COLOR"
    file = scene.node_tree.nodes["Passes"]
    file.file_slots.new("normal-")
    view_layer.update()
    layers = next(n for n in scene.node_tree.nodes if n.type == "R_LAYERS")
    scene.node_tree.links.new(layers.outputs[aov.name], file.inputs[-1])
    materials = {s.material for o in objects for s in o.material_slots}
    for mat in materials:
        graph = mat.node_tree
        bsdf = next(n for n in graph.nodes if n.type == "BSDF_PRINCIPLED")
        if not bsdf.inputs["Normal"].is_linked:
            raise ValueError("Expected the independent glTF import to wire the source normal map")
        normal = bsdf.inputs["Normal"].links[0].from_socket
        output = graph.nodes.new("ShaderNodeOutputAOV")
        output.aov_name = aov.name
        encode = graph.nodes.new("ShaderNodeVectorMath")
        encode.operation = "MULTIPLY_ADD"
        encode.inputs[1].default_value = (0.5, 0.5, 0.5)
        encode.inputs[2].default_value = (0.5, 0.5, 0.5)
        graph.links.new(normal, encode.inputs[0])
        graph.links.new(encode.outputs[0], output.inputs["Color"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/road-debris")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/road-debris/cycles")
    a = p.parse_args()
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("zyx", [33, 3, -4], degrees=True).as_matrix()
    world[:3, 3] = [5.7, -3.3, 1.25]
    for pair in PAIRS:
        scene, camera = rig.setup_scene(64)
        rig.plane(scene, world)
        # A small raised host marker tests which object's visible surface is actually nearest.
        bpy.ops.mesh.primitive_cube_add(size=1)
        marker = bpy.context.object
        transform = np.diag([0.18, 0.11, 0.13, 1.0])
        transform[:3, 3] = [0.30, -0.36, 0.065]
        marker.matrix_world = Matrix((world @ transform).tolist())
        marker.pass_index = 2
        objects = rig.imported(a.assets / pair / "asset.glb", world, 3)
        for obj in objects:
            obj.pass_index = 3 if "@tyre" in obj.name or "@crate" in obj.name else 4
        normal_pass(objects, scene)
        for view, yaw, pitch, eye in VIEWS:
            for show in (False, True):
                rig.render(
                    scene,
                    camera,
                    world,
                    objects,
                    a.out.resolve() / pair,
                    view,
                    yaw,
                    show,
                    eye=eye,
                    pitch_deg=pitch,
                )
                if show:
                    folder = a.out / pair / view / "native"
                    img = bpy.data.images.load(str((folder / "normal-0001.exr").resolve()))
                    rgba = np.empty(len(img.pixels), "f4")
                    img.pixels.foreach_get(rgba)
                    normal = rgba.reshape(rig.SIZE, rig.SIZE, 4)[::-1, :, :3] * 2 - 1
                    np.save(folder / "shading-world.npy", normal)
                    bpy.data.images.remove(img)
    bpy.ops.wm.read_factory_settings(use_empty=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
