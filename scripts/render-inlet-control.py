"""Render both mixed openings and a slender foreground target in independent Cycles fisheyes."""

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix
from scipy.spatial.transform import Rotation

rig = import_module("render-cone-cycles")
pavement = import_module("render-pavement-cycles")
cover = import_module("render-round-cover-cycles")


def build(scene, ground, recipe, models):
    road = rig.plane(scene, ground)
    material = road.data.materials[0]
    bpy.data.objects.remove(road, do_unlink=True)
    objects, plugs = [], []
    for index, row in enumerate(recipe["instances"], 1):
        path = models / row["asset"]
        if not (path / "asset.glb").is_file():
            (path,) = models.glob(f"*/series/assets/*/{row['asset']}")
        pose = np.eye(4)
        pose[:2, 3] = row["xy_m"]
        pose[:3, :3] = Rotation.from_euler("z", row["heading_deg"], degrees=True).as_matrix()
        with np.load(path / "surface.npz") as field:
            half = np.array([float(field["width"]), float(field["length"])]) / 2
        corners = pavement.rectangle(-half[0], half[0], -half[1], half[1])
        inner = corners @ pose[:2, :2].T + pose[:2, 3]
        lo, hi = (-0.15, 20.0) if row["id"] == "pit" else (-20.0, -0.15)
        if inner[:, 0].min() < lo or inner[:, 0].max() > hi:
            raise ValueError("Reference pavement tiles must enclose their own opening")
        xy = np.vstack([pavement.rectangle(lo, hi, -20, 20), inner])
        faces = [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
        pavement.tile("Fixed road", np.c_[xy, np.zeros(8)], faces, material, ground)
        plugs.append(
            pavement.tile("Intact asphalt", np.c_[inner, np.zeros(4)], [(0, 1, 2, 3)], material, ground)
        )
        for obj in rig.imported(path / "asset.glb", ground @ pose, 1):
            name = obj.name.split(".")[0]
            if name == "Fracture lip and exposed bed":
                obj.pass_index = index * 32 + 2
            elif name in cover.PARTS:
                obj.pass_index = index * 32 + cover.PARTS[name]
            elif name not in ("Road around the aperture", "Road adjoining the fracture"):
                raise ValueError(f"Unknown reference mesh {name}")
            objects.append(obj)
    # A narrow bright survey bar gives simultaneous background/foreground samples along both edges.
    bpy.ops.mesh.primitive_cube_add(size=1)
    bar = bpy.context.object
    transform = np.diag([3.55, 0.043, 0.06, 1.0])
    transform[:3, :3] = Rotation.from_euler("z", 9, degrees=True).as_matrix() @ transform[:3, :3]
    transform[:3, 3] = [0, -0.01, 0.078]
    bar.matrix_world = Matrix((ground @ transform).tolist())
    bar.pass_index = 2
    paint = bpy.data.materials.new("Amber marker")
    paint.use_nodes = True
    paint.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.91, 0.31, 0.015, 1.0)
    graph = paint.node_tree
    marker = graph.nodes.new("ShaderNodeEmission")
    mix = graph.nodes.new("ShaderNodeMixShader")
    graph.links.new(graph.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], mix.inputs[0])
    graph.links.new(graph.nodes["Principled BSDF"].outputs[0], mix.inputs[1])
    graph.links.new(marker.outputs[0], mix.inputs[2])
    graph.links.new(mix.outputs[0], graph.nodes["Material Output"].inputs["Surface"])
    bar.data.materials.append(paint)
    return objects, plugs, marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("layouts", nargs="+", type=Path)
    args = parser.parse_args()
    ground = np.eye(4)
    ground[:3, :3] = Rotation.from_euler("zyx", [28, -3, 4], degrees=True).as_matrix()
    ground[:3, 3] = [-2.12, 9.68, 1.5]
    rig.SIZE = 768
    for layout in args.layouts:
        recipe = json.loads(layout.read_text())
        scene, cam = rig.setup_scene(20)
        scene.cycles.filter_width = 0.01
        scene.render.dither_intensity = 0.0
        objects, plugs, marker = build(scene, ground, recipe, args.assets)
        shots = (("side", 74, 24, [0, -1.58, 1.19]), ("rear", 176, 37, [0, 1.58, 1.39]))
        for name, yaw, pitch, eye in shots:
            for color, radiance in (("amber", (0.91, 0.31, 0.015)), ("cyan", (0.015, 0.62, 0.95))):
                marker.inputs["Color"].default_value = (*radiance, 1)
                for with_pair in (False, True):
                    for plug in plugs:
                        plug.hide_render = with_pair
                    rig.render(
                        scene,
                        cam,
                        ground,
                        objects,
                        args.out.resolve() / recipe["id"] / color,
                        name,
                        yaw,
                        with_pair,
                        eye=eye,
                        pitch_deg=pitch,
                    )
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
