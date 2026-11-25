"""Render two unrectified 220-degree camera pairs with independent Cycles depth and visibility."""

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np
from scipy.spatial.transform import Rotation

rig = import_module("render-cone-cycles")
inlets = import_module("render-inlet-control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("layouts", nargs="+", type=Path)
    args = parser.parse_args()
    ground = np.eye(4)
    ground[:3, :3] = Rotation.from_euler("zyx", [21, -3, 4], degrees=True).as_matrix()
    ground[:3, 3] = [-2.16, 5.37, 1.4]
    rig.SIZE = 448
    for path in args.layouts:
        spec = json.loads(path.read_text())
        scene, camera = rig.setup_scene(8)
        scene.cycles.filter_width = 0.01
        scene.render.dither_intensity = 0.0
        bpy.context.view_layer.use_pass_position = True
        nodes = scene.node_tree.nodes
        file = nodes["Passes"]
        file.file_slots.new("position-")
        layers = next(node for node in nodes if node.type == "R_LAYERS")
        scene.node_tree.links.new(layers.outputs["Position"], file.inputs["position-"])
        objects, plugs, _ = inlets.build(
            scene, ground, spec, args.models, split_x=-0.31
        )
        for shot, yaw, pitch, eye in (
            ("side", 70, 24, [0, -1.58, 1.19]),
            ("rear", 176, 37, [0, 1.58, 1.39]),
        ):
            out = args.out.resolve() / spec["id"] / shot
            for plug in plugs:
                plug.hide_render = False
            secondary_eye = np.array(eye) + [
                0.22 * np.cos(np.deg2rad(yaw)),
                -0.22 * np.sin(np.deg2rad(yaw)),
                0.018,
            ]
            rig.render(
                scene,
                camera,
                ground,
                objects,
                out,
                "right",
                yaw + 7,
                False,
                eye=secondary_eye,
                pitch_deg=pitch - 3,
            )
            rig.render(
                scene,
                camera,
                ground,
                objects,
                out,
                "left",
                yaw,
                False,
                eye=eye,
                pitch_deg=pitch,
            )
            for plug in plugs:
                plug.hide_render = True
            rig.render(
                scene,
                camera,
                ground,
                objects,
                out,
                "left",
                yaw,
                True,
                eye=eye,
                pitch_deg=pitch,
            )
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
