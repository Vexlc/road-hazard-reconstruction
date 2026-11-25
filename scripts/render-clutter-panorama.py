"""Independently render paired traffic obstructions on the seam of a full-sphere stereo rig."""

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix
import numpy as np
from scipy.spatial.transform import Rotation

rig = import_module("render-cone-cycles")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("pairs", nargs="+")
    args = p.parse_args()
    ground = np.eye(4)
    ground[:3, :3] = Rotation.from_euler("zyx", [18, -2, 4], degrees=True).as_matrix()
    ground[:3, 3] = [4.06, -2.19, 1.45]
    for pair in args.pairs:
        scene, camera = rig.setup_scene(12)
        scene.cycles.filter_width = 0.01
        camera.data.panorama_type = "EQUIRECTANGULAR"
        scene.render.resolution_x = 960
        scene.render.resolution_y = 480
        scene.render.dither_intensity = 0
        bpy.context.view_layer.use_pass_position = True
        layers = next(n for n in scene.node_tree.nodes if n.type == "R_LAYERS")
        file = scene.node_tree.nodes["Passes"]
        file.file_slots.new("position-")
        scene.node_tree.links.new(layers.outputs["Position"], file.inputs["position-"])
        rig.plane(scene, ground)
        rows = json.loads((args.models / pair / "metadata.json").read_text())[
            "instances"
        ]
        objects = rig.imported(args.models / pair / "asset.glb", ground, 3)
        for obj in objects:
            (i,) = [i for i, r in enumerate(rows) if "@" + r["id"] in obj.name]
            obj.pass_index = 3 + i
        # Move the seam onto each member in turn. A bar remains in the host in both views.
        bpy.ops.mesh.primitive_cube_add(size=1)
        bar = bpy.context.object
        mat = np.diag([1.78, 0.046, 0.068, 1.0])
        mat[:3, 3] = [0, -0.26, 0.14]
        bar.matrix_world = Matrix((ground @ mat).tolist())
        bar.pass_index = 2
        for i, r in enumerate(rows):
            # Aim the seam at the geometry centre; a wrench can extend well past its support origin.
            coords = []
            for obj in objects:
                if obj.pass_index != i + 3:
                    continue
                vertices = np.array([v.co[:] for v in obj.data.vertices])
                pose = np.linalg.inv(ground) @ np.array(obj.matrix_world)
                coords.append(vertices @ pose[:3, :3].T + pose[:3, 3])
            bounds = np.concatenate(coords)
            center_x = (bounds[:, 0].min() + bounds[:, 0].max()) / 2
            close = bounds[:, 2].max() < 0.12
            distance = 0.83 if close else 1.55
            eye = np.array([center_x, distance if i else -distance, 0.91])
            yaw = 0 if i else 180
            out = args.out.resolve() / pair / r["id"]
            rig.render(
                scene,
                camera,
                ground,
                objects,
                out,
                "secondary",
                yaw - 4,
                False,
                eye=eye + [0.16, -0.012, 0.025],
                pitch_deg=-1,
            )
            for inserted in (False, True):
                rig.render(
                    scene,
                    camera,
                    ground,
                    objects,
                    out,
                    "primary",
                    yaw,
                    inserted,
                    eye=eye,
                    pitch_deg=3,
                )
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
