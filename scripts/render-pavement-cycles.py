"""Independently seat the measured meshes into cutouts on opposite sides of a crowned road in Cycles."""

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
ROOT = rig.ROOT


def tile(name, vertices, faces, material, world):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    uv = mesh.uv_layers.new()
    for loop in mesh.loops:
        uv.data[loop.index].uv = tuple(vertices[loop.vertex_index, :2] / 40)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.materials.append(material)
    obj.matrix_world = Matrix(world.tolist())
    obj.pass_index = 1
    return obj


def rectangle(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/measured-pavement")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/measured-pavement/cycles")
    args = p.parse_args()
    out = args.out.resolve()
    data = json.loads((ROOT / "gen2hole/examples/measured-pavement/crown-layout.json").read_text())
    scene, camera = rig.setup_scene(48)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("zyx", [71, 3, -2], degrees=True).as_matrix()
    world[:3, 3] = [-7.2, 11.8, 0.8]
    source = rig.plane(scene, world)
    material = source.data.materials[0]
    bpy.data.objects.remove(source, do_unlink=True)
    flat = rectangle(-1, 1, -20, 20)
    tile("Crown", np.c_[flat, np.zeros(4)], [(0, 1, 2, 3)], material, world)
    plugs, meshes, truth = [], [], []
    for index, row in enumerate(data["instances"]):
        x, y = row["xy_m"]
        sign = np.sign(x)
        slope = -0.065 if x > 0 else 0.083

        def grade(x):
            return slope * (x - sign)

        normal = np.array([-slope, 0, 1.0])
        normal /= np.linalg.norm(normal)
        theta = np.deg2rad(row["heading_deg"])
        forward = np.array([np.cos(theta), np.sin(theta), 0])
        forward -= normal * (forward @ normal)
        forward /= np.linalg.norm(forward)
        frame = np.eye(4)
        frame[:3, :3] = np.stack([forward, np.cross(normal, forward), normal], axis=1)
        frame[:3, 3] = [x, y, grade(x)]
        folder = args.assets / row["asset"]
        if not (folder / "asset.glb").exists():
            (folder,) = args.assets.glob(f"*/series/assets/*/{row['asset']}")
        with np.load(folder / "surface.npz") as f:
            half = float(f["width"]) / 2
            length = float(f["length"]) / 2
        outer = rectangle(1, 20, -20, 20) if x > 0 else rectangle(-20, -1, -20, 20)
        outer = np.c_[outer, grade(outer[:, 0])]
        inside = rectangle(-half, half, -length, length) @ frame[:3, :2].T + frame[:3, 3]
        faces = [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
        tile("Pavement apron", np.r_[outer, inside], faces, material, world)
        plugs.append(tile("Intact pavement", inside, [(0, 1, 2, 3)], material, world))
        objs = rig.imported(folder / "asset.glb", world @ frame, 1)
        beds = [o for o in objs if "exposed bed" in o.name]
        if len(beds) != 1:
            raise ValueError(f"Expected one labelled measured bed: {[o.name for o in objs]}")
        beds[0].pass_index = 3 + index
        meshes.extend(objs)
        truth.append((world @ frame).tolist())
    bpy.ops.mesh.primitive_cube_add(size=1)
    barrier = bpy.context.object
    transform = np.diag([0.45, 0.16, 0.36, 1.0])
    transform[:3, 3] = [2.42, -0.45, 0.12]
    barrier.matrix_world = Matrix((world @ transform).tolist())
    barrier.pass_index = 2
    for name, yaw, eye in [
        ("center", 0, [0, -3.7, 1.45]),
        ("edge", 68, [0, -2.8, 1.15]),
        ("rear", 171, [0, 3.2, 1.2]),
    ]:
        for show in (False, True):
            for plug in plugs:
                plug.hide_render = show
            rig.render(scene, camera, world, meshes, out, name, yaw, show, eye=eye)
            folder = out / name / ("native" if show else "host")
            ids = np.load(folder / "id.npy")
            if not set((1, 2)).issubset(set(np.unique(ids))) and name != "edge":
                raise ValueError(f"Missing foreground reference in {name}")
            assert (
                np.all(np.isfinite(np.load(folder / "range.npy")[ids > 0]))
                and (folder / "rgb.png").stat().st_size
            )
    (out / "reference.json").write_text(json.dumps(truth, indent=2) + "\n")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print("Verified native crowned road passes.", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
