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


def intact_road(plug, objects, cut):
    """Recover the native road BRDF outside the opening and place intact asphalt inside it."""
    original = next(o for o in objects if "Road adjoining" in o.name).data.materials[0]
    surface = original.copy()
    graph = surface.node_tree
    bsdf = next(n for n in graph.nodes if n.type == "BSDF_PRINCIPLED")
    out = next(n for n in graph.nodes if n.type == "OUTPUT_MATERIAL")
    # Generate a binary material selector from the measured aperture, with Blender's bottom-up image rows.
    image = bpy.data.images.new(
        "Reference asphalt boundary",
        cut.shape[1],
        cut.shape[0],
        alpha=False,
        is_data=True,
    )
    image.pixels.foreach_set(
        np.repeat(cut[::-1, :, None].astype("f4"), 4, axis=2).ravel()
    )
    mask = graph.nodes.new("ShaderNodeTexImage")
    mask.image = image
    mask.interpolation = "Closest"
    asphalt = graph.nodes.new("ShaderNodeBsdfPrincipled")
    asphalt.inputs["Base Color"].default_value = (0.105, 0.098, 0.09, 1)
    asphalt.inputs["Roughness"].default_value = 0.92
    mix = graph.nodes.new("ShaderNodeMixShader")
    graph.links.new(mask.outputs["Color"], mix.inputs[0])
    graph.links.new(bsdf.outputs[0], mix.inputs[1])
    graph.links.new(asphalt.outputs[0], mix.inputs[2])
    graph.links.new(mix.outputs[0], out.inputs["Surface"])
    plug.data.materials[0] = surface
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for loop in plug.data.loops:
        plug.data.uv_layers.active.data[loop.index].uv = uv[loop.vertex_index]


def save_albedo(folder):
    image = bpy.data.images.load(str(folder / "albedo-0001.exr"))
    values = np.empty(len(image.pixels), "f4")
    image.pixels.foreach_get(values)
    colour = np.clip(values.reshape(rig.SIZE, rig.SIZE, 4)[::-1, :, :3], 0, 1)
    encoded = np.where(
        colour <= 0.0031308, 12.92 * colour, 1.055 * colour ** (1 / 2.4) - 0.055
    )
    np.save(folder / "albedo-srgb.npy", np.uint8(np.rint(encoded * 255)))
    bpy.data.images.remove(image)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/measured-pavement")
    p.add_argument(
        "--out", type=Path, default=ROOT / "outputs/measured-pavement/cycles"
    )
    p.add_argument(
        "--layout",
        type=Path,
        default=ROOT / "gen2hole/examples/measured-pavement/crown-layout.json",
    )
    p.add_argument(
        "--paint-control",
        action="store_true",
        help="Add two removable road markings entirely within the measured openings",
    )
    p.add_argument(
        "--matched-apron",
        action="store_true",
        help="Use the native source pavement BRDF around each cut in the control host",
    )
    p.add_argument(
        "--albedo-pass",
        action="store_true",
        help="Export the host renderer's unlit diffuse color for independent material calibration",
    )
    args = p.parse_args()
    out = args.out.resolve()
    data = json.loads(args.layout.read_text())
    scene, camera = rig.setup_scene(48)
    if args.albedo_pass:
        bpy.context.view_layer.use_pass_diffuse_color = True
        bpy.context.view_layer.update()
        graph = scene.node_tree
        passes = next(n for n in graph.nodes if n.type == "R_LAYERS")
        file = graph.nodes["Passes"]
        file.file_slots.new("albedo-")
        graph.links.new(passes.outputs["DiffCol"], file.inputs[-1])
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("zyx", [71, 3, -2], degrees=True).as_matrix()
    world[:3, 3] = [-7.2, 11.8, 0.8]
    source = rig.plane(scene, world)
    material = source.data.materials[0]
    bpy.data.objects.remove(source, do_unlink=True)
    flat = rectangle(-1, 1, -20, 20)
    tile("Crown", np.c_[flat, np.zeros(4)], [(0, 1, 2, 3)], material, world)
    plugs, meshes, truth, marks = [], [], [], []
    if args.paint_control:
        paint = bpy.data.materials.new("Faded white road marking")
        paint.diffuse_color = (0.77, 0.77, 0.72, 1.0)
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
            cut = f["defect_mask"].astype(bool)
            if args.paint_control:
                from scipy.ndimage import distance_transform_edt

                step = np.array([2 * length, 2 * half]) / (np.array(cut.shape) - 1)
                clear = distance_transform_edt(cut, sampling=step)
                rc = np.array(np.unravel_index(clear.argmax(), clear.shape))
                local = (rc * step - [length, half])[::-1] * [1, -1]
                radius = clear.max() * 0.87
                stripe = (
                    rectangle(
                        -radius * 0.9,
                        radius * 0.9,
                        -min(0.08, radius * 0.3),
                        min(0.08, radius * 0.3),
                    )
                    + local
                )
                vertices = np.c_[stripe, np.full(4, 0.0004)]
                marks.append(
                    tile(
                        "Removable paint on intact asphalt",
                        vertices,
                        [(0, 1, 2, 3)],
                        paint,
                        world @ frame,
                    )
                )
        outer = rectangle(1, 20, -20, 20) if x > 0 else rectangle(-20, -1, -20, 20)
        outer = np.c_[outer, grade(outer[:, 0])]
        inside = (
            rectangle(-half, half, -length, length) @ frame[:3, :2].T + frame[:3, 3]
        )
        faces = [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
        tile("Pavement apron", np.r_[outer, inside], faces, material, world)
        plugs.append(tile("Intact pavement", inside, [(0, 1, 2, 3)], material, world))
        objs = rig.imported(folder / "asset.glb", world @ frame, 1)
        beds = [o for o in objs if "exposed bed" in o.name]
        if len(beds) != 1:
            raise ValueError(
                f"Expected one labelled measured bed: {[o.name for o in objs]}"
            )
        if args.matched_apron:
            intact_road(plugs[-1], objs, cut)
        beds[0].pass_index = 3 + index
        if "water" in row:
            pools = [o for o in objs if "Water surface" in o.name]
            retained = json.loads((folder / "metadata.json").read_text())["water"]["retained_liters"]
            if len(pools) != int(retained > 0):
                raise ValueError("Water surface count disagrees with the retained volume")
            for pool in pools:
                pool.pass_index = 16 + index
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
        if marks:
            for plug in plugs:
                plug.hide_render = False
            for mark in marks:
                mark.hide_render = False
            rig.render(
                scene, camera, world, meshes, out / "painted", name, yaw, False, eye=eye
            )
            for mark in marks:
                mark.hide_render = True
        for show in (False, True):
            for plug in plugs:
                plug.hide_render = show
            rig.render(scene, camera, world, meshes, out, name, yaw, show, eye=eye)
            folder = out / name / ("native" if show else "host")
            if args.albedo_pass and not show:
                save_albedo(folder)
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
