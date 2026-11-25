"""Render physical and metric oracle passes with Blender Cycles' equidistant panoramic camera."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SIZE = 512
FOV = 220.0


def imported(path, matrix, index):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    models = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    for obj in models:
        obj.matrix_world = Matrix(matrix.tolist()) @ obj.matrix_world
        obj.pass_index = index
    return models


def plane(scene, world):
    bpy.ops.mesh.primitive_plane_add(size=40)
    road = bpy.context.object
    road.pass_index = 1
    road.matrix_world = Matrix(world.tolist())
    mat = bpy.data.materials.new("Photographed asphalt")
    mat.use_nodes = True
    source = ROOT / "gen2hole/examples/traffic-cones/cycles"
    tex = source.parent / "asphalt_02_diff_1k.jpg"
    if (
        hashlib.sha256(tex.read_bytes()).hexdigest()
        != json.loads((source / "source.json").read_text())["sha256"]
    ):
        raise ValueError("Changed road texture")
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    color = nodes.new("ShaderNodeTexImage")
    color.image = bpy.data.images.load(str(tex))
    uv = nodes.new("ShaderNodeTexCoord")
    repeat = nodes.new("ShaderNodeVectorMath")
    repeat.operation = "SCALE"
    repeat.inputs["Scale"].default_value = 16
    links.new(uv.outputs["UV"], repeat.inputs[0])
    links.new(repeat.outputs[0], color.inputs[0])
    surface = nodes.get("Principled BSDF")
    surface.inputs["Roughness"].default_value = 0.9
    links.new(color.outputs["Color"], surface.inputs["Base Color"])
    road.data.materials.append(mat)
    scene.world = bpy.data.worlds.new("Diffuse sky")
    scene.world.use_nodes = True
    env = scene.world.node_tree.nodes["Background"]
    env.inputs["Color"].default_value = (0.87, 0.93, 1.0, 1)
    env.inputs["Strength"].default_value = 0.7
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 2
    sun.angle = np.deg2rad(2)
    lamp = bpy.data.objects.new("Sun", sun)
    scene.collection.objects.link(lamp)
    lamp.rotation_euler = (0.44, -0.63, -0.3)
    return road


def render(scene, camera, world, objects, out, name, yaw, with_pair, *, eye=None, pitch_deg=15):
    folder = out / name / ("native" if with_pair else "host")
    folder.mkdir(exist_ok=True, parents=True)
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch_deg)
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0])
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    pose[:3, 3] = [1.10, 1.76, 0.89] if name == "return-view" else [0, -1.76, 0.89]
    if eye is not None:
        pose[:3, 3] = eye
    pose = world @ pose
    camera.matrix_world = Matrix((pose @ np.diag([1, -1, -1, 1])).tolist())
    for o in objects:
        o.hide_render = not with_pair
    scene.render.filepath = str(folder / "rgb.png")
    file = scene.node_tree.nodes.get("Passes")
    file.base_path = str(folder)
    bpy.ops.render.render(write_still=True)
    # Cycles writes named scalar EXR channels; use its own reader to preserve metres and object IDs.
    passes = ("range", "id", "position") if "position-" in file.inputs else ("range", "id")
    for stem in passes:
        path = folder / (stem + "-0001.exr")
        image = bpy.data.images.load(str(path))
        values = np.empty(len(image.pixels), "f4")
        image.pixels.foreach_get(values)
        pixels = values.reshape(image.size[1], image.size[0], 4)[::-1]
        channel = pixels[..., :3] if stem == "position" else pixels[..., 0]
        np.save(folder / (stem + ".npy"), channel.astype("u1") if stem == "id" else channel)
        bpy.data.images.remove(image)
    if camera.data.type == "PANO" and camera.data.panorama_type == "EQUIRECTANGULAR":
        lens = {"projection": ["EQUIRECTANGULAR", scene.render.resolution_x, scene.render.resolution_y]}
    else:
        lens = {"ortho_width_m": camera.data.ortho_scale} if camera.data.type == "ORTHO" else {"fov_deg": FOV}
        lens["size"] = SIZE
    (folder / "camera.json").write_text(
        json.dumps(
            dict(
                camera_to_world=pose.tolist(),
                **lens,
                reference_ground_to_world=world.tolist(),
            ),
            indent=2,
        )
        + "\n"
    )


def setup_scene(samples):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.seed = 6
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 5
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = False
    camera_data = bpy.data.cameras.new("220 degree fisheye")
    camera_data.type = "PANO"
    camera_data.panorama_type = "FISHEYE_EQUIDISTANT"
    camera_data.fisheye_fov = np.deg2rad(FOV)
    camera = bpy.data.objects.new("Sensor", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    bpy.context.view_layer.use_pass_z = True
    bpy.context.view_layer.use_pass_object_index = True
    scene.use_nodes = True
    graph = scene.node_tree
    graph.nodes.clear()
    passes = graph.nodes.new("CompositorNodeRLayers")
    comp = graph.nodes.new("CompositorNodeComposite")
    graph.links.new(passes.outputs["Image"], comp.inputs[0])
    file = graph.nodes.new("CompositorNodeOutputFile")
    file.name = "Passes"
    file.format.file_format = "OPEN_EXR"
    file.format.color_depth = "32"
    file.format.color_mode = "RGB"
    file.file_slots[0].path = "range-"
    file.file_slots.new("id-")
    graph.links.new(passes.outputs["Depth"], file.inputs[0])
    graph.links.new(passes.outputs["IndexOB"], file.inputs[1])
    return scene, camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/traffic-cones")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/traffic-cones/cycles")
    args = p.parse_args()
    out = args.out.resolve()
    scene, camera = setup_scene(128)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("zyx", [36, -4, 3], degrees=True).as_matrix()
    world[:3, 3] = [11.7, -5.8, 2.1]
    plane(scene, world)
    # A third, pre-existing cone is a measured host occluder for the inserted pair.
    occ = np.eye(4)
    occ[:3, 3] = [0.32, -0.58, 0]
    imported(args.assets / "C236/asset.glb", world @ occ, 2)
    objects = imported(args.assets / "C236-C237/asset.glb", world, 3)
    if len(objects) != 2:
        raise ValueError("Expected two separately labelled GLB instances")
    for obj in objects:
        obj.pass_index = 3 if "@upright" in obj.name else 4
    for name, yaw in [("center", 4), ("left-edge", 92), ("return-view", -155)]:
        for show in (False, True):
            render(scene, camera, world, objects, out, name, yaw, show)

    for camera_name in ("center", "left-edge", "return-view"):
        for state in ("host", "native"):
            folder = out / camera_name / state
            ids = np.load(folder / "id.npy")
            ranges = np.load(folder / "range.npy")
            if ranges.shape != (SIZE, SIZE) or not np.all((ranges[ids > 0] > 0) & (ranges[ids > 0] < 100)):
                raise ValueError("Invalid native metric depth")
            if not (folder / "rgb.png").stat().st_size:
                raise ValueError("Missing native colour pass")
            if ids.shape != (SIZE, SIZE) or set(np.unique(ids)) != (
                set(range(3)) if state == "host" else set(range(5))
            ):
                raise ValueError(f"Incomplete native object-index pass: {camera_name}/{state}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print("Verified six Cycles renders and their distance and instance passes.", flush=True)


if __name__ == "__main__":
    main()
    # The embedded bpy 4.2 wheel can fault when Python unloads Cycles after a completed render.
    # Every render, disk reread and pass check above must succeed before taking this exit path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
