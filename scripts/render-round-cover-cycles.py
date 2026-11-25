"""Render independent Cycles part/depth passes with real mesh apertures on a flat metric street."""

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

rig = import_module("render-cone-cycles")
tile = import_module("render-pavement-cycles").tile
ROOT = rig.ROOT
EXAMPLES = ROOT / "gen2hole/examples/scanned-round-covers"
PARTS = {
    "Cast ring and cover seat": 5,
    "Cast seat riser and ledge": 5,
    "Inspection shaft": 6,
    "Chipped concrete collar": 8,
    "Shaft bottom": 7,
    "Lid face": 9,
    "Lid underside": 10,
    "Lid aperture walls and perimeter": 11,
}


def cut_road(corners, rings):
    from mathutils import Vector
    from mathutils.geometry import delaunay_2d_cdt

    loops = [corners, *rings]
    boundaries = []
    start = 0
    for loop in loops:
        boundaries.append(list(range(start, start + len(loop))))
        start += len(loop)
    points, _, faces, _, _, owners = delaunay_2d_cdt(
        [Vector(p) for p in np.concatenate(loops)], [], boundaries, 0, 1e-7
    )
    xy = np.asarray(points)
    tri = np.asarray([face for face, source in zip(faces, owners) if list(source) == [0]])
    edge1 = xy[tri[:, 1]] - xy[tri[:, 0]]
    edge2 = xy[tri[:, 2]] - xy[tri[:, 0]]
    area = abs(np.cross(edge1, edge2)).sum() / 2
    loop_areas = [abs(np.cross(p, np.roll(p, -1, axis=0)).sum()) / 2 for p in loops]
    expected = loop_areas[0] - sum(loop_areas[1:])
    if abs(area - expected) > 1e-4:
        raise ValueError("Reference host surface does not match the requested apertures")
    return xy, tri


def build(scene, world, recipe, assets, profiles, position_pass=False):
    source = rig.plane(scene, world)
    asphalt = source.data.materials[0]
    bpy.data.objects.remove(source, do_unlink=True)
    rings, plugs, objects = [], [], []
    corners = np.array([[-12.0, -12.0], [12.0, -12.0], [12.0, 12.0], [-12.0, 12.0]])
    for i, row in enumerate(recipe["instances"], 1):
        key = row["asset"]
        config = profiles[key]
        radius = config["size_m"][0] / 2 + config["frame_band_m"]
        frame = np.eye(4)
        frame[:2, 3] = row["xy_m"]
        frame[:3, :3] = Rotation.from_euler("z", row["heading_deg"], degrees=True).as_matrix()
        angles = np.arange(256) * 2 * np.pi / 256
        unit_circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        direct = assets / key / "asset.glb"
        if not direct.exists():
            (direct,) = assets.glob(f"*/series/assets/*/{key}/asset.glb")
        if position_pass:
            half = np.array(config["size_m"]) / 2 + config["frame_band_m"]
            if config["shape"] == "rect":
                local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
            elif config["shape"] == "round":
                local = unit_circle * half
            else:
                raise ValueError("Expected a rectangular or round utility-cover frame")
            meta = json.loads((direct.parent / "metadata.json").read_text())
            install = np.array(meta.get("frame_transform") or np.eye(4))
            outline = local @ install[:2, :2].T + install[:2, 3]
        else:
            outline = radius * unit_circle
        outline = outline @ frame[:2, :2].T + frame[:2, 3]
        rings.append(outline)
        plugs.append(
            tile(
                "Original road disk",
                np.c_[outline, np.zeros(len(outline))],
                [list(range(len(outline)))],
                asphalt,
                world,
            )
        )
        for obj in rig.imported(direct, world @ frame, 1):
            name = obj.name.split(".")[0]
            if name == "Road around the aperture":
                bpy.data.objects.remove(obj, do_unlink=True)
                continue
            obj.pass_index = i * 32 + PARTS[name]
            objects.append(obj)
    if position_pass:
        xy, faces = cut_road(corners, rings)
    else:
        xy = np.concatenate([corners, *rings])
        faces = Delaunay(xy).simplices
        for circle in rings:
            center = circle.mean(0)
            radius = np.linalg.norm(circle[0] - center)
            faces = faces[
                np.linalg.norm(xy[faces].mean(1) - center, axis=1) >= radius * np.cos(np.pi / len(circle))
            ]
    tile("Host asphalt with two open circles", np.c_[xy, np.zeros(len(xy))], faces.tolist(), asphalt, world)
    bpy.ops.mesh.primitive_cube_add(size=1)
    obstacle = bpy.context.object
    transform = np.diag([0.18, 0.08, 0.10, 1.0])
    # Suspended calibration target clears the shifted cover and the host asphalt.
    transform[:3, 3] = [0.86, -0.14, 0.16]
    obstacle.matrix_world = Matrix((world @ transform).tolist())
    obstacle.pass_index = 2
    return objects, plugs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "runs/scanned-round-covers")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/scanned-round-covers/cycles")
    parser.add_argument("--layout", type=Path, action="append", help="Layout JSON beside the numbered scan folders")
    parser.add_argument("--readout", action="store_true", help="Also render eight consecutive moving camera poses")
    parser.add_argument("--position-pass", action="store_true", help="Export an unfiltered world-position pass and cut registered rectangular or round frames")
    args = parser.parse_args()
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("zyx", [37, -3, 4], degrees=True).as_matrix()
    world[:3, 3] = [-3.27, 7.55, 1.85]
    layouts = args.layout or [EXAMPLES / (k + ".json") for k in ("M240-A241", "A240-M241")]
    for path in layouts:
        layout = json.loads(path.read_text())
        key = layout["id"]
        profiles = {}
        for survey in sorted(path.parent.glob("*/survey.json")):
            config = json.loads(survey.read_text())
            for state in config["states"]:
                profiles[state["id"]] = config["perforations"]
        scene, camera = rig.setup_scene(32)
        if args.position_pass:
            bpy.context.view_layer.use_pass_position = True
            file = scene.node_tree.nodes["Passes"]
            file.file_slots.new("position-")
            source = next(n for n in scene.node_tree.nodes if n.type == "R_LAYERS")
            scene.node_tree.links.new(source.outputs["Position"], file.inputs["position-"])
        objects, plugs = build(
            scene, world, layout, args.assets, profiles, args.position_pass
        )
        shots = [("rear", 180, 28, [0, 2.4, 1.3]), ("ortho", 0, 67, [0, -1.17, 2.76])]
        if args.position_pass:
            shots.append(("side", 76, 24, [0, -1.38, 0.92]))
        if args.readout:
            # Eight time slices of a 24 ms readout: forward speed 12.5 m/s and yaw rate 25 deg/s.
            for index in range(8):
                fraction = (index + 0.5) / 8
                shots.append((f"readout-{index}", 180 + 0.6 * fraction, 28, [0, 1.58 + .30 * fraction, 1.19]))
        for view, yaw, pitch, eye in shots:
            camera.data.type = "ORTHO" if view == "ortho" else "PANO"
            camera.data.ortho_scale = 3.15
            for show in (False, True):
                for plug in plugs:
                    plug.hide_render = show
                rig.render(
                    scene,
                    camera,
                    world,
                    objects,
                    args.out.resolve() / key,
                    view,
                    yaw,
                    show,
                    eye=eye,
                    pitch_deg=pitch,
                )
    bpy.ops.wm.read_factory_settings(use_empty=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
