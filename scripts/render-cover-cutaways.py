"""Render aligned inspections and sectioned views from a family of exported cover GLBs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import bpy
import bmesh
from mathutils import Vector
import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gen2hole/recipes' / name) for name in ('common', 'series')]
from render_mesh import prepare_view


def cut(obj, point, normal):
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    world = obj.matrix_world
    bmesh.ops.bisect_plane(
        mesh,
        geom=[*mesh.verts, *mesh.edges, *mesh.faces],
        plane_co=world.inverted() @ Vector(point),
        plane_no=(world.to_3x3().transposed() @ Vector(normal)).normalized(),
        dist=1e-6,
        clear_inner=True,
    )
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()


def check_export(original, section):
    source, cutaway = [trimesh.load(path, force='scene', process=False) for path in (original, section)]
    solids = {}
    if set(source.graph.nodes_geometry) != set(cutaway.graph.nodes_geometry):
        raise ValueError('Sectioned delivery lost a named part')
    for name in source.graph.nodes_geometry:
        parts = []
        for scene in (source, cutaway):
            transform, geometry = scene.graph[name]
            mesh = scene.geometry[geometry]
            points = trimesh.transform_points(mesh.vertices, transform)[mesh.faces]
            parts.append(
                (points, mesh.visual.uv[mesh.faces], mesh.visual.material.baseColorTexture.convert('RGB'))
            )
        (a, ua, ia), (b, ub, ib) = parts
        if ia.size != ib.size or ia.tobytes() != ib.tobytes():
            raise ValueError('Sectioned material changed: ' + name)
        if name.startswith('Lid ') or name == 'Shaft bottom':
            if len(a) != len(b):
                raise ValueError('Sectioned view clipped a solid: ' + name)
            _, idx = cKDTree(a.mean(1)).query(b.mean(1))
            ref = a[idx]
            shifts = np.argmin(
                [np.linalg.norm(b - np.roll(ref, i, axis=1), axis=2).max(1) for i in range(3)], axis=0
            )
            order = (np.arange(3)[None, :] - shifts[:, None]) % 3
            error = float(abs(b - ref[np.arange(len(b))[:, None], order]).max())
            uv_error = float(abs(ub - ua[idx[:, None], order]).max())
            if len(np.unique(idx)) != len(b) or max(error, uv_error) > 2e-6:
                raise ValueError('Sectioned view changed solid geometry or UV: ' + name)
            solids[name] = dict(triangles=len(b), max_coordinate_error_m=error, max_uv_error=uv_error)
        elif len(b) >= len(a) or b[:, :, 2].max() > 0.012002:
            raise ValueError('Inspection geometry crossed its cut plane: ' + name)
    proof = dict(
        source_glb_sha256=sha256(original.read_bytes()).hexdigest(),
        cutaway_glb_sha256=sha256(section.read_bytes()).hexdigest(),
        preserved_parts=solids,
        visible_halfspace='y >= -0.012 m for the wall, seat and pavement',
        shaft_floor_intact=True,
    )
    section.with_suffix('.json').write_text(json.dumps(proof, indent=2) + '\n')


def render(run):
    bundle = run / 'series'
    specs = json.loads((bundle / 'calibration/survey.json').read_text())
    ids = [s['id'] for s in specs['states']]
    manifest = {r['id']: r for r in json.loads((bundle / 'assets/asset_manifest.json').read_text())}
    points = []
    for key in ids:
        with np.load(bundle / 'mesh_cache' / (key + '.npz')) as mesh:
            take = np.array([n.startswith(('Cast ring', 'Lid ')) for n in mesh['part_names']])[
                mesh['face_part']
            ]
            points.append(mesh['xyz'][np.unique(mesh['faces'][take])])
    points = np.vstack(points)
    radius = max(manifest[ids[0]]['panel_size_m']) / 2
    depth = max(manifest[k]['depth_design_m'] for k in ids)
    for key in ids:
        dst = run / 'evaluation' / key
        dst.mkdir(parents=True, exist_ok=True)
        asset = (bundle / 'assets' / manifest[key]['glb']).parent
        scene, _ = prepare_view(asset, bundle / 'mesh_cache', 'inspection', 960, 665)
        scene.cycles.samples = 32
        for mode, offset in [('aligned', [0.16, -1.0, 1.31]), ('cutaway', [0.23, -1.0, 0.45])]:
            away = Vector(offset).normalized()
            center = Vector(
                (
                    (points[:, 0].min() + points[:, 0].max()) * 0.5,
                    0,
                    -0.5 * depth if mode == 'cutaway' else 0,
                )
            )
            if mode == 'cutaway':
                for ob in list(scene.objects):
                    if ob.name.startswith('Inspection ground'):
                        bpy.data.objects.remove(ob, do_unlink=True)
                    elif ob.type == 'MESH' and not ob.name.startswith(('Lid ', 'Shaft bottom')):
                        cut(ob, (0, -0.012, 0), (0, 1, 0))
                        if ob.name.startswith('Road '):
                            cut(ob, (0, radius * 1.23, 0), (0, -1, 0))
                            cut(ob, (-radius * 1.23, 0, 0), (1, 0, 0))
                            cut(ob, (max(points[:, 0]) + radius * 0.15, 0, 0), (-1, 0, 0))
                lamp = bpy.data.lights.new('Light through cut', 'AREA')
                lamp.energy = 22
                lamp.shape = 'DISK'
                lamp.size = 1.1
                light = bpy.data.objects.new(lamp.name, lamp)
                scene.collection.objects.link(light)
                light.location = (0.28, -1.3, 0.23)
                light.rotation_euler = (center - light.location).to_track_quat('-Z', 'Y').to_euler()
                bpy.ops.object.select_all(action='DESELECT')
                for ob in scene.objects:
                    if ob.type == 'MESH':
                        ob.select_set(True)
                view_mesh = dst / 'cutaway.glb'
                bpy.ops.export_scene.gltf(
                    filepath=str(view_mesh.resolve()),
                    export_format='GLB',
                    use_selection=True,
                    export_animations=False,
                    export_cameras=False,
                    export_lights=False,
                )
                check_export(asset / 'asset.glb', view_mesh)
            cam = scene.camera
            cam.location = center + away * 3.2
            cam.rotation_euler = (-away).to_track_quat('-Z', 'Y').to_euler()
            cam.data.ortho_scale = max(
                1.12 * (points[:, 0].max() - points[:, 0].min()),
                2.85 * radius,
                (depth + 0.37) * 960 / 665 if mode == 'cutaway' else 0,
            )
            scene.render.filepath = str((dst / (mode + '.jpg')).resolve())
            bpy.ops.render.render(write_still=True)
    print('Rendered matched cameras for', *ids, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    render(parser.parse_args().run)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
