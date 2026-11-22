"""Extract connected openings with a measured road margin from a registered pothole GLB."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance
import trimesh


BED = 'Fracture lip and exposed bed'
ROAD = 'Road adjoining the fracture'
MAPS = ('baseColorTexture', 'metallicRoughnessTexture', 'normalTexture')


def read(path):
    return json.loads(path.read_text())


def planar(vertices):
    return vertices[:, [0, 2]] * [1, -1]


def crop_part(mesh, faces, offset):
    used, inverse = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(
        vertices=mesh.vertices[used] - offset,
        faces=inverse.reshape(-1, 3),
        vertex_normals=mesh.vertex_normals[used],
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=mesh.visual.uv[used], material=mesh.visual.material),
    )


def export_checked(path, parts, offset, *, metadata=None):
    scene = trimesh.Scene()
    if metadata:
        scene.metadata.update(metadata)
    for name, (parent, faces) in parts.items():
        scene.add_geometry(crop_part(parent, faces, offset), geom_name=name, node_name=name)
    blob = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    loaded = trimesh.load_scene(BytesIO(blob), file_type='glb', process=False)
    worst = 0.0
    for name, (parent, faces) in parts.items():
        child = loaded.geometry[name]
        if not np.array_equal(loaded.graph[name][0], np.eye(4)):
            raise ValueError('Unexpected component transform')
        error = abs((child.vertices[child.faces] + offset) - parent.vertices[faces]).max()
        worst = max(worst, float(error))
        if error > 1e-6 or not np.array_equal(child.visual.uv[child.faces], parent.visual.uv[faces]):
            raise ValueError('Cropped faces differ from the source mesh')
        if not np.allclose(
            child.vertex_normals[child.faces], parent.vertex_normals[faces], rtol=0, atol=2e-6
        ):
            raise ValueError('Cropped normals differ from the source mesh')
        for channel in MAPS:
            if not np.array_equal(
                getattr(child.visual.material, channel), getattr(parent.visual.material, channel)
            ):
                raise ValueError('Changed material channel ' + channel)
    with TemporaryDirectory(dir=path.parent) as stage:
        tmp = Path(stage) / path.name
        tmp.write_bytes(blob)
        tmp.replace(path)
    return worst


def fit_polygon(vertices, capture, labels):
    frame = capture['geometry']
    local = vertices[:, [0, 2, 1]] * [1, -1, 1]
    world = local @ np.array(frame['local_axes_in_anchor_camera']).T + frame['local_origin_in_anchor_m']
    (key,) = capture['fit']
    pose = np.array(capture['poses_view_to_anchor'][key])
    camera = (world - pose[:3, 3]) @ pose[:3, :3]
    pixels = camera @ np.array(capture['inputs'][key]['K']).T
    u, v = np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int).T
    h, w = labels.shape
    on_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    votes = labels[v[on_image], u[on_image]]
    votes = votes[votes > 0]
    if len(votes) < 0.5 * len(vertices):
        raise ValueError('Too little overlap with the source annotation')
    return int(np.bincount(votes).argmax())


def components(run, key, rows, margin):
    bundle = run / 'series'
    cal = bundle / 'calibration'
    src = bundle / 'assets/pothole' / key / 'asset.glb'
    capture = read(cal / (key + '_capture.json'))
    evidence = read(run / 'evaluation/measurements.json')[key]
    if sha256(src.read_bytes()).hexdigest() != evidence['glb_sha256'] or len(capture['fit']) != 1:
        raise ValueError('Expected a checked single fitting frame with its original outline')
    labels, count = ndimage.label(np.asarray(Image.open(cal / (key + '_outline.png'))) > 0)
    frame_guard = 2
    border = set(
        np.r_[
            labels[:frame_guard].ravel(),
            labels[-frame_guard:].ravel(),
            labels[:, :frame_guard].ravel(),
            labels[:, -frame_guard:].ravel(),
        ]
    )
    scene = trimesh.load_scene(src, process=False)
    bed, road = (scene.geometry[name] for name in (BED, ROAD))
    for mesh in (bed, road):
        # Write the decoded maps losslessly when trimesh serializes the cropped GLB.
        for channel in MAPS:
            getattr(mesh.visual.material, channel).format = 'PNG'
    vertex_ids = trimesh.graph.connected_component_labels(bed.edges_unique)
    face_ids = vertex_ids[bed.faces[:, 0]]
    groups = np.unique(face_ids)
    if len(groups) != count or not np.all(vertex_ids[bed.faces] == face_ids[:, None]):
        raise ValueError('Source polygons and separated mesh openings disagree')
    centers = np.stack([planar(bed.vertices[vertex_ids == g]).mean(0) for g in groups])
    ordering = np.argsort(-centers[:, 1], kind='stable')
    if not 1 <= rows <= len(groups):
        raise ValueError('Row count outside component count')
    # Rows run along asset-local X. Split on the widest lateral (Y) gaps.
    cuts = np.sort(np.argsort(np.diff(-centers[ordering, 1]))[-(rows - 1) :] + 1) if rows > 1 else []
    lanes = np.split(ordering, cuts)
    gaps = []
    dest = run / 'evaluation' / key / 'components'
    dest.mkdir(parents=True, exist_ok=True)
    road_xy = planar(road.triangles_center)
    tree = cKDTree(road_xy)
    pieces = {}
    used_polygons = set()
    for lane, indices in enumerate(lanes, 1):
        ordered = indices[np.argsort(centers[indices, 0])]
        gaps.extend(np.linalg.norm(np.diff(centers[ordered], axis=0), axis=1).tolist())
        for ordinal, idx in enumerate(ordered, 1):
            name = f'R{lane}-{ordinal:02}'
            faces = bed.faces[face_ids == groups[idx]]
            vertices = bed.vertices[vertex_ids == groups[idx]]
            poly = fit_polygon(vertices, capture, labels)
            if poly in used_polygons:
                raise ValueError('Two mesh components matched one source polygon')
            used_polygons.add(poly)
            xy = planar(vertices)
            center = centers[idx]
            axis = np.linalg.eigh(np.cov(xy.T))[1]
            short, long = sorted(np.ptp((xy - center) @ axis, axis=0))
            distance_to_center = np.linalg.norm(xy - center, axis=1).max()
            around = np.asarray(tree.query_ball_point(center, distance_to_center + margin))
            rim = around[cKDTree(xy).query(road_xy[around])[0] <= margin]
            offset = np.array([center[0], 0.0, -center[1]])
            if not len(rim):
                raise ValueError('Missing adjacent road around ' + name)
            datum = float(np.median(road.triangles_center[rim, 1]))
            comp = dict(
                row=lane,
                fit_polygon=poly,
                center_local_xy_m=center.tolist(),
                length_m=float(long),
                width_m=float(short),
                rim_elevation_m=datum,
                rim_relative_depth_p95_m=float(datum - np.percentile(vertices[:, 1], 5)),
                near_fit_image_edge=poly in border,
                bed_triangles=len(faces),
                heldout={},
            )
            if poly not in border:
                path = dest / (name + '.glb')
                comp.update(
                    file=path.name,
                    gltf_translation_to_parent_m=offset.tolist(),
                    max_reassembled_axis_error_m=export_checked(
                        path, {BED: (bed, faces), ROAD: (road, road.faces[rim])}, offset
                    ),
                    sha256=sha256(path.read_bytes()).hexdigest(),
                )
            pieces[name] = comp
    # Match only independently measured complete centers, never reusing a source opening.
    names = list(pieces)
    xy = np.array([pieces[n]['center_local_xy_m'] for n in names])
    nearest = distance.cdist(xy, xy)
    np.fill_diagonal(nearest, np.inf)
    for keyframe, result in evidence['heldout'].items():
        observed = np.asarray(result['heldout_centers_local_m']).reshape(-1, 2)
        costs = distance.cdist(xy, observed)
        first, second = linear_sum_assignment(costs)
        for i, j in zip(first, second):
            if costs[i, j] < nearest[i].min() * 0.5:
                pieces[names[i]]['heldout'][keyframe] = dict(
                    center_local_xy_m=observed[j].tolist(), distance_m=float(costs[i, j])
                )
    report = dict(
        id=key,
        glb_sha256=evidence['glb_sha256'],
        road_mesh_margin_m=margin,
        fit_image_border_guard_px=frame_guard,
        along_row_spacing_m=gaps,
        components=pieces,
    )
    (dest / 'index.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(
        f'{len(pieces)} openings, {sum("file" in p for p in pieces.values())} complete component GLBs at {dest}'
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--rows', type=int, default=1, help='Number of observed rows, along local X')
    p.add_argument(
        '--margin', type=float, default=0.035, help='Width of accompanying measured pavement, metres'
    )
    a = p.parse_args()
    if not np.isfinite(a.margin) or a.margin <= 0:
        p.error('Use a positive road margin')
    components(a.run, a.id, a.rows, a.margin)
