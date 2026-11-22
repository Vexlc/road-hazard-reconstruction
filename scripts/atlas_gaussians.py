"""Native CUDA reference renders using the unmodified degree-one Gaussian PLY in each atlas entry."""

import hashlib
from itertools import product
import json
import math
from pathlib import Path
import struct

import numpy as np
from plyfile import PlyData
import torch
from gsplat.rendering import rasterization

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'tools/asset_atlas'
BACKGROUND = np.array([223, 225, 216]) / 255


def catalog():
    text = (ATLAS / 'catalog.js').read_text()
    return {r['id']: r for r in json.loads(text.removeprefix('window.roadAssets=')[:-2])}


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def geometry_frame(row, aspect):
    """Frame the rim, movable panels and crack or pit from their original glTF metre bounds."""
    with (ATLAS / row['glb']).open('rb') as stream:
        header = stream.read(20)
        size = struct.unpack_from('<I', header, 12)[0]
        model = json.loads(stream.read(size))
        meshes = model['meshes']
        focus = [
            m
            for m in meshes
            if m.get('name', '').startswith(('Lid', 'Hinge', 'Fracture', 'Cast ring', 'Water'))
            or 'collar' in m.get('name', '').lower()
        ]
        corners = []
        normal = None
        for mesh in focus or meshes:
            for prim in mesh['primitives']:
                attr = prim['attributes']
                xyz = model['accessors'][attr['POSITION']]
                corners.extend(product(*zip(xyz['min'], xyz['max'])))
                if mesh.get('name') != 'Lid face' or 'NORMAL' not in attr:
                    continue
                acc = model['accessors'][attr['NORMAL']]
                view = model['bufferViews'][acc['bufferView']]
                if (
                    acc['type'] == 'VEC3'
                    and acc['componentType'] == 5126
                    and view.get('byteStride', 12) == 12
                ):
                    stream.seek(28 + size + view.get('byteOffset', 0) + acc.get('byteOffset', 0))
                    normals = np.frombuffer(stream.read(acc['count'] * 12), '<f4').reshape(-1, 3)
                    normal = normals.mean(axis=0)
    corners = np.array(corners)
    theta = -18.0 if row['category'] in ('坑洞', '裂缝') else 18.0
    if normal is not None and np.hypot(normal[0], normal[2]) > 0.3:
        theta = math.degrees(math.atan2(normal[0], normal[2])) + 24
    phi = 42.0 if row['category'] in ('坑洞', '裂缝') else 49.0
    center = (corners.min(0) + corners.max(0)) * 0.5
    camera = dict(
        theta=math.radians(theta), phi=math.radians(phi), target=center.tolist(), radius=1.0, fov=30.0
    )
    view, _ = projection(camera, 1700, 1000)
    local = (corners - center)[:, [0, 2, 1]] * [1, -1, 1]
    xyz = local @ view[:3, :3].T
    slope = np.tan(np.deg2rad(camera['fov'] / 2)) * 0.85
    distance = np.maximum(np.abs(xyz[:, 0]) / (slope * aspect), np.abs(xyz[:, 1]) / slope) - xyz[:, 2]
    camera['radius'] = max(0.05, float(distance.max()))
    return camera


def projection(camera, width, height):
    # First form camera-to-world in glTF Y-up. Then change basis to the local PLY Z-up frame.
    t, p, rad = camera['theta'], camera['phi'], camera['radius']
    away = np.array([np.sin(t) * np.sin(p), np.cos(p), np.cos(t) * np.sin(p)])
    right = np.array([np.cos(t), 0, -np.sin(t)])
    gltf_to_ply = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c2w = np.eye(4)
    c2w[:3, :3] = gltf_to_ply @ np.column_stack([right, np.cross(right, away), -away])
    c2w[:3, 3] = gltf_to_ply @ (np.asarray(camera['target']) + rad * away)
    f = 0.5 * height / np.tan(np.deg2rad(camera['fov'] / 2))
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return np.linalg.inv(c2w), K


def gpu(data):
    return torch.as_tensor(np.ascontiguousarray(data), device='cuda', dtype=torch.float32)


def load(row):
    vertices = PlyData.read(ATLAS / row['ply'], mmap='r')['vertex'].data
    if len(vertices) != row['gaussians']:
        raise ValueError('Inventory count differs from PLY for ' + row['id'])

    def cols(keys):
        return np.stack([vertices[name] for name in keys], axis=1)

    def block(stem, n):
        return cols([stem + str(k) for k in range(n)])

    features = np.concatenate(
        [block('f_dc_', 3)[:, None, :], block('f_rest_', 9).reshape(-1, 3, 3).transpose(0, 2, 1)], axis=1
    )
    state = dict(
        means=gpu(cols('xyz')),
        quats=gpu(block('rot_', 4)),
        scales=gpu(block('scale_', 3)).exp(),
        opacities=gpu(vertices['opacity'].copy()).sigmoid(),
        colors=gpu(features),
    )
    if any(not torch.isfinite(t).all() for t in state.values()) or (state['quats'].norm(dim=1) < 1e-7).any():
        raise ValueError('Invalid rendering parameter in ' + row['id'])
    return state


@torch.inference_mode()
def render(state, camera, width, height):
    view, K = projection(camera, width, height)
    image, alpha, _ = rasterization(
        **state,
        viewmats=gpu(view)[None],
        Ks=gpu(K)[None],
        width=width,
        height=height,
        sh_degree=1,
        near_plane=0.001,
        far_plane=100.0,
        packed=False,
        rasterize_mode='antialiased',
        render_mode='RGB+ED',
        backgrounds=gpu(BACKGROUND)[None],
    )
    image = image[0].cpu().numpy()
    alpha = alpha[0, :, :, 0].cpu().numpy()
    if not np.isfinite(image).all() or not np.isfinite(alpha).all() or not np.any(alpha > 0.5):
        raise ValueError('No finite foreground in the rendered view')
    return (np.clip(image[:, :, :3], 0, 1) * 255 + 0.5).astype('u1'), alpha, image[:, :, 3]
