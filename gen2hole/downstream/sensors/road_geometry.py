"""Geometry shared by the ray replay and the vehicle surface queries (metres, Z up)."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import open3d as o3d

PACKAGE_NAMES = ('道路病害图生三维', '道路病害图生三维_新增100')
PARTS = ('background', 'adjoining road', 'fracture bed and lip', 'fracture sidewall',
         'water surface', 'cast ring and seat', 'shaft wall', 'shaft bottom',
         'concrete collar', 'lid face', 'lid underside', 'lid edge', 'hinge',
         'pavement block top', 'pavement block edge', 'pavement block underside', 'subgrade', 'socket wall', 'visible debris face', 'opening inner rim')
EXACT_PARTS = {
    'Fracture inset rim': 19, 'Road around the aperture': 1, 'Road adjoining the fracture': 1,
    'Fracture lip and exposed bed': 2, 'Fracture sidewall in surface coordinates': 3,
    'Water surface': 4, 'Cast ring and cover seat': 5, 'Inspection shaft': 6,
    'Shaft bottom': 7, 'Chipped concrete collar': 8, 'Lid face': 9,
    'Lid underside with casting ribs': 10, 'Lid rolled edge': 11,
    'Lid underside': 10, 'Lid aperture walls and perimeter': 11,
    'Lid fragment face': 9, 'Lid fragment underside': 10, 'Lid fragment aperture walls and perimeter': 11,
    'Detached pavement top': 13, 'Detached pavement edges': 14, 'Detached pavement underside': 15,
    'Exposed slab foundation': 16, 'Subgrade bearing': 16, 'Subgrade bearing surface': 16, 'Slab socket wall': 17,
}
CLASSES = {'pothole': 1, 'missing': 2, 'raised': 3}
SITES = ('034', '109', '110')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for buf in iter(lambda: stream.read(1048576), b''):
            digest.update(buf)
    return digest.hexdigest()


def load_inventory(parent):
    assets, scenes, anchors = {}, {}, {}
    parent = Path(parent)
    # Each gen2hole run declares its editions. Historical releases use the original folder names.
    marker = parent / 'run.json'
    packages = read_json(marker)['editions'] if marker.is_file() else PACKAGE_NAMES
    for name in packages:
        root = Path(parent) / name
        for meta in root.glob('assets/*/*/metadata.json'):
            key = meta.parent.name
            assert key not in assets
            assets[key] = (meta.parent, read_json(meta))
        for path in sorted(root.glob('scenes/*/scene.json')):
            info = read_json(path)
            case = path.parent.name
            assert case not in scenes
            scenes[case] = (path, info)
            for pl in info['placements']:
                group = (info['site'], pl['slot'])
                center, basis = np.array(pl['center_world']), np.array(pl['basis_columns'])
                assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-5) and np.linalg.det(basis) > .99999
                assert pl['size_multiplier'] == 1.
                if group in anchors:
                    old_center, old_basis = anchors[group]
                    assert np.allclose(old_center, center, rtol=0, atol=1e-7) and np.allclose(old_basis, basis, rtol=0, atol=1e-7)
                anchors[group] = center, basis
    if not assets:
        raise ValueError(f"No rebuilt or published asset packages in {parent}")
    return assets, scenes, anchors


def part_id(name):
    if name.startswith('Fracture debris '):
        return 18
    return 12 if name.startswith('Hinge ') else EXACT_PARTS[name]


def glb_triangles(path, *, names=False, appearance=False, normal_map=False):
    """Read the flat, uncompressed indexed meshes exported by the two reconstruction packages."""
    data = Path(path).read_bytes()
    magic, version, total, jsize, tag = struct.unpack_from('<4sIIII', data)
    if (magic, version, total, tag) != (b'glTF', 2, len(data), 0x4e4f534a):
        raise ValueError(f'Not a glTF 2 binary file: {path}')
    end = 20 + jsize
    gltf = json.loads(data[20:end])
    blen, btag = struct.unpack_from('<II', data, end)
    assert btag == 0x004e4942 and end+8+blen == len(data)
    binary = memoryview(data)[end+8:]

    def accessor(i):
        a = gltf['accessors'][i]
        view = gltf['bufferViews'][a['bufferView']]
        dtype = {5121:'u1', 5123:'<u2', 5125:'<u4', 5126:'<f4'}[a['componentType']]
        size = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[a['type']]
        if view.get('byteStride', np.dtype(dtype).itemsize*size) != np.dtype(dtype).itemsize*size or 'sparse' in a:
            raise ValueError('Unexpected non-packed asset accessor')
        return np.frombuffer(binary, dtype=dtype, offset=view.get('byteOffset',0)+a.get('byteOffset',0),
                             count=a['count']*size).reshape(-1, size)

    for i in gltf['scenes'][gltf.get('scene', 0)]['nodes']:
        node = gltf['nodes'][i]
        if set(node) - {'name','mesh','extras'}:
            raise ValueError('Expected world-aligned exported nodes')
        mesh = gltf['meshes'][node['mesh']]
        for primitive in mesh['primitives']:
            assert primitive.get('mode', 4) == 4
            p = accessor(primitive['attributes']['POSITION'])
            v = np.stack((p[:,0], -p[:,2], p[:,1]), axis=-1)
            f = accessor(primitive['indices']).reshape(-1, 3).astype(np.uint32)
            part = (mesh['name'] if names else part_id(mesh['name']), np.ascontiguousarray(v), f)
            if appearance or normal_map:
                uv = accessor(primitive['attributes']['TEXCOORD_0'])
                material = gltf['materials'][primitive['material']]
                tex = material['normalTexture'] if normal_map else material['pbrMetallicRoughness']['baseColorTexture']
                source = gltf['textures'][tex['index']]['source']
                view = gltf['bufferViews'][gltf['images'][source]['bufferView']]
                start = view.get('byteOffset', 0)
                part += (uv, binary[start:start + view['byteLength']])
            yield part


class Surface:
    def __init__(self, path, threads=2, exclude=(), include_names=None):
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
        self.threads = threads
        self.parts = []
        lo, hi = [], []
        for name, v, tri in glb_triangles(path, names=True):
            part = part_id(name)
            if part in exclude or (include_names is not None and name not in include_names):
                continue
            gid = self.scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(tri))
            assert gid == len(self.parts)
            self.parts.append(part)
            lo.append(v.min(axis=0)); hi.append(v.max(axis=0))
        self.parts = np.asarray(self.parts, 'u1')
        self.low, self.high = np.min(lo, axis=0), np.max(hi, axis=0)

    def cast(self, origins, directions):
        rays = np.ascontiguousarray(np.concatenate((origins, directions), axis=-1), dtype='f4')
        hit = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=self.threads)
        dist, ids = hit['t_hit'].numpy(), hit['geometry_ids'].numpy()
        valid = np.isfinite(dist)
        parts = np.zeros(len(dist), 'u1')
        parts[valid] = self.parts[ids[valid]]
        return dist, parts, hit['primitive_normals'].numpy()
