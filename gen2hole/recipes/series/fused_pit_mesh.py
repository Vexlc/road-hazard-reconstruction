"""Build the native mesh directly from the recovered metric surface and photographed colors."""

import argparse
import json

import numpy as np
from PIL import Image
from mesh_io import Mesh, ROOT, grid_faces


def surface_mesh(field, color, meta, diagnostic=False, region_mode=None):
    h = field['height']
    width = float(field['width'])
    y, x = np.meshgrid(
        np.linspace(width / 2, -width / 2, h.shape[0]),
        np.linspace(-width / 2, width / 2, h.shape[1]),
        indexing='ij',
    )
    xyz = np.stack([x, y, h], -1).reshape(-1, 3)
    uv = np.c_[x.ravel() / width + 0.5, y.ravel() / width + 0.5]
    faces = grid_faces(*h.shape, flip=False)
    if np.cross(xyz[faces[0, 1]] - xyz[faces[0, 0]], xyz[faces[0, 2]] - xyz[faces[0, 0]])[2] < 0:
        faces = faces[:, [0, 2, 1]]
    bed = field['defect_mask'].ravel()[faces].mean(1) > 0.5
    mesh = Mesh()
    if diagnostic:
        mat = mesh.material('Height sample source', color, 0.94, 0.0, 0.0, lossless_color=True)
    else:
        mat = mesh.material('Photographed road surface', color, 0.94, 0.0, 1.8)
    if 'region_ids' in field:
        a, b, c = field['region_ids'].ravel()[faces].T
        code = np.where((a == b) | (a == c), a, np.where(b == c, b, 0))
        parts = [('Road adjoining the fracture', ~bed, (208, 211, 203)),
                 ('Fracture lip and exposed bed', bed & (code == 0), (119, 96, 75))]
        for i, name in enumerate(meta['region_names'], 1):
            hue = (234, 165, 58) if name == 'Fracture inset rim' else ((45, 164, 154) if i%2 else (92, 195, 204))
            parts.append((name, bed & (code == i), hue))
        for i, (name, selected, hue) in enumerate(parts):
            if i < 2 and region_mode == 'isolated':
                continue
            material = mat
            if region_mode == 'labels':
                swatch = np.full((4,4,3), hue, dtype='u1')
                material = mesh.material(name, swatch, .95, 0, 0, lossless_color=True)
            mesh.add(name, xyz, faces[selected], uv, material, road=i==0)
    else:
        mesh.add('Road adjoining the fracture', xyz, faces[~bed], uv, mat, road=True)
        mesh.add('Fracture lip and exposed bed', xyz, faces[bed], uv, mat)
    if 'water' in meta:
        from pit_water import add_water

        meta['water'] = add_water(mesh, xyz, faces, field, color, meta['water'])
    return mesh


def build(key, comparisons_only=False):
    calibration = ROOT / 'calibration'
    meta = json.loads((calibration / (key + '.json')).read_text())
    with np.load(calibration / (key + '.npz'), allow_pickle=False) as data:
        field = dict(data)
    with Image.open(ROOT / 'references' / (key + '_ortho.png')) as image:
        color = np.array(image.convert('RGB'))
    if 'region_ids' in field:
        dest = ROOT / 'comparison' / key
        dest.mkdir(parents=True, exist_ok=True)
        for mode in ('labels','isolated'):
            (dest / (mode + '.glb')).write_bytes(surface_mesh(field, color, meta, region_mode=mode).glb_bytes())
    elif 'support_kind' in field:
        dest = ROOT / 'comparison' / key
        dest.mkdir(parents=True, exist_ok=True)
        with np.load(calibration / (key + '_range_only.npz'), allow_pickle=False) as baseline:
            (dest / 'range-only.glb').write_bytes(surface_mesh(baseline, color, {}).glb_bytes())
        palette = np.uint8([[230, 145, 68], [67, 91, 112], [59, 193, 188]])
        colors = palette[field['support_kind']]
        colors[~field['defect_mask']] = [220, 220, 210]
        mesh = surface_mesh(field, colors, {}, diagnostic=True)
        (dest / 'support.glb').write_bytes(mesh.glb_bytes())
    elif comparisons_only:
        raise ValueError('No comparison meshes for ' + key)
    if comparisons_only:
        return
    mesh = surface_mesh(field, color, meta)
    width = float(field['width'])
    cache = ROOT / 'mesh_cache'
    cache.mkdir(exist_ok=True)
    # The surrounding preview plane also uses the sampled intact-road texture.
    with Image.open(ROOT / 'references' / (key + '_pavement.jpg')) as grain:
        grain.save(cache / (key + '_ground.jpg'), quality=94)
    meta['road_projection'] = [239.5, 239.5, 479 / width, 479 / width, len(field['height'])]
    mesh.save(ROOT / 'assets/pothole' / key, cache, meta, field)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('keys', nargs='+')
    parser.add_argument('--comparisons-only', action='store_true')
    args = parser.parse_args()
    for key in args.keys:
        build(key, args.comparisons_only)
