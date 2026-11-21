"""Build the native mesh directly from the recovered metric surface and photographed colors."""

import json
import sys
import numpy as np
from PIL import Image
from mesh_io import Mesh, ROOT, grid_faces


def build(key):
    calibration = ROOT / 'calibration'
    meta = json.loads((calibration / (key + '.json')).read_text())
    field = dict(np.load(calibration / (key + '.npz'), allow_pickle=False))
    color = np.array(Image.open(ROOT / 'references' / (key + '_ortho.png')).convert('RGB'))
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
    mat = mesh.material('Photographed road surface', color, 0.94, 0.0, 1.8)
    mesh.add('Road adjoining the fracture', xyz, faces[~bed], uv, mat, road=True)
    mesh.add('Fracture lip and exposed bed', xyz, faces[bed], uv, mat)
    cache = ROOT / 'mesh_cache'
    cache.mkdir(exist_ok=True)
    # The surrounding preview plane also uses the sampled intact-road texture.
    with Image.open(ROOT / 'references' / (key + '_pavement.jpg')) as grain:
        grain.save(cache / (key + '_ground.jpg'), quality=94)
    meta['road_projection'] = [239.5, 239.5, 479 / width, 479 / width, len(h)]
    mesh.save(ROOT / 'assets/pothole' / key, cache, meta, field)


if __name__ == '__main__':
    for key in sys.argv[1:]:
        build(key)
