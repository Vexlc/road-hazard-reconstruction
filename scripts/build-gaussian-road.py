"""Build a metre-scale host road with each chamber's actual pavement cutout."""

import base64
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'tools/asset_atlas'
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles, read_json, sha256


def opening(vertices, faces):
    # Weld the texture seam before tracing the two boundaries of the annular pavement mesh.
    xy, ids = np.unique(np.rint(vertices[:, :2] * 1e5).astype('i4'), axis=0, return_inverse=True)
    tris = ids.reshape(-1)[faces]
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges, frequency = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    neighbors = {}
    for a, b in edges[frequency == 1]:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if not all(len(pair) == 2 for pair in neighbors.values()):
        raise ValueError('Pavement boundaries must be closed contours')
    loops = []
    while neighbors:
        start = next(iter(neighbors))
        previous, current = -1, start
        loop = []
        while current in neighbors:
            loop.append(current)
            a, b = neighbors.pop(current)
            previous, current = current, b if a == previous else a
        if current != start:
            raise ValueError('Open pavement boundary')
        loops.append((xy[loop] * 1e-5).astype('f4'))
    if len(loops) != 2:
        raise ValueError('Expected one inner and one outer pavement boundary')
    inner = min(loops, key=cv2.contourArea)
    profile = cv2.approxPolyDP(inner, 0.00025, closed=True)[:, 0]
    if cv2.contourArea(profile, oriented=True) < 0:
        profile = profile[::-1]
    if cv2.pointPolygonTest(profile, (0, 0), False) != 1:
        raise ValueError('Pavement opening does not surround the chamber')
    return np.rint(profile * 1e5).astype('i4').tolist()


def main():
    rows = json.loads((ATLAS / 'catalog.js').read_text().removeprefix('window.roadAssets=')[:-2])
    profiles = {}
    for row in rows:
        meta = read_json(ATLAS / row['metadata']) if row.get('metadata') else {}
        if meta.get('category') not in ('missing', 'raised'):
            continue
        path = ATLAS / row['glb']
        for name, v, f in glb_triangles(path, names=True):
            if name == 'Road around the aperture':
                profiles[row['id']] = dict(glb=sha256(path), xy=opening(v, f))
                break
    survey = read_json(ROOT / 'gen2hole/examples/scanned-cracks/survey.json')['materials']['asphalt_02']
    source = ROOT / 'work/reconstruction-inputs/scanned-pavement/asphalt_02/asphalt_02_diff_2k.jpg'
    if sha256(source) != survey['files']['Diffuse']['sha256']:
        raise ValueError('Changed Poly Haven asphalt colour map')
    with Image.open(source) as full:
        small = full.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
    buf = BytesIO()
    small.save(buf, format='JPEG', quality=88)
    road = dict(
        profiles=profiles,
        tile_m=survey['dimensions_mm'][1] / 1000,
        texture='data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii'),
    )
    target = ATLAS / 'road-host-data.js'
    with NamedTemporaryFile(dir=target.parent, mode='w', encoding='utf-8', delete=False) as stream:
        stream.write('window.gaussianRoad=' + json.dumps(road, separators=(',', ':')) + ';\n')
    temporary = Path(stream.name)
    temporary.chmod(0o644)
    temporary.replace(target)
    print(len(profiles), 'apertures,', target.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
