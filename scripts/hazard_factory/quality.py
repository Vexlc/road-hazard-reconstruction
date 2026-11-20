"""Independent geometry, serialized asset and provenance checks."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(project, directory, cache, expected):
    sys.path.insert(0, str(project / 'outputs/road_hazard_expansion/reproduction'))
    from verify_library import audit
    # The old auditor also checks GLB textures, bounds, quaternion units and PLY finiteness.
    result = audit((directory.parent.parent, directory / 'metadata.json'))
    if result['status'] != 'validated':
        raise ValueError(result)
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'recipe differs at {key}')
    if meta['size_multiplier'] != 1. or meta['scale_kind'] != 'physical_design':
        raise ValueError('demonstration scaling cannot enter the physical design library')
    with np.load(cache / (meta['id']+'.npz')) as mesh:
        vertices, faces = mesh['xyz'], mesh['faces']
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                        triangles[:, 2]-triangles[:, 0]), axis=1)*.5
        bad = float(np.mean(areas < 1e-12))
        if bad > .02 or not np.isfinite(vertices).all():
            raise ValueError(f'degenerate triangle fraction {bad}')
        geometry_hash = hashlib.sha256(vertices.tobytes()+faces.tobytes()+mesh['uv'].tobytes()).hexdigest()
        area = float(areas.sum())
    with np.load(directory / 'surface.npz') as field:
        center = field['height'][field['alpha'] > .95]
        depth = float(-center.min())
        edge = field['height'][field['alpha'] < .001]
        edge_height = float(np.max(abs(edge))) if len(edge) else 0.
        if not .25*meta['depth_design_m'] < depth < 1.6*meta['depth_design_m']:
            raise ValueError(f'depth outside design tolerance: {depth}')
        if edge_height > .001:
            raise ValueError(f'nonflat road seam: {edge_height}')
    if meta['category'] == 'raised':
        detail = meta['lid_geometry_detail']
        if detail['top_triangles'] < 90000 or detail['relief_std_m'] < .0001:
            raise ValueError('cover geometric relief missing')
    ply = PlyData.read(directory / 'asset.ply', mmap='r')['vertex'].data
    q50 = float(np.median(np.exp(ply['scale_0'])))
    if q50 > .010:
        raise ValueError(f'coarse Gaussian footprint {q50}')
    return {'passed': True, 'geometry_sha256': geometry_hash, 'mesh_area_m2': area,
            'degenerate_triangle_fraction': bad, 'deepest_surface_m': depth,
            'outer_seam_max_height_m': edge_height, 'gaussian_tangent_sigma_median_m': q50,
            'glb_sha256': result['glb_sha256'], 'ply_sha256': result['ply_sha256'],
            'gaussians': meta['gaussians'], 'appearance_sh_fit_mae': meta['appearance_sh_fit_mae']}


if __name__ == '__main__':
    project, directory, cache, recipe = map(Path, sys.argv[1:])
    report = validate(project, directory, cache, json.loads(recipe.read_text(encoding='utf-8')))
    (directory / 'quality.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
