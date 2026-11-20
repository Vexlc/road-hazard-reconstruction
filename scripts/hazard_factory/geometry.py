"""Detailed cover geometry, retaining the existing complete shaft/road builders."""
import json
from pathlib import Path
import sys

import numpy as np


def build(project, recipe_file, output, cache):
    sys.path.insert(0, str(project / 'outputs/road_hazard_expansion/reproduction'))
    import make_variants as m
    cfg = json.loads(recipe_file.read_text(encoding='utf-8'))
    cfg['mesh_runtime'] = {'numpy': np.__version__, 'trimesh': m.trimesh.__version__,
                           'python': sys.version}
    m.TEXTURE_SIZE = 1024
    original_map = m.read_map
    original_resample = m.resample_field
    # Defaults in the legacy functions were bound to 512 at definition time.
    m.read_map = lambda root, material, channel, size=None: original_map(
        root, material, channel, m.TEXTURE_SIZE if size is None else size)
    m.resample_field = lambda values, width=None: original_resample(
        values, m.TEXTURE_SIZE if width is None else width)
    original_disk = m.disk
    original_face = m.cast_face

    def pattern(x, y):
        radius = np.hypot(x, y)
        phi = np.arctan2(y, x)
        p = cfg['seed'] % 4
        if p == 0:
            value = np.clip((.5 + .5*np.sin(69*(x+y))*np.sin(69*(x-y))-.62)/.36, 0, 1)
        elif p == 1:
            value = m.smoothstep((.38 - (np.sin(64*x)**2 + np.sin(63*y)**2))/.34)
        elif p == 2:
            value = m.smoothstep((.86 - (abs(np.sin(18*phi)) + abs(np.sin(31*radius))))/.75)
        else:
            value = m.smoothstep((.63 - (np.sin(48*x)**2 + np.sin(29*y)**2))/.44)
        rr = (abs(x)**cfg['exponent'] + abs(y)**cfg['exponent'])**(1/cfg['exponent'])
        return value*m.smoothstep((.84-rr)/.08)*m.smoothstep((rr-.18)/.09)

    def detailed_disk(mesh, label, profile, theta, z, texture, radius_factor=1,
                      transform=None, reverse=False):
        if label != 'Lid face':
            return original_disk(mesh, label, profile, theta, z, texture,
                                 radius_factor, transform, reverse)
        theta = np.linspace(0, 2*np.pi, 385)
        profile = m.profile_shape(cfg, theta)
        radii = np.linspace(.003, radius_factor, 129)
        xy = radii[:, None, None]*profile
        normalized = xy / (np.max(abs(profile), axis=0)*radius_factor)
        nx, ny = normalized[..., 0], normalized[..., 1]
        wear = cfg['lid_wear']
        modulation = 1 - wear*(.5+.5*np.sin(nx*13.7+cfg['seed']%31)*np.sin(ny*17.3))
        relief = cfg['lid_relief_m']*pattern(nx, ny)*modulation
        # Recessed lifting-key pockets remain real depressed surfaces, not painted holes.
        pockets = np.zeros_like(relief)
        for cx in (-.48, .48):
            pockets = np.maximum(pockets, np.exp(-((nx-cx)/.045)**8-(ny/.10)**8))
        relief -= .0045*pockets
        relief += .00018*np.sin(nx*91+ny*71)*np.sin(nx*107-ny*83)*wear
        points = np.concatenate([xy, (z+relief)[..., None]], axis=-1)
        uv = normalized*.47 + .5
        faces = m.faces_grid(*points.shape[:2])
        n = points.shape[0]*points.shape[1]
        ids = np.arange(len(theta)-1)
        faces = np.r_[faces, np.c_[np.full(len(ids), n), ids, ids+1]]
        vertices = np.r_[points.reshape(-1, 3), [[0., 0., float(z)]]]
        if transform is not None:
            vertices = transform(vertices)
        mesh.add(label, vertices, faces, np.r_[uv.reshape(-1, 2), [[.5, .5]]], texture)
        cfg['lid_geometry_detail'] = {'relief_range_m': [float(relief.min()), float(relief.max())],
                                      'relief_std_m': float(relief.std()),
                                      'top_triangles': len(faces), 'lifting_recess_depth_m': .0045}

    def worn_face(root, config):
        base = original_face(root, config)
        x, y = m.axes(1, m.TEXTURE_SIZE)
        # Low-frequency oxidation, independent of lighting and geometric highlights.
        oxidation = config.get('lid_wear', 0)*(.5+.5*np.sin(x*7.1+config['seed']%37)*np.sin(y*8.7))
        return base*(1-.16*oxidation[..., None]) + oxidation[..., None]*np.array([.022, .009, .003])

    m.disk = detailed_disk
    m.cast_face = worn_face
    asset = m.Asset(cfg['id'], cache)
    if cfg['category'] == 'pothole':
        field = m.build_pit(asset, cfg, project / 'work/road_hazard_sources')
    else:
        field = m.build_manhole(asset, cfg, project / 'work/road_hazard_sources')
    asset.save(output, cfg, field)
    # Preserve part IDs for detail-aware Gaussian allocation (lid face vs hidden base).
    cache_file = cache / (cfg['id']+'.npz')
    with np.load(cache_file) as saved:
        arrays = dict(saved)
    arrays['part_names'] = np.array([p[0] for p in asset.parts])
    arrays['face_part'] = np.concatenate([np.full(len(p[2]), i, dtype='i2')
                                           for i, p in enumerate(asset.parts)])
    np.savez(cache_file, **arrays)


if __name__ == '__main__':
    project, recipe_file, output, cache = map(Path, sys.argv[1:])
    build(project, recipe_file, output, cache)
